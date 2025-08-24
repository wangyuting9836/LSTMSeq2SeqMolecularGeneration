import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, lstm_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, lstm_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        """
        x: (B, T, V)
        返回 (n_layers, B, lstm_dim)
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n  # (n_layers, B, lstm_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size, lstm_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, lstm_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_dim, vocab_size)

    def forward(self, x, h, c):
        """
        x: (B, T, V)
        h: (n_layers, B, lstm_dim)
        c: (n_layers, B, lstm_dim)
        返回
            logits: (B, T, V),
            (h, c): ((n_layers, B, lstm_dim), (n_layers, B, lstm_dim))
        """
        output, (h, c) = self.lstm(x, (h, c)) # output: (B, T, lstm_dim)  h/c: (n_layers, B, lstm_dim)
        logits = self.fc(output)  # (B, T, V)
        return logits, h, c


class SMILESAutoencoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            lstm_dim=512,
            latent_dim=256,
            n_layers=1,
            dropout=0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        # 解码器
        self.encoder = Encoder(vocab_size, lstm_dim, n_layers, dropout)
        # 瓶颈层：把 n_layers*lstm_dim*2 -> latent_dim
        self.bottleneck = nn.Linear(2 * n_layers * lstm_dim, latent_dim)
        # 恢复：latent_dim -> n_layers*lstm_dim
        self.latent2hidden = nn.Linear(latent_dim, n_layers * lstm_dim)
        self.latent2cell = nn.Linear(latent_dim, n_layers * lstm_dim)
        # 编码器
        self.decoder = Decoder(vocab_size, lstm_dim, n_layers, dropout)

    def encode(self, x):
        h_n, c_n = self.encoder(x)  # (n_layers, B, lstm_dim)
        # 展平层维
        states = torch.cat([h_n, c_n], dim=0)  # (2*n_layers, B, lstm_dim)
        states = states.permute(1, 0, 2).contiguous()  # (B, 2*n_layers, lstm_dim)
        states = states.view(states.size(0), -1)  # (B, 2*n_layers*lstm_dim)
        z = torch.relu(self.bottleneck(states))  # (B, latent_dim)
        return z

    def decode(self, z, target, teacher_forcing=True):
        """
        z: (B, latent_dim)
        target: (B, T, V)  用于teacher forcing
        返回  (B, T - 1, V)
        """
        B = z.size(0)
        h = torch.relu(self.latent2hidden(z))  # (B, n_layers*lstm_dim)
        c = torch.relu(self.latent2cell(z))  # (B, n_layers*lstm_dim)

        # reshape -> (n_layers, B, lstm_dim)
        h = h.view(B, self.n_layers, -1).permute(1, 0, 2).contiguous()
        c = c.view(B, self.n_layers, -1).permute(1, 0, 2).contiguous()

        if teacher_forcing:
            logits, _, _ = self.decoder(target[:, :-1, :], h, c)
        else:
            # 自回归生成
            logits = []
            inp = target[:, :1, :]  # <go>
            for t in range(target.size(1) - 1):
                out, h, c = self.decoder(inp, h, c)
                logits.append(out)
                inp = target[:, t + 1: t + 2, :]
            logits = torch.cat(logits, dim=1)
        return logits

    def forward(self, x, y, teacher_forcing=True):
        """
        x: (B, T, V)
        y: (B, T, V)  用于teacher forcing
        返回
            logits: (B, T - 1, V)
            z: (B, latent_dim) 潜空间向量
        """
        z = self.encode(x)
        logits = self.decode(z, y, teacher_forcing)
        return logits, z
