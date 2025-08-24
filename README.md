# 用 LSTM Seq2Seq 从零搭一个「分子生成器」

**Author:** Yuting Wang

**Date:** 2025-08-24

**Link:** https://zhuanlan.zhihu.com/p/1942879596027552982



目录

收起

0\. 为什么用 Seq2Seq 生成分子？

1\. 数据集准备

2\. 根据数据集构建词汇表， 把SMILES 变成one-hot张量

3\. 模型

4\. 训练验证

5\. 随机从潜空间采样生成分子

6\. 潜空间插值生成分子

## 0\. 为什么用 Seq2Seq 生成分子？

Seq2Seq 把 SMILES 当成普通字符串：

**编码器** → 把整个分子压缩成潜空间向量 **z**；

**解码器** → 从 潜空间向量**z** 一步一步吐出字符，直到 `<eos>`。

只要训练得当，解码器就能学会 SMILES 语法，甚至化学合理性。

## 1\. 数据集准备

这里使用GDB Database。

GDB-11数据库通过应用简单的化学稳定性与合成可行性规则，枚举了所有由最多11个碳、氮、氧、氟原子组成的有机小分子。

GDB-13数据库通过应用简单的化学稳定性与合成可行性规则，枚举了所有由最多13个碳、氮、氧、硫、氯原子组成的有机小分子。该数据库拥有977,468,314个分子结构，是迄今为止全球最大的公开有机小分子数据库。

链接如下。

[GDB Databases​gdb.unibe.ch/downloads/](https://link.zhihu.com/?target=https%3A//gdb.unibe.ch/downloads/)

## 2\. 根据数据集构建词汇表， 把SMILES 变成one-hot张量

```text
train_ds, test_ds, char2idx, idx2char = load_and_split_data("gdb11/gdb11_size08.smi", sample_size=None)
```

函数实现如下。

```python
def build_vocabulary(smiles_list):
    """根据SMILES列表构建词汇表与映射字典"""
    chars = set("".join(smiles_list))
    chars.update([GO, EOS])
    chars = sorted(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def vectorize(smiles_list, char2idx, max_len=None):
    """
    将SMILES字符串列表转化为one-hot张量
    返回 (X, Y)
    """
    if max_len is None:
        max_len = max(len(s) for s in smiles_list) + 5

    vocab_size = len(char2idx)
    X = np.zeros((len(smiles_list), max_len, vocab_size), dtype=np.float32)
    Y = np.zeros((len(smiles_list), max_len, vocab_size), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        # X: 以<go>开头
        X[i, 0, char2idx[GO]] = 1
        for t, c in enumerate(smi):
            if t + 1 >= max_len:
                break
            X[i, t + 1, char2idx[c]] = 1
        # 剩余位置填<EOS>
        X[i, len(smi) + 1:, char2idx[EOS]] = 1

        # Y: 从smi第一个字符开始，最后以<EOS>结尾
        for t, c in enumerate(smi):
            if t >= max_len:
                break
            Y[i, t, char2idx[c]] = 1
        Y[i, len(smi):, char2idx[EOS]] = 1

    return torch.tensor(X), torch.tensor(Y)

def load_and_split_data(smi_path, train_ratio=0.8, max_len=None, sample_size=None):
    """
    读取.smi文件并划分训练/测试集
    返回 (train_ds, test_ds, char2idx, idx2char)
    """
    data = pd.read_csv(smi_path, sep="\t", header=None, names=["smiles", "No", "Int"])
    smiles = data["smiles"].tolist()
    if sample_size:
        smiles = smiles[:sample_size]

    char2idx, idx2char = build_vocabulary(smiles)
    X, Y = vectorize(smiles, char2idx, max_len)

    dataset = TensorDataset(X, Y)
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    train_ds, test_ds = random_split(
        dataset,
        [n_train, n_total - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds, test_ds, char2idx, idx2char
```

## 3\. 模型

| 模块 | 作用 |
| --- | --- |
| Encoder | nn.LSTM → (h, c)，层数=2，隐藏维=512 |
| Encoder | Linear(2*n_layers*512 → 256)，ReLU |
| Decoder | nn.LSTM → (h, c)，层数=2，隐藏维=512 |

```python
model = SMILESAutoencoder(
    vocab_size=len(char2idx),
    lstm_dim=512,
    latent_dim=256,
    n_layers=2,
    dropout=0.1)
```

具体代码如下，加入了teacher\_forcing开关。理论上teacher\_forcing应该收敛更快，没有测试。

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, lstm_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, lstm_dim, n_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        # x: (B, T, V)
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n  # (n_layers, B, lstm_dim)

class Decoder(nn.Module):
    def __init__(self, vocab_size, lstm_dim, n_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, lstm_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_dim, vocab_size)

    def forward(self, x, h, c):
        # x: (B, T, V)  h/c: (n_layers, B, lstm_dim)
        output, (h, c) = self.lstm(x, (h, c))
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
        # 编码器
        self.encoder = Encoder(vocab_size, lstm_dim, n_layers, dropout)
        # 瓶颈层：把 n_layers*lstm_dim*2 -> latent_dim
        self.bottleneck = nn.Linear(2 * n_layers * lstm_dim, latent_dim)
        # 恢复：latent_dim -> n_layers*lstm_dim
        self.latent2hidden = nn.Linear(latent_dim, n_layers * lstm_dim)
        self.latent2cell = nn.Linear(latent_dim, n_layers * lstm_dim)
        # 解码器
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
        z = self.encode(x)
        logits = self.decode(z, y, teacher_forcing)
        return logits, z
```

## 4\. 训练验证

-   **Teacher Forcing**：解码器每一步都看到真值，收敛更快。
-   **ReduceLROnPlateau**：验证集 10 个 epoch 不下降 → 学习率 × 0.5。
-   **Early Stopping**：验证集 10 个 epoch 不下降 → 直接停。

```python
train(model, train_loader, test_loader, epochs=50, lr=1e-3, patience=10)
```

具体实现如下。

```python
def train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=50,
        lr=1e-3,
        save_path="best_model.pth",
        patience=10,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, y, teacher_forcing=True)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:, :].reshape(-1, y.size(-1)).argmax(dim=-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x, y, teacher_forcing=True)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:, :].reshape(-1, y.size(-1)).argmax(dim=-1),
                )
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"Train loss: {avg_train:.4f}  Val loss: {avg_val:.4f}"
        )

        # 保存最佳模型
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
```

## 5\. 随机从潜空间采样生成分子

```python
# 4. 加载最佳模型并生成分子
model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=device)) # 加载最佳模型

z_rand = torch.randn(1, 256, device=device) # 随机从潜在空间采样
smi = generate_from_latent(model, z_rand, char2idx, idx2char, temperature=1.0)
print("Random generation:", smi)
visualize_one_smiles(smi) # 可视化
```

具体实现如下。

```Python
@torch.no_grad()
def generate_from_latent(model, z, char2idx, idx2char, max_len=50, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    vocab_size = len(char2idx)

    n_layers = model.n_layers  # 取得层数

    # 1. 把潜在向量 z -> h, c
    h = torch.relu(model.latent2hidden(z))  # (1, n_layers*lstm_dim)
    c = torch.relu(model.latent2cell(z))

    # 2. reshape -> (n_layers, 1, lstm_dim)
    h = h.view(1, n_layers, -1).permute(1, 0, 2).contiguous()
    c = c.view(1, n_layers, -1).permute(1, 0, 2).contiguous()

    # 3. 生成
    inp = torch.zeros(1, 1, vocab_size, device=device)
    inp[0, 0, char2idx["<go>"]] = 1.0
    smiles = ""

    for _ in range(max_len):
        logits, h, c = model.decoder(inp, h, c)
        logits = logits.squeeze(0) / temperature  # (1, V)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        next_char = idx2char[next_id]
        if next_char == "<eos>":
            break
        smiles += next_char
        inp.zero_()
        inp[0, 0, next_id] = 1.0
    return smiles

def visualize_one_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        Draw.ShowMol(mol, size=(300, 300), kekulize=False)
    else:
        print("Invalid molecule to display.")
```

生成的分子如下。

![](https://pic4.zhimg.com/v2-d54e5ba9c3ed3c83cbfe159c05ed3935_1440w.jpg)

**温度采样机制：**解释一下上面temperature的作用。temperature直接影响 softmax 输出的概率分布，从而影响torch.multinomial(probs, 1)采样的结果，从而控制生成分子的“保守 vs. 冒险”程度。

记上面decoder的输出logits为 $o$ ，则

$p_i=\frac{exp(\frac{o_i}{T})}{\varSigma _jexp(\frac{o_j}{T})}$

-   $T=1$ ：原始 softmax，不做任何缩放。
-   $T > 1$ ：（高温） → 概率更均匀，低分字符也有机会被采样 → 多样性增加、有效性降低。
-   $T < 1$ ：（低温） → 概率更尖锐，高分字符优势更强 → 多样性降低、有效性增加。

## 6\. 潜空间插值生成分子

随机选取两个分子，通过解码器映射到潜空间，然后插值生成十个潜空间向量 **z**，在利用上面的generate\_from\_latent生成分子。这可以让已有的分子“变形”。

```Python
with torch.no_grad():
    idx1, idx2 = random.sample(range(len(test_ds)), 2)
    x1, _ = test_ds[idx1]
    x2, _ = test_ds[idx2]
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    z1 = model.encode(x1)
    z2 = model.encode(x2)

inter_smiles = interpolate(model, z1.squeeze(0), z2.squeeze(0), n_steps=10, char2idx=char2idx, idx2char=idx2char)
print("Interpolation smiles:", inter_smiles)
visualize_smiles(inter_smiles) # 可视化

def interpolate(model, z1, z2, n_steps=10, **kwargs):
    alphas = np.linspace(0, 1, n_steps)
    smiles_list = []
    for alpha in alphas:
        z = alpha * z2 + (1 - alpha) * z1
        smi = generate_from_latent(model, z.unsqueeze(0), **kwargs)
        smiles_list.append(smi)
    return smiles_list
```

生成的分子如下。

![](https://pic3.zhimg.com/v2-bc3e02ac129b49b981f445dd2dd72002_1440w.jpg)
