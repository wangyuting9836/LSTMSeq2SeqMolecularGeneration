import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os


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
