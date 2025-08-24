import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

# 特殊标记
GO = "<go>"
EOS = "<eos>"


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


def indices_to_smiles(indices, idx2char):
    """将索引序列转为SMILES字符串"""
    chars = []
    for idx in indices:
        c = idx2char[idx.item()]
        if c == EOS:
            break
        if c != GO:
            chars.append(c)
    return "".join(chars)
