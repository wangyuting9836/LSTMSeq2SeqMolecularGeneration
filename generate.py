import torch
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np


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


def interpolate(model, z1, z2, n_steps=10, **kwargs):
    alphas = np.linspace(0, 1, n_steps)
    smiles_list = []
    for alpha in alphas:
        z = alpha * z2 + (1 - alpha) * z1
        smi = generate_from_latent(model, z.unsqueeze(0), **kwargs)
        smiles_list.append(smi)
    return smiles_list


def visualize_one_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        Draw.MolToFile(mol, "molecule.png", size=(300, 300), kekulize=True)
        Draw.MolToImage(mol, size=(300, 300), kekulize=True)
        Draw.ShowMol(mol, size=(300, 300), kekulize=False)
    else:
        print("Invalid molecule to display.")


def visualize_smiles(smiles_list, mols_per_row=5):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row)
        img.show()
    else:
        print("No valid molecules to display.")
