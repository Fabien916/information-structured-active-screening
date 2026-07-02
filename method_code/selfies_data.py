#!/usr/bin/env python3
import csv
import random
import re

import selfies
from rdkit import Chem
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]

# Valid two-character element symbols in SMILES (e.g., Cl, Br, Si).
# We only merge uppercase+lowercase when the pair is a real element symbol.
TWO_CHAR_ELEMENTS = {
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au",
    "Ba", "Be", "Bh", "Bi", "Bk", "Br",
    "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Cn", "Co", "Cr", "Cs", "Cu",
    "Db", "Ds", "Dy",
    "Er", "Es", "Eu",
    "Fe", "Fl", "Fm", "Fr",
    "Ga", "Gd", "Ge",
    "He", "Hf", "Hg", "Ho", "Hs",
    "In", "Ir",
    "Kr",
    "La", "Li", "Lr", "Lu", "Lv",
    "Mc", "Md", "Mg", "Mn", "Mo", "Mt",
    "Na", "Nb", "Nd", "Ne", "Nh", "Ni", "No", "Np",
    "Og", "Os",
    "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu",
    "Ra", "Rb", "Re", "Rf", "Rg", "Rh", "Rn", "Ru",
    "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr",
    "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "Ts",
    "Xe",
    "Yb",
    "Zn", "Zr",
}

# Aromatic two-character symbols that may appear outside brackets.
AROMATIC_TWO_CHAR_ELEMENTS = {"as", "se"}


BRACKET_ATOM_RE = re.compile(
    r"^(?P<isotope>\d+)?"
    r"(?P<symbol>\*|[A-Z][a-z]?|[a-z])"
    r"(?P<chiral>@@?|)?"
    r"(?P<hydrogen>H\d*)?"
    r"(?P<charge>(?:[-+]{1,3}|[+-]\d+)?)"
    r"(?P<class>:\d+)?$"
)


def pick_smiles_column(fieldnames, preferred):
    if preferred in fieldnames:
        return preferred
    for alt in ["SMILES_clean", "SMILES"]:
        if alt in fieldnames:
            return alt
    raise ValueError(f"SMILES column '{preferred}' not found in CSV header.")


def to_selfies(smiles: str) -> str:
    try:
        return selfies.encoder(smiles)
    except Exception:
        return ""


def to_smiles(selfies_str: str) -> str:
    try:
        return selfies.decoder(selfies_str)
    except Exception:
        return ""


def canonicalize_mol(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return canonicalize_mol(mol)


def tokenize_selfies(selfies_str: str):
    try:
        return list(selfies.split_selfies(selfies_str))
    except Exception:
        return []


def _tokenize_bracket_atom(content: str):
    match = BRACKET_ATOM_RE.fullmatch(content)
    if match is None:
        raise ValueError(f"Unsupported bracket atom token: [{content}]")
    tokens = ["["]
    for key in ["isotope", "symbol", "chiral", "hydrogen", "charge", "class"]:
        val = match.group(key)
        if val:
            tokens.append(val)
    tokens.append("]")
    return tokens


def tokenize_smiles(smiles_str: str):
    tokens = []
    i = 0
    while i < len(smiles_str):
        ch = smiles_str[i]
        if ch == "[":
            j = smiles_str.find("]", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed bracket atom in SMILES: {smiles_str}")
            tokens.extend(_tokenize_bracket_atom(smiles_str[i + 1 : j]))
            i = j + 1
            continue
        if ch == "%" and i + 2 < len(smiles_str) and smiles_str[i + 1 : i + 3].isdigit():
            tokens.append(smiles_str[i : i + 3])
            i += 3
            continue
        if ch.isupper():
            if i + 1 < len(smiles_str) and smiles_str[i + 1].islower():
                pair = smiles_str[i : i + 2]
                if pair in TWO_CHAR_ELEMENTS:
                    tokens.append(pair)
                    i += 2
                    continue
            tokens.append(ch)
            i += 1
            continue
        if ch.islower():
            if i + 1 < len(smiles_str):
                pair = smiles_str[i : i + 2]
                if pair in AROMATIC_TWO_CHAR_ELEMENTS:
                    tokens.append(pair)
                    i += 2
                    continue
            tokens.append(ch)
            i += 1
            continue
        if ch in "\\#()-.+/=0123456789:":
            tokens.append(ch)
            i += 1
            continue
        raise ValueError(f"Unsupported SMILES token '{ch}' in: {smiles_str}")
    return tokens


def detokenize_tokens(tokens):
    return "".join(tokens)


def tokens_to_smiles(token_type: str, tokens):
    seq = detokenize_tokens(tokens)
    if token_type == "selfies":
        return seq, to_smiles(seq)
    if token_type == "smiles":
        return seq, seq
    raise ValueError(f"Unsupported token_type: {token_type}")


class TokenVocab:
    def __init__(self, token_sequences):
        tokens = set()
        for seq in token_sequences:
            tokens.update(seq)
        self.tokens = SPECIAL_TOKENS + sorted(tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.start_id = self.token_to_id["<s>"]
        self.end_id = self.token_to_id["</s>"]
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def from_tokens(cls, tokens):
        vocab = cls.__new__(cls)
        vocab.tokens = list(tokens)
        vocab.token_to_id = {t: i for i, t in enumerate(vocab.tokens)}
        vocab.id_to_token = {i: t for t, i in vocab.token_to_id.items()}
        vocab.pad_id = vocab.token_to_id["<pad>"]
        vocab.start_id = vocab.token_to_id["<s>"]
        vocab.end_id = vocab.token_to_id["</s>"]
        vocab.unk_id = vocab.token_to_id["<unk>"]
        return vocab

    def encode(self, tokens):
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.id_to_token.get(i, "<unk>") for i in ids]


class GraphSelfiesDataset(Dataset):
    def __init__(
        self,
        csv_path,
        smiles_column,
        input_is_selfies,
        token_type,
        min_len,
        max_len,
        max_degree,
        atom_types=None,
    ):
        self.csv_path = csv_path
        self.smiles_column = smiles_column
        self.input_is_selfies = input_is_selfies
        self.token_type = token_type
        self.min_len = min_len
        self.max_len = max_len
        self.max_degree = max_degree

        self.smiles_list, self.used_smiles_column = self._load_smiles()
        self.mols, self.invalid_molecules = self._collect_molecules()
        if atom_types is None:
            self.atom_types = self._collect_atom_types()
        else:
            if not isinstance(atom_types, (list, tuple)) or len(atom_types) == 0:
                raise ValueError("atom_types must be a non-empty list/tuple when provided")
            self.atom_types = sorted(set(str(a) for a in atom_types))
        self.atom_type_to_idx = {a: i for i, a in enumerate(self.atom_types)}
        self.samples, self.sample_smiles, self.stats = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_token_sequences(self):
        return [tokens for _, tokens in self.samples]

    def split(self, train_split, seed):
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        cut = int(len(indices) * train_split)
        train_idx = indices[:cut]
        val_idx = indices[cut:]
        return Subset(self, train_idx), Subset(self, val_idx)

    def _load_smiles(self):
        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header.")
            col = pick_smiles_column(reader.fieldnames, self.smiles_column)
            smiles_list = []
            for row in reader:
                s = (row.get(col) or "").strip()
                if s:
                    smiles_list.append(s)
            return smiles_list, col

    def _collect_molecules(self):
        mols = []
        invalid = 0
        for s in self.smiles_list:
            smiles = to_smiles(s) if self.input_is_selfies else s
            if not smiles:
                invalid += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid += 1
                continue
            mols.append((s, mol))
        return mols, invalid

    def _collect_atom_types(self):
        atoms = set()
        for _, mol in self.mols:
            for atom in mol.GetAtoms():
                atoms.add(atom.GetSymbol())
        return sorted(atoms)

    def _mol_to_graph(self, mol):
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        features = []
        for atom in mol.GetAtoms():
            elem = [0.0] * len(self.atom_type_to_idx)
            idx = self.atom_type_to_idx.get(atom.GetSymbol())
            if idx is None:
                return None
            elem[idx] = 1.0
            degree = min(atom.GetTotalDegree(), self.max_degree)
            degree_onehot = [0.0] * (self.max_degree + 1)
            degree_onehot[degree] = 1.0
            aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
            formal_charge = [float(atom.GetFormalCharge())]
            in_ring = [1.0 if atom.IsInRing() else 0.0]
            features.append(elem + degree_onehot + aromatic + formal_charge + in_ring)
        x = torch.tensor(features, dtype=torch.float)

        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def _build_samples(self):
        samples = []
        sample_smiles = []
        too_short = 0
        too_long = 0
        invalid_repr = 0
        invalid_graph = 0

        for src, mol in self.mols:
            if self.token_type == "selfies":
                if self.input_is_selfies:
                    selfies_str = src
                else:
                    selfies_str = to_selfies(src)
                if not selfies_str:
                    invalid_repr += 1
                    continue
                tokens = tokenize_selfies(selfies_str)
            elif self.token_type == "smiles":
                smiles_str = to_smiles(src) if self.input_is_selfies else src
                smiles_str = smiles_str.strip()
                if not smiles_str:
                    invalid_repr += 1
                    continue
                tokens = tokenize_smiles(smiles_str)
            else:
                raise ValueError(f"Unsupported token_type: {self.token_type}")
            if not tokens:
                invalid_repr += 1
                continue
            if len(tokens) < self.min_len:
                too_short += 1
                continue
            if len(tokens) > self.max_len:
                too_long += 1
                continue
            graph = self._mol_to_graph(mol)
            if graph is None:
                invalid_graph += 1
                continue
            samples.append((graph, tokens))
            sample_smiles.append(canonicalize_mol(mol))

        stats = {
            "too_short": too_short,
            "too_long": too_long,
            "invalid_repr": invalid_repr,
            "invalid_graph": invalid_graph,
        }
        return samples, sample_smiles, stats


def collate_fn(batch, vocab):
    graphs, token_lists = zip(*batch)
    graph_batch = Batch.from_data_list(graphs)
    seqs = []
    for tokens in token_lists:
        ids = [vocab.start_id] + vocab.encode(tokens) + [vocab.end_id]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    seq_batch = pad_sequence(seqs, batch_first=True, padding_value=vocab.pad_id)
    return graph_batch, seq_batch
