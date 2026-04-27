from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
from rdkit import Chem

@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor

# Feature helpers
def _one_hot(value, choices) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]

def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs(includeNeighbors=True)
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    in_ring = 1.0 if atom.IsInRing() else 0.0

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_oh = _one_hot(hyb, hyb_choices)

    # Chirality
    chiral = atom.GetChiralTag()
    chiral_choices = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    chiral_oh = _one_hot(chiral, chiral_choices)

    # Some extra structural signals
    implicit_valence = atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)
    explicit_valence = atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
    num_radical = atom.GetNumRadicalElectrons()
    mass = atom.GetMass()

    return [
        float(atomic_num),
        float(degree),
        float(formal_charge),
        float(total_h),
        float(implicit_valence),
        float(explicit_valence),
        float(num_radical),
        float(mass),
        is_aromatic,
        in_ring,
        *hyb_oh,       
        *chiral_oh,  
    ]

def bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    bt_choices = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bt_oh = _one_hot(bt, bt_choices)  # 4

    is_conj = 1.0 if bond.GetIsConjugated() else 0.0
    is_ring = 1.0 if bond.IsInRing() else 0.0

    # Stereo
    stereo = bond.GetStereo()
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    stereo_oh = _one_hot(stereo, stereo_choices)

    return [
        *bt_oh,
        is_conj,
        is_ring,
        *stereo_oh,
    ]

# Graph conversion
def smiles_to_graph(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)

        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)

    return x, edge_index, edge_attr

def batch_graphs(graph_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Graph:
    xs, eis, eas, batch = [], [], [], []
    node_offset = 0

    for g_idx, (x, ei, ea) in enumerate(graph_list):
        n = x.size(0)
        xs.append(x)

        if ei.numel() > 0:
            eis.append(ei + node_offset)
            eas.append(ea)

        batch.append(torch.full((n,), g_idx, dtype=torch.long))
        node_offset += n

    X = torch.cat(xs, dim=0)
    B = torch.cat(batch, dim=0)

    if len(eis) == 0:
        EI = torch.zeros((2, 0), dtype=torch.long)
        EA = torch.zeros((0, 12), dtype=torch.float32)
    else:
        EI = torch.cat(eis, dim=1)
        EA = torch.cat(eas, dim=0)

    return Graph(x=X, edge_index=EI, edge_attr=EA, batch=B)