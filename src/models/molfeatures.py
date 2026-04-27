from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def morgan_dense(smiles: str, radius: int, nbits: int, gen=None) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if gen is None:
        gen = GetMorganGenerator(radius=radius, fpSize=nbits)

    fp = gen.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.float32)

    for bit in fp.GetOnBits():
        arr[bit] = 1.0

    return arr


def build_morgan_cache(smiles_map: dict[str, str], radius: int, nbits: int) -> dict[str, np.ndarray]:
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    cache: dict[str, np.ndarray] = {}

    for drug_id, smiles in smiles_map.items():
        vec = morgan_dense(smiles, radius=radius, nbits=nbits, gen=gen)
        if vec is not None:
            cache[str(drug_id)] = vec

    return cache