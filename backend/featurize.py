import pandas as pd
from pymatgen.core import Composition, Element
from matminer.featurizers.composition import ElementFraction
import numpy as np

# List of 111 target elements
TARGET_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
    'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr'
]

# Superheavy elements to drop if present
EXTRA_ELEMENTS = ['Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

# 300 spacegroups
SPACEGROUPS = list(range(1, 231))

def featurize_formula_with_spacegroups(formula: str) -> pd.DataFrame:
    """
    Featurize a chemical formula into 415 columns for all 230 spacegroups (vectorized):
      - 111 target element fractions
      - 12 element-based descriptors
      - 300 one-hot spacegroup columns (one row per spacegroup)

    Returns:
        df_final (pd.DataFrame): 300 rows, 415 columns
    """
    comp = Composition(formula)
    ef = ElementFraction()
    ef_features = ef.featurize(comp)
    df_base = pd.DataFrame([ef_features], columns=ef.feature_labels())

    # Ensure all target element columns exist
    for el in TARGET_ELEMENTS:
        if el not in df_base.columns:
            df_base[el] = 0.0
    df_base = df_base[TARGET_ELEMENTS]

    # Drop superheavy elements if present
    df_base = df_base.drop(columns=[el for el in EXTRA_ELEMENTS if el in df_base.columns])

    # Element-based descriptors
    el_objs = [Element(e) for e in comp.elements]
    fractions = np.array([comp.get_atomic_fraction(e) for e in comp.elements])

    def safe_attr(el, attr):
        try:
            val = getattr(el, attr)
            return float(val) if val is not None else np.nan
        except Exception:
            return np.nan

    atomic_masses = np.array([safe_attr(e, "atomic_mass") for e in el_objs], dtype=np.float64)
    en_values = np.array([safe_attr(e, "X") for e in el_objs], dtype=np.float64)
    cov_radii = np.array([safe_attr(e, "atomic_radius_calculated") for e in el_objs], dtype=np.float64)
    ea_values = np.array([safe_attr(e, "electron_affinity") for e in el_objs], dtype=np.float64)

    # Add descriptors to base
    df_base["n_atoms"] = comp.num_atoms
    df_base["n_elements"] = len(comp.elements)
    df_base["avg_atomic_mass"] = np.nansum(atomic_masses * fractions)
    df_base["en_mean"] = np.nanmean(en_values)
    df_base["en_max"] = np.nanmax(en_values)
    df_base["en_min"] = np.nanmin(en_values)
    df_base["en_range"] = df_base["en_max"] - df_base["en_min"]
    df_base["avg_covalent_radius"] = np.nanmean(cov_radii)
    df_base["ea_mean"] = np.nanmean(ea_values)
    df_base["ea_max"] = np.nanmax(ea_values)
    df_base["ea_min"] = np.nanmin(ea_values)
    df_base["ea_range"] = df_base["ea_max"] - df_base["ea_min"]

    # --- Vectorized creation of 300 spacegroup rows ---
    n_spacegroups = len(SPACEGROUPS)
    df_vectorized = pd.concat([df_base]*n_spacegroups, ignore_index=True)

    # Create one-hot spacegroup matrix
    spacegroup_cols = [f"spacegroup_{sg}" for sg in SPACEGROUPS]
    df_vectorized[spacegroup_cols] = np.eye(n_spacegroups, dtype=np.float32)

    # Ensure column order: elements → descriptors → spacegroups
    final_columns = TARGET_ELEMENTS + [
        "n_atoms", "n_elements", "avg_atomic_mass", "en_mean", "en_max",
        "en_min", "en_range", "avg_covalent_radius", "ea_mean",
        "ea_max", "ea_min", "ea_range"
    ] + spacegroup_cols

    df_final = df_vectorized[final_columns].astype(np.float32)
    return df_final
