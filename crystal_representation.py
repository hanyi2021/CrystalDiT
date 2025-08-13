import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional
import json
from pymatgen.core.structure import Structure
import warnings
from tqdm import tqdm
import pickle
import time
import shutil

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# Periodic table mapping - atomic number to (period, group) position
ATOMIC_NUMBER_TO_POSITION = {
    # Period 1
    1: (1, 1),    # H
    2: (1, 18),   # He
    
    # Period 2
    3: (2, 1),    # Li
    4: (2, 2),    # Be
    5: (2, 13),   # B
    6: (2, 14),   # C
    7: (2, 15),   # N
    8: (2, 16),   # O
    9: (2, 17),   # F
    10: (2, 18),  # Ne
    
    # Period 3
    11: (3, 1),   # Na
    12: (3, 2),   # Mg
    13: (3, 13),  # Al
    14: (3, 14),  # Si
    15: (3, 15),  # P
    16: (3, 16),  # S
    17: (3, 17),  # Cl
    18: (3, 18),  # Ar
    
    # Period 4
    19: (4, 1),   # K
    20: (4, 2),   # Ca
    21: (4, 3),   # Sc
    22: (4, 4),   # Ti
    23: (4, 5),   # V
    24: (4, 6),   # Cr
    25: (4, 7),   # Mn
    26: (4, 8),   # Fe
    27: (4, 9),   # Co
    28: (4, 10),  # Ni
    29: (4, 11),  # Cu
    30: (4, 12),  # Zn
    31: (4, 13),  # Ga
    32: (4, 14),  # Ge
    33: (4, 15),  # As
    34: (4, 16),  # Se
    35: (4, 17),  # Br
    36: (4, 18),  # Kr
    
    # Period 5
    37: (5, 1),   # Rb
    38: (5, 2),   # Sr
    39: (5, 3),   # Y
    40: (5, 4),   # Zr
    41: (5, 5),   # Nb
    42: (5, 6),   # Mo
    43: (5, 7),   # Tc
    44: (5, 8),   # Ru
    45: (5, 9),   # Rh
    46: (5, 10),  # Pd
    47: (5, 11),  # Ag
    48: (5, 12),  # Cd
    49: (5, 13),  # In
    50: (5, 14),  # Sn
    51: (5, 15),  # Sb
    52: (5, 16),  # Te
    53: (5, 17),  # I
    54: (5, 18),  # Xe
    
    # Period 6 (including lanthanides)
    55: (6, 1),   # Cs
    56: (6, 2),   # Ba
    57: (6, 3),   # La
    58: (6, 3.01),  # Ce (Lanthanide)
    59: (6, 3.02),  # Pr (Lanthanide)
    60: (6, 3.03),  # Nd (Lanthanide)
    61: (6, 3.04),  # Pm (Lanthanide)
    62: (6, 3.05),  # Sm (Lanthanide)
    63: (6, 3.06),  # Eu (Lanthanide)
    64: (6, 3.07),  # Gd (Lanthanide)
    65: (6, 3.08),  # Tb (Lanthanide)
    66: (6, 3.09),  # Dy (Lanthanide)
    67: (6, 3.10), # Ho (Lanthanide)
    68: (6, 3.11), # Er (Lanthanide)
    69: (6, 3.12), # Tm (Lanthanide)
    70: (6, 3.13), # Yb (Lanthanide)
    71: (6, 3.14), # Lu (Lanthanide)
    72: (6, 4),   # Hf
    73: (6, 5),   # Ta
    74: (6, 6),   # W
    75: (6, 7),   # Re
    76: (6, 8),   # Os
    77: (6, 9),   # Ir
    78: (6, 10),  # Pt
    79: (6, 11),  # Au
    80: (6, 12),  # Hg
    81: (6, 13),  # Tl
    82: (6, 14),  # Pb
    83: (6, 15),  # Bi
    84: (6, 16),  # Po
    85: (6, 17),  # At
    86: (6, 18),  # Rn
    
    # Period 7 (including actinides)
    87: (7, 1),   # Fr
    88: (7, 2),   # Ra
    89: (7, 3),   # Ac
    90: (7, 3.01),  # Th (Actinide)
    91: (7, 3.02),  # Pa (Actinide)
    92: (7, 3.03),  # U (Actinide)
    93: (7, 3.04),  # Np (Actinide)
    94: (7, 3.05),  # Pu (Actinide)
    95: (7, 3.06),  # Am (Actinide)
    96: (7, 3.07),  # Cm (Actinide)
    97: (7, 3.08),  # Bk (Actinide)
    98: (7, 3.09),  # Cf (Actinide)
    99: (7, 3.10), # Es (Actinide)
    100: (7, 3.11), # Fm (Actinide)
    101: (7, 3.12), # Md (Actinide)
    102: (7, 3.13), # No (Actinide)
    103: (7, 3.14), # Lr (Actinide)
    104: (7, 4),   # Rf
    105: (7, 5),   # Db
    106: (7, 6),   # Sg
    107: (7, 7),   # Bh
    108: (7, 8),   # Hs
    109: (7, 9),   # Mt
    110: (7, 10),  # Ds
    111: (7, 11),  # Rg
    112: (7, 12),  # Cn
    113: (7, 13),  # Nh
    114: (7, 14),  # Fl
    115: (7, 15),  # Mc
    116: (7, 16),  # Lv
    117: (7, 17),  # Ts
    118: (7, 18),  # Og
    
    # Invalid atom placeholder
    0: (0, 0),
}

# Reverse mapping - (period, group) to atomic number
POSITION_TO_ATOMIC_NUMBER = {position: number for number, position in ATOMIC_NUMBER_TO_POSITION.items()}

def normalize_row(row):
    """Normalize period number to [-1, 1] range"""
    return (row / 7.0) * 2 - 1

def normalize_column(column):
    """Normalize group number to [-1, 1] range"""
    return (column / 18.0) * 2 - 1

def denormalize_row(norm_row):
    """Convert [-1, 1] range back to period number"""
    row = ((norm_row + 1) / 2) * 7
    return round(row)

def denormalize_column(norm_column):
    """Convert [-1, 1] range back to group number"""
    column = ((norm_column + 1) / 2) * 18
    return round(column)

def preprocess_dataset(csv_file, output_file, max_atoms=20, max_length=46.7425):
    """
    Preprocess dataset by parsing CIF strings into lattice vectors and atom features
    
    Args:
        csv_file: Path to dataset CSV file
        output_file: Path to preprocessed data output file
        max_atoms: Maximum number of atoms per crystal
        max_length: Maximum lattice vector length for normalization
    
    Returns:
        Number of successfully preprocessed samples
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(output_file) and not output_file.endswith('.tmp'):
        print(f"Preprocessed file already exists: {output_file}, skipping preprocessing")
        return 0
        
    data = pd.read_csv(csv_file)
    print(f"Starting dataset preprocessing: {csv_file}, total samples: {len(data)}")
    
    processed_data = []
    success_count = 0
    error_count = 0
    
    temp_file = output_file + '.tmp'
    
    try:
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing progress"):
            try:
                cif_string = row['cif']
                structure = Structure.from_str(cif_string, fmt="cif")
                
                # Get lattice vectors and normalize
                lattice_vectors = structure.lattice.matrix
                normalized_lattice = lattice_vectors / max_length
                
                # Get atom information
                atomic_numbers = []
                atom_coords = []
                
                for site in structure.sites:
                    if len(atomic_numbers) >= max_atoms:
                        break
                    
                    atomic_numbers.append(site.specie.Z)
                    atom_coords.append(site.frac_coords)
                
                # Pad with invalid atoms if necessary
                while len(atomic_numbers) < max_atoms:
                    atomic_numbers.append(0)
                    atom_coords.append([-1, -1, -1])
                
                # Convert atomic numbers to periodic table positions and normalize
                atom_features = []
                for z, coords in zip(atomic_numbers, atom_coords):
                    if z > 0:  # Valid atom
                        row, column = ATOMIC_NUMBER_TO_POSITION[z]
                        norm_row = normalize_row(row)
                        norm_column = normalize_column(column)
                        atom_features.append([norm_row, norm_column, coords[0], coords[1], coords[2]])
                    else:  # Invalid atom
                        atom_features.append([-1, -1, -1, -1, -1])
                
                lattice_vectors_np = normalized_lattice.astype(np.float32)
                atom_features_np = np.array(atom_features, dtype=np.float32)
                
                processed_data.append({
                    'lattice_vectors': lattice_vectors_np,
                    'atom_features': atom_features_np,
                    'index': idx
                })
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                error_count += 1
                
                # Create default zero lattice
                lattice_vectors_np = np.zeros((3, 3), dtype=np.float32)
                atom_features_np = np.array([[-1, -1, -1, -1, -1] for _ in range(max_atoms)], dtype=np.float32)
                
                processed_data.append({
                    'lattice_vectors': lattice_vectors_np,
                    'atom_features': atom_features_np,
                    'index': idx
                })
        
        # Save preprocessed data to temporary file
        with open(temp_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Atomic rename operation
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
        
        print(f"Preprocessing completed: success {success_count}, failed {error_count}")
        print(f"Preprocessed data saved to: {output_file}")
        
    except Exception as e:
        print(f"Serious error during preprocessing: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise e
    
    return success_count


class CrystalDataset(Dataset):
    """
    Crystal structure dataset
    Supports periodic table position representation for atoms
    Can load preprocessed data or parse from CIF strings
    """
    def __init__(self, csv_file, max_atoms=20, max_length=46.7425, transform=None, preprocessed_file=None, force_preprocess=False):
        """
        Args:
            csv_file: Path to dataset CSV file
            max_atoms: Maximum number of atoms per crystal
            max_length: Maximum lattice vector length for normalization
            transform: Optional data transform
            preprocessed_file: Path to preprocessed file, auto-generated if None
            force_preprocess: Force repreprocessing even if preprocessed file exists
        """
        self.csv_file = csv_file
        self.max_atoms = max_atoms
        self.max_length = max_length
        self.transform = transform
        
        if preprocessed_file is None:
            preprocessed_dir = os.path.join(os.path.dirname(csv_file), "preprocessed")
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            csv_basename = os.path.basename(csv_file).split('.')[0]
            self.preprocessed_file = os.path.join(
                preprocessed_dir, 
                f"{csv_basename}_preprocessed_ma{max_atoms}.pkl"
            )
        else:
            self.preprocessed_file = preprocessed_file
        
        # Check if preprocessing is needed
        if force_preprocess or not os.path.exists(self.preprocessed_file):
            preprocess_dataset(csv_file, self.preprocessed_file, max_atoms, max_length)
        
        # Load preprocessed data with retry mechanism
        max_retries = 5
        retry_count = 0
        retry_delay = 1
        
        while retry_count < max_retries:
            try:
                with open(self.preprocessed_file, 'rb') as f:
                    self.data = pickle.load(f)
                break
            except (EOFError, pickle.UnpicklingError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Failed to load dataset, retrying ({retry_count}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load preprocessed data {self.preprocessed_file} after multiple attempts: {e}")
        
        print(f"Loaded dataset: {csv_file}, samples: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_data = self.data[idx]
        
        lattice_vectors = sample_data['lattice_vectors']
        atom_features = sample_data['atom_features']
        
        lattice_vectors_tensor = torch.from_numpy(lattice_vectors)
        atom_features_tensor = torch.from_numpy(atom_features)
        
        sample = {
            'lattice_vectors': lattice_vectors_tensor,
            'atom_features': atom_features_tensor,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
