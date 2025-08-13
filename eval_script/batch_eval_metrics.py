#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Crystal Structure Evaluation Script
Efficient parallel processing for batch crystal structure evaluation

Core features:
1. Parallel data loading
2. Global data in memory
3. Batch parallel computing by metric type
4. Intelligent core allocation

Usage:
python batch_eval_metrics.py ./your_cif_folder --csv_folder ./datasets/mp_20
"""

import os
import sys
import json
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Third-party libraries
from pymatgen.core import Structure, Element, Composition
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
from scipy.stats import wasserstein_distance
import smact
from smact.screening import pauling_test

# Suppress warnings
warnings.filterwarnings("ignore")

# Global constants
DEFAULT_TIMEOUT = 30
STRUCTURE_CUTOFF = 0.5

class TimeoutException(Exception):
    pass

import signal
def timeout_handler(signum, frame):
    raise TimeoutException("Computation timeout")

class SimplifiedCrystal:
    """Simplified Crystal class with essential properties and methods"""
    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        
        # Check NaN values
        if np.isnan(self.lengths).any() or np.isinf(self.lengths).any():
            self.lengths = np.array([1, 1, 1]) * 100
            crys_array_dict['lengths'] = self.lengths
            self.constructed = False
            self.invalid_reason = 'nan_value'
        
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)

        self.get_structure()
        self.get_composition()
        self.get_validity()
        
    def get_structure(self):
        """Build pymatgen Structure object - FlowMM compatible"""
        if (1 > self.atom_types).any() or (self.atom_types > 104).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
            return
        elif min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
            return
        elif np.isnan(self.lengths).any() or np.isnan(self.angles).any() or np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
            return
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
                
    def get_composition(self):
        """Get chemical composition"""
        elem_counter = Counter(self.atom_types)
        if len(elem_counter) == 0:
            self.elems = ()
            self.comps = ()
            return
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())
        
    def get_validity(self):
        """Check validity"""
        if self.constructed:
            if len(self.elems) == 0:
                self.comp_valid = False
            else:
                self.comp_valid = smact_validity(self.elems, self.comps)
            self.struct_valid = structure_validity(self.structure)
        else:
            self.comp_valid = False
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

def structure_validity(crystal, cutoff=STRUCTURE_CUTOFF):
    """Check crystal structure physical validity - FlowMM compatible"""
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

def smact_validity(comp, count, use_pauling_test=True, include_alloys=True, timeout=DEFAULT_TIMEOUT):
    """Check chemical validity"""
    chemical_symbols = [
        'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
        if len(set(elem_symbols)) == 1:
            return True
        if include_alloys:
            is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
            if all(is_metal_list):
                return True

        threshold = np.max(count)
        compositions = []
        oxn = 1
        for oxc in ox_combos:
            oxn *= len(oxc)
        if oxn > 1e7:
            return False
        for ox_states in itertools.product(*ox_combos):
            stoichs = [(c,) for c in count]
            cn_e, cn_r = smact.neutral_ratios(
                ox_states, stoichs=stoichs, threshold=threshold)
            if cn_e:
                if use_pauling_test:
                    try:
                        electroneg_OK = pauling_test(ox_states, electronegs)
                    except TypeError:
                        electroneg_OK = True
                else:
                    electroneg_OK = True
                if electroneg_OK:
                    return True
        return False
        
    except TimeoutException:
        return False
    finally:
        signal.alarm(0)

def get_gt_crys_ori(cif):
    """Build Crystal object from CIF"""
    try:
        structure = Structure.from_str(cif, fmt='cif')
        lattice = structure.lattice
        crys_array_dict = {
            'frac_coords': structure.frac_coords,
            'atom_types': np.array([_.Z for _ in structure.species]),
            'lengths': np.array(lattice.abc),
            'angles': np.array(lattice.angles),
        }
        return SimplifiedCrystal(crys_array_dict)
    except Exception as e:
        return None

def structure_to_crystal(structure):
    """Convert pymatgen Structure to Crystal object"""
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords': structure.frac_coords,
        'atom_types': np.array([site.specie.Z for site in structure.sites]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles),
    }   
    return SimplifiedCrystal(crys_array_dict)

def get_chemsys(composition):
    """Get chemical system from composition"""
    try:
        comp = Composition(composition)
        return list(comp.elements)
    except:
        return []

def to_structure(cif_content):
    """Convert CIF content to Structure object"""
    try:
        if isinstance(cif_content, str):
            if os.path.exists(cif_content):
                structure = Structure.from_file(cif_content)
            else:
                structure = Structure.from_str(cif_content, fmt="cif")
        else:
            structure = cif_content
        return structure
    except Exception as e:
        return None

def get_matches(structure: Structure, alternatives: pd.Series, matcher: StructureMatcher) -> Tuple[List[int], List[float]]:
    """Get structure matching results"""
    structure = to_structure(structure)
    
    if structure is None:
        return [], []
        
    matches, rms_dists = [], []
    for ind, alt in alternatives.items():
        alt_structure = to_structure(alt)
        
        if alt_structure is None:
            continue
            
        try:
            rms_dist = matcher.get_rms_dist(structure, alt_structure)
            if rms_dist is not None:
                rms_dist, *_ = rms_dist
                rms_dists.append(rms_dist)
                matches.append(ind)
        except Exception as e:
            continue
    
    return matches, rms_dists

# Module-level functions to avoid pickle errors
def load_cif_with_folder_info(args):
    """Load single CIF file with folder info"""
    cif_path, folder_name = args
    try:
        structure = Structure.from_file(str(cif_path))
        key = f"{folder_name}/{cif_path.name}"
        return key, structure
    except Exception as e:
        return None, None

def convert_structure_with_key_info(item):
    """Convert Structure to Crystal object while maintaining key info"""
    key, structure = item
    try:
        crystal = structure_to_crystal(structure)
        return key, crystal
    except Exception as e:
        return key, None

def extract_crystal_features(crystal):
    """Extract features from single crystal: density, number of elements"""
    try:
        if not crystal.constructed:
            return None, None
        
        # Density calculation
        density = crystal.structure.density
        
        # Number of elements
        nelems = len(set(crystal.structure.species))
        
        return density, nelems
    except Exception as e:
        return None, None

def extract_folder_features_with_sampling(task):
    """Extract features for entire folder with 1000 sampling"""
    folder_name, valid_crystals = task
    
    try:
        densities = []
        nelems = []
        
        for crystal in valid_crystals:
            try:
                if not crystal.constructed:
                    continue
                
                # Density calculation
                density = crystal.structure.density
                densities.append(density)
                
                # Number of elements
                nelem = len(set(crystal.structure.species))
                nelems.append(nelem)
                    
            except Exception:
                continue
        
        # Sample 1000 if available, otherwise use all
        if len(densities) >= 1000:
            np.random.seed(42)  # Fixed seed for reproducibility
            indices = np.random.choice(len(densities), 1000, replace=False)
            sampled_densities = [densities[i] for i in indices]
            sampled_nelems = [nelems[i] for i in indices]
        else:
            sampled_densities = densities
            sampled_nelems = nelems
        
        return folder_name, (sampled_densities, sampled_nelems)
    except Exception as e:
        return folder_name, None

def calculate_novel_global_task(item):
    """Global parallel calculation of Novel metric"""
    folder_name, filename, structure, gen_chem, train_same_chem_structures = item
    
    try:
        # Skip single element structures
        if len(gen_chem) < 2:
            return folder_name, filename, False
            
        matcher = StructureMatcher()
        
        # Calculate Novelty (compare with training set)
        novel_matches = []
        if len(train_same_chem_structures) > 0:
            novel_matches, _ = get_matches(structure, pd.Series(train_same_chem_structures), matcher)
        
        is_novel = len(novel_matches) == 0
        return folder_name, filename, is_novel
        
    except Exception as e:
        return folder_name, filename, False

def calculate_unique_folder_task(item):
    """Calculate Unique metric for single folder"""
    folder_name, folder_crystals = item
    
    try:
        # Group by chemical system
        current_by_chemsys = defaultdict(list)
        crystal_list = []
        
        for filename, crystal in folder_crystals.items():
            if not crystal.constructed:
                continue
            try:
                composition = crystal.structure.composition.formula
                chemsys = tuple(sorted(get_chemsys(composition)))
                
                # Skip single element structures
                if len(chemsys) < 2:
                    continue
                
                crystal_list.append((filename, crystal.structure, chemsys))
                current_by_chemsys[chemsys].append(crystal.structure)
            except:
                continue
        
        # Calculate Unique for each structure
        unique_results = {}
        matcher = StructureMatcher()
        
        for filename, structure, gen_chem in crystal_list:
            # Exclude self from same chemical system structures
            same_chemsys_current = [s for s in current_by_chemsys.get(gen_chem, []) 
                                  if not np.array_equal(s.frac_coords, structure.frac_coords)]
            
            unique_matches = []
            if len(same_chemsys_current) > 0:
                unique_matches, _ = get_matches(structure, pd.Series(same_chemsys_current), matcher)
            
            is_unique = len(unique_matches) == 0
            unique_results[filename] = is_unique
        
        return folder_name, unique_results
        
    except Exception as e:
        return folder_name, {}

class CPUManager:
    """CPU core intelligent allocation manager"""
    
    def __init__(self):
        self.total_cores = cpu_count()
        print(f"Detected {self.total_cores} CPU cores")
        
        # Core allocation strategy
        self.allocations = self._calculate_allocations()
        
    def _calculate_allocations(self) -> Dict[str, int]:
        """Intelligent core allocation based on CPU count"""
        total = self.total_cores
        
        # Reserve 3% for system
        available = int(total * 0.97)
        
        # Optimized allocation strategy
        allocations = {
            'data_loading': max(1, int(available * 0.30)),
            'validity_computing': max(1, int(available * 0.20)),
            'distribution_computing': max(1, int(available * 0.25)),
            'un_computing': max(1, int(available * 0.25)),
        }
        
        print(f"CPU core allocation strategy:")
        for task, cores in allocations.items():
            print(f"  {task}: {cores} cores")
        
        return allocations
    
    def get_cores(self, task: str) -> int:
        """Get core count for specified task"""
        return self.allocations.get(task, 1)

class GlobalDataManager:
    """Global data manager - load all data to memory once"""
    
    def __init__(self, main_folder: str, csv_folder: str, cpu_manager: CPUManager):
        self.main_folder = Path(main_folder)
        self.csv_folder = Path(csv_folder)
        self.cpu_manager = cpu_manager
        
        # Data containers
        self.reference_datasets = {}
        self.all_structures = {}  # {folder_name/file_name: structure}
        self.all_crystals = {}    # {folder_name/file_name: crystal}
    
    def load_reference_datasets(self):
        """Load reference datasets"""
        print("="*60)
        print("Stage 1: Loading reference datasets")
        print("="*60)
        
        datasets = {}
        required_files = ['train.csv', 'val.csv', 'test.csv']
        
        for filename in required_files:
            csv_path = self.csv_folder / filename
            if not csv_path.exists():
                print(f"Skipping non-existent file: {filename}")
                continue
                
            print(f"Processing reference dataset: {filename}")
            df = pd.read_csv(csv_path)
            
            if 'cif' not in df.columns:
                print(f"No 'cif' column in CSV file, skipping: {filename}")
                continue
            
            print(f"Parsing {len(df)} structures in parallel...")
            
            n_workers = self.cpu_manager.get_cores('data_loading')
            with Pool(n_workers) as pool:
                crystals = list(tqdm(
                    pool.imap(get_gt_crys_ori, df['cif'].tolist()),
                    total=len(df),
                    desc=f"Parsing {filename.replace('.csv', '')}"
                ))
            
            valid_crystals = [c for c in crystals if c is not None]
            print(f"{filename} processing complete: {len(valid_crystals)} valid structures")
            datasets[filename.replace('.csv', '')] = valid_crystals
        
        self.reference_datasets = datasets
    
    def load_all_structures(self):
        """Load all CIF files at once"""
        print("="*60)
        print("Stage 2: Loading all CIF files at once")
        print("="*60)
        
        # Collect all CIF files
        all_cif_files = []
        folder_names = []
        
        for folder_path in self.main_folder.iterdir():
            if not folder_path.is_dir():
                continue
                
            cif_files = list(folder_path.glob("*.cif"))
            for cif_file in cif_files:
                all_cif_files.append(cif_file)
                folder_names.append(folder_path.name)
        
        print(f"Found {len(all_cif_files)} CIF files from {len(set(folder_names))} folders")
        
        if len(all_cif_files) == 0:
            print("No CIF files found")
            return
        
        # Parallel load all CIF files at once
        print("Starting parallel loading of all CIF files...")
        n_workers = self.cpu_manager.get_cores('data_loading')
        
        tasks = list(zip(all_cif_files, folder_names))
        
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(load_cif_with_folder_info, tasks),
                total=len(tasks),
                desc="Loading all CIF files"
            ))
        
        # Organize results
        for key, structure in results:
            if key is not None and structure is not None:
                self.all_structures[key] = structure
        
        print(f"Loading complete: {len(self.all_structures)} structures")
    
    def convert_all_to_crystals(self):
        """Convert all Structures to Crystal objects at once"""
        print("="*60)
        print("Stage 3: Converting to Crystal objects at once")
        print("="*60)
        
        print(f"Starting conversion of {len(self.all_structures)} Structures to Crystal objects...")
        
        n_workers = self.cpu_manager.get_cores('data_loading')
        
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(convert_structure_with_key_info, self.all_structures.items()),
                total=len(self.all_structures),
                desc="Converting Crystal objects"
            ))
        
        # Organize results
        for key, crystal in results:
            if crystal is not None:
                self.all_crystals[key] = crystal
        
        print(f"Conversion complete: {len(self.all_crystals)} Crystal objects")
    
    def get_folder_data(self, folder_name: str) -> Dict[str, SimplifiedCrystal]:
        """Get data for specified folder"""
        folder_data = {}
        prefix = f"{folder_name}/"
        
        for key, crystal in self.all_crystals.items():
            if key.startswith(prefix):
                filename = key[len(prefix):]
                folder_data[filename] = crystal
        
        return folder_data
    
    def get_all_folder_names(self) -> List[str]:
        """Get all folder names"""
        folder_names = set()
        for key in self.all_crystals.keys():
            folder_name = key.split('/')[0]
            folder_names.add(folder_name)
        return sorted(folder_names)

class BatchMetricsCalculator:
    """Batch metrics calculator"""
    
    def __init__(self, data_manager: GlobalDataManager, cpu_manager: CPUManager):
        self.data_manager = data_manager
        self.cpu_manager = cpu_manager
        
        # Preprocess reference datasets
        self.test_crystals = data_manager.reference_datasets.get('test', [])
        
        # Pre-build chemical system indices
        self._build_chemical_system_indices()
        
        # Pre-compute reference dataset features
        self._precompute_reference_features()
        
    def _precompute_reference_features(self):
        """Pre-compute all features of reference datasets with sampling"""
        print("Pre-computing reference dataset features...")
        
        test_crystals = self.data_manager.reference_datasets.get('test', [])
        valid_crystals = [c for c in test_crystals if c.valid and c.constructed]
        
        if len(valid_crystals) == 0:
            self.test_features = {'densities': [], 'nelems': []}
            return
        
        print(f"  Pre-computing test set features ({len(valid_crystals)} structures)...")
        
        # Parallel feature computation
        n_workers = self.cpu_manager.get_cores('distribution_computing')
        
        with Pool(n_workers) as pool:
            features_list = list(tqdm(
                pool.imap(extract_crystal_features, valid_crystals),
                total=len(valid_crystals),
                desc="Extracting test features"
            ))
        
        # Separate features
        densities = [f[0] for f in features_list if f[0] is not None]
        nelems = [f[1] for f in features_list if f[1] is not None]
        
        # Sample 1000 if available, otherwise use all
        if len(densities) >= 1000:
            np.random.seed(42)  # Fixed seed for reproducibility
            indices = np.random.choice(len(densities), 1000, replace=False)
            sampled_densities = [densities[i] for i in indices]
            sampled_nelems = [nelems[i] for i in indices]
        else:
            sampled_densities = densities
            sampled_nelems = nelems
        
        self.test_features = {
            'densities': sampled_densities,
            'nelems': sampled_nelems
        }
        
        print(f"  Test set feature pre-computation complete: density={len(sampled_densities)}, nelems={len(sampled_nelems)}")
        print("Reference dataset feature pre-computation complete!")
    
    def _build_chemical_system_indices(self):
        """Pre-build chemical system indices"""
        print("Pre-building chemical system indices...")
        
        # Training set chemical system index
        self.train_by_chemsys = defaultdict(list)
        train_crystals = self.data_manager.reference_datasets.get('train', [])
        for crystal in train_crystals:
            if not crystal.constructed:
                continue
            try:
                composition = crystal.structure.composition.formula
                chemsys = tuple(sorted(get_chemsys(composition)))
                # Skip single element structures
                if len(chemsys) < 2:
                    continue
                self.train_by_chemsys[chemsys].append(crystal.structure)
            except:
                continue
        
        print(f"Chemical system indexing complete: train={len(self.train_by_chemsys)}")
    
    def calculate_validity_metrics_batch(self, folder_names: List[str]) -> Dict[str, Dict]:
        """Batch calculate validity metrics"""
        print("="*60)
        print("Batch calculating validity metrics")
        print("="*60)
        
        results = {}
        
        for folder_name in tqdm(folder_names, desc="Calculating validity metrics"):
            folder_data = self.data_manager.get_folder_data(folder_name)
            
            total_count = len(folder_data)
            comp_valid_count = sum(1 for c in folder_data.values() if c.comp_valid)
            struct_valid_count = sum(1 for c in folder_data.values() if c.struct_valid)
            valid_count = sum(1 for c in folder_data.values() if c.valid)
            
            results[folder_name] = {
                "chemical_validity": comp_valid_count / total_count if total_count > 0 else 0,
                "structural_validity": struct_valid_count / total_count if total_count > 0 else 0,
                "overall_validity": valid_count / total_count if total_count > 0 else 0,
                "counts": {
                    "total": total_count,
                    "chemical_valid": comp_valid_count,
                    "structural_valid": struct_valid_count,
                    "overall_valid": valid_count
                }
            }
        
        return results
    
    def calculate_distribution_metrics_batch(self, folder_names: List[str]) -> Dict[str, Dict]:
        """Batch calculate distribution distance metrics with sampling"""
        print("="*60)
        print("Batch calculating distribution distance metrics with 1000 sampling")
        print("="*60)
        
        # Step 1: Parallel extract features for all folders with sampling
        print("Step 1: Batch extracting features for all folders with sampling...")
        
        n_workers = self.cpu_manager.get_cores('distribution_computing')
        
        # Prepare tasks for all folders
        folder_tasks = []
        for folder_name in folder_names:
            folder_data = self.data_manager.get_folder_data(folder_name)
            valid_crystals = [c for c in folder_data.values() if c.valid and c.constructed]
            if len(valid_crystals) > 0:
                folder_tasks.append((folder_name, valid_crystals))
        
        # Parallel feature extraction with sampling
        with Pool(n_workers) as pool:
            folder_features = list(tqdm(
                pool.imap(extract_folder_features_with_sampling, folder_tasks),
                total=len(folder_tasks),
                desc="Extracting folder features with sampling"
            ))
        
        # Step 2: Vectorized distance calculation
        print("Step 2: Vectorized distribution distance calculation...")
        
        results = {}
        
        for folder_name, features in tqdm(folder_features, desc="Calculating distribution distances"):
            if features is None:
                results[folder_name] = {"error": "No valid crystal structures"}
                continue
            
            pred_densities, pred_nelems = features
            folder_results = {}
            
            # Only compare with test set
            if len(self.test_features['densities']) == 0:
                folder_results['d_density_test'] = None
                folder_results['d_elements_test'] = None
            else:
                try:
                    # Density distribution distance
                    wdist_density = wasserstein_distance(pred_densities, self.test_features['densities'])
                    folder_results['d_density_test'] = wdist_density
                    
                    # Number of elements distribution distance
                    wdist_elements = wasserstein_distance(pred_nelems, self.test_features['nelems'])
                    folder_results['d_elements_test'] = wdist_elements
                    
                except Exception as e:
                    folder_results['d_density_test'] = None
                    folder_results['d_elements_test'] = None
            
            results[folder_name] = folder_results
        
        return results
    
    def calculate_un_metrics_batch(self, folder_names: List[str]) -> Dict[str, Dict]:
        """Batch calculate UN metrics with single element filtering"""
        print("="*60)
        print("Batch calculating UN metrics with single element filtering")
        print("="*60)
        
        if len(self.data_manager.reference_datasets.get('train', [])) == 0:
            return {name: {"error": "Training set is empty"} for name in folder_names}
        
        # Step 1: Global parallel computation of Novel metric
        print("Step 1: Global parallel computation of Novel metric...")
        all_novel_tasks = []
        folder_structure_counts = {}
        
        for folder_name in folder_names:
            folder_data = self.data_manager.get_folder_data(folder_name)
            folder_structure_counts[folder_name] = 0
            
            for filename, crystal in folder_data.items():
                if not crystal.constructed:
                    continue
                try:
                    composition = crystal.structure.composition.formula
                    chemsys = tuple(sorted(get_chemsys(composition)))
                    
                    # Skip single element structures
                    if len(chemsys) < 2:
                        continue
                        
                    same_chemsys_train = self.train_by_chemsys.get(chemsys, [])
                    
                    all_novel_tasks.append((folder_name, filename, crystal.structure, chemsys, same_chemsys_train))
                    folder_structure_counts[folder_name] += 1
                except:
                    continue
        
        print(f"Prepared {len(all_novel_tasks)} Novel tasks")
        
        # Global parallel Novel computation
        n_workers = min(self.cpu_manager.total_cores - 5, len(all_novel_tasks))
        
        with Pool(n_workers) as pool:
            all_novel_results = list(tqdm(
                pool.imap(calculate_novel_global_task, all_novel_tasks),
                total=len(all_novel_tasks),
                desc=f"Global Novel computation ({n_workers} cores)"
            ))
        
        # Step 2: Folder parallel computation of Unique metric
        print("Step 2: Folder parallel computation of Unique metric...")
        folder_unique_tasks = []
        
        for folder_name in folder_names:
            folder_data = self.data_manager.get_folder_data(folder_name)
            # Only pass constructed crystals
            valid_folder_data = {k: v for k, v in folder_data.items() if v.constructed}
            if len(valid_folder_data) > 0:
                folder_unique_tasks.append((folder_name, valid_folder_data))
        
        # Folder-level parallel Unique computation
        n_workers_unique = min(self.cpu_manager.get_cores('un_computing'), len(folder_unique_tasks))
        
        with Pool(n_workers_unique) as pool:
            all_unique_results = list(tqdm(
                pool.imap(calculate_unique_folder_task, folder_unique_tasks),
                total=len(folder_unique_tasks),
                desc=f"Folder parallel Unique computation ({n_workers_unique} cores)"
            ))
        
        # Step 3: Merge results and statistics by folder
        print("Step 3: Merging Novel and Unique results...")
        results = {}
        
        # Group Novel results by folder
        novel_by_folder = defaultdict(dict)
        for folder_name, filename, is_novel in all_novel_results:
            novel_by_folder[folder_name][filename] = is_novel
        
        # Convert Unique results to dictionary
        unique_by_folder = {folder_name: unique_results for folder_name, unique_results in all_unique_results}
        
        # Statistics for each folder
        for folder_name in folder_names:
            novel_results = novel_by_folder.get(folder_name, {})
            unique_results = unique_by_folder.get(folder_name, {})
            
            # Count various metrics
            unique_count = sum(1 for is_unique in unique_results.values() if is_unique)
            novel_count = sum(1 for is_novel in novel_results.values() if is_novel)
            
            # Calculate UN (both unique and novel)
            detailed_results = {}
            un_count = 0
            
            # Merge results, ensure files have both unique and novel
            all_filenames = set(unique_results.keys()) | set(novel_results.keys())
            
            for filename in all_filenames:
                is_unique = unique_results.get(filename, False)
                is_novel = novel_results.get(filename, False)
                is_un = is_unique and is_novel
                
                detailed_results[filename] = {
                    "is_unique": is_unique,
                    "is_novel": is_novel,
                    "is_unique_and_novel": is_un
                }
                
                if is_un:
                    un_count += 1
            
            total_count = len(detailed_results)
            
            results[folder_name] = {
                "unique_rate": unique_count / total_count if total_count > 0 else 0,
                "novel_rate": novel_count / total_count if total_count > 0 else 0,
                "un_rate": un_count / total_count if total_count > 0 else 0,
                "counts": {
                    "unique": unique_count,
                    "novel": novel_count,
                    "unique_and_novel": un_count,
                    "total": total_count
                },
                "detailed_results": detailed_results,
                "chemical_systems_info": {
                    "training_set": len(self.train_by_chemsys)
                }
            }
        
        return results

class OptimizedCrystalEvaluator:
    """Optimized crystal structure evaluator main controller"""
    
    def __init__(self, main_folder: str, csv_folder: str):
        self.main_folder = Path(main_folder)
        self.csv_folder = Path(csv_folder)
        
        # Validate input
        if not self.main_folder.exists():
            raise ValueError(f"Main folder does not exist: {main_folder}")
        if not self.csv_folder.exists():
            raise ValueError(f"CSV folder does not exist: {csv_folder}")
        
        # Create managers
        self.cpu_manager = CPUManager()
        self.data_manager = GlobalDataManager(main_folder, csv_folder, self.cpu_manager)
        
        # Create output directory
        self.output_folder = Path(f"{self.main_folder.name}_flowmm_aligned_analysis")
        self.output_folder.mkdir(exist_ok=True)
        
        # Status file
        self.status_file = self.output_folder / ".processing_status.json"
        
        print(f"Output directory: {self.output_folder}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get processing status"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed_folders": [], "failed_folders": []}
    
    def update_processing_status(self, status: Dict[str, Any]):
        """Update processing status"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Failed to update status file: {e}")
    
    def run_evaluation(self):
        """Run complete evaluation process"""
        print("="*80)
        print("FlowMM-Aligned Crystal Structure Evaluation Started")
        print("="*80)
        
        start_time = time.time()
        
        # Stage 1-3: Global data loading
        self.data_manager.load_reference_datasets()
        self.data_manager.load_all_structures()  
        self.data_manager.convert_all_to_crystals()
        
        # Get all folders
        all_folder_names = self.data_manager.get_all_folder_names()
        print(f"Found {len(all_folder_names)} folders to process")
        
        if len(all_folder_names) == 0:
            print("No folders containing CIF files found")
            return
        
        # Check completed folders
        status = self.get_processing_status()
        completed = set(status.get("completed_folders", []))
        pending_folders = [name for name in all_folder_names if name not in completed]
        
        print(f"Completed: {len(completed)} folders")
        print(f"Pending: {len(pending_folders)} folders")
        
        if len(pending_folders) == 0:
            print("All folders have been processed!")
            return
        
        # Create batch calculator
        calculator = BatchMetricsCalculator(self.data_manager, self.cpu_manager)
        
        # Stage 4: Batch calculation of all metrics
        print("="*60)
        print("Starting batch calculation of all metrics")
        print("="*60)
        
        validity_results = calculator.calculate_validity_metrics_batch(pending_folders)
        
        distribution_results = calculator.calculate_distribution_metrics_batch(pending_folders)
        
        un_results = calculator.calculate_un_metrics_batch(pending_folders)
        
        # Stage 5: Save results
        print("="*60)
        print("Saving results")
        print("="*60)
        
        for folder_name in tqdm(pending_folders, desc="Saving results"):
            try:
                # Assemble complete results
                complete_results = {
                    "folder_name": folder_name,
                    "validity_metrics": validity_results.get(folder_name, {}),
                    "distribution_metrics": distribution_results.get(folder_name, {}),
                    "un_metrics": un_results.get(folder_name, {}),
                    "processing_info": {
                        "total_cif_files": len(self.data_manager.get_folder_data(folder_name)),
                        "alignment": "flowmm_compatible",
                        "modifications": ["flowmm_structure_construction", "flowmm_validity_check", "1000_sampling", "single_element_filtering"]
                    }
                }
                
                # Save JSON (complete results)
                json_output = self.output_folder / f"{folder_name}_results.json"
                with open(json_output, 'w') as f:
                    json.dump(complete_results, f, indent=2, default=str)
                
                # Save CSV (simplified results)
                csv_output = self.output_folder / f"{folder_name}_results.csv"
                csv_data = []
                
                if "un_metrics" in complete_results and "detailed_results" in complete_results["un_metrics"]:
                    for filename, un_data in complete_results["un_metrics"]["detailed_results"].items():
                        row = {
                            "filename": filename,
                            "is_unique": un_data["is_unique"],
                            "is_novel": un_data["is_novel"],
                            "is_unique_and_novel": un_data["is_unique_and_novel"]
                        }
                        csv_data.append(row)
                
                if csv_data:
                    pd.DataFrame(csv_data).to_csv(csv_output, index=False)
                
                # Update status
                status["completed_folders"].append(folder_name)
                self.update_processing_status(status)
                
            except Exception as e:
                print(f"Error saving results for folder {folder_name}: {e}")
                status.setdefault("failed_folders", []).append({
                    "folder": folder_name,
                    "error": str(e)
                })
                self.update_processing_status(status)
                continue
        
        # Generate summary report
        self._generate_summary_report(pending_folders, validity_results, distribution_results, un_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("="*80)
        print("FlowMM-Aligned batch processing complete!")
        print(f"Processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average per folder: {total_time/len(pending_folders):.2f} seconds")
        print(f"Processed folders: {len(pending_folders)}")
        print(f"Total CIF files: {len(self.data_manager.all_structures)}")
        print(f"Results saved in: {self.output_folder}")
        print("FlowMM alignment: structure construction, validity check, 1000 sampling, single element filtering")
        print("="*80)
    
    def _generate_summary_report(self, folder_names: List[str], validity_results: Dict, distribution_results: Dict, un_results: Dict):
        """Generate summary report"""
        summary_data = []
        
        for folder_name in folder_names:
            validity = validity_results.get(folder_name, {})
            distribution = distribution_results.get(folder_name, {})
            un = un_results.get(folder_name, {})
            
            row = {
                "folder_name": folder_name,
                "chemical_validity": validity.get("chemical_validity", 0),
                "structural_validity": validity.get("structural_validity", 0),
                "overall_validity": validity.get("overall_validity", 0),
                "d_density_test": distribution.get("d_density_test", None),
                "d_elements_test": distribution.get("d_elements_test", None),
                "unique_rate": un.get("unique_rate", 0),
                "novel_rate": un.get("novel_rate", 0),
                "un_rate": un.get("un_rate", 0),
                "total_structures": validity.get("counts", {}).get("total", 0)
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_folder / "summary_report.csv", index=False)
        
        print(f"Summary report saved: {self.output_folder / 'summary_report.csv'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FlowMM-Aligned Crystal Structure Evaluation Script')
    parser.add_argument('main_folder', type=str, help='Main folder path containing multiple subfolders')
    parser.add_argument('--csv_folder', type=str, required=True, 
                       help='Folder path containing train.csv, val.csv, test.csv')
    
    args = parser.parse_args()
    
    # Check input folders
    if not os.path.exists(args.main_folder):
        print(f"Error: Main folder does not exist: {args.main_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.csv_folder):
        print(f"Error: CSV folder does not exist: {args.csv_folder}")
        sys.exit(1)
    
    # Create evaluator and start processing
    evaluator = OptimizedCrystalEvaluator(args.main_folder, args.csv_folder)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
