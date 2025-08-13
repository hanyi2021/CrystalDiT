#!/usr/bin/env python3
"""
DFT Post-Processor - Second Stage Evaluation
Functions:
1. Read CHGNet evaluation results
2. Load and match DFT calculation results
3. Apply DFT energy corrections
4. Recalculate DFT-based hull energies
5. Calculate final SUN metrics
6. Generate complete evaluation report

Note: dft_results_folder should contain dft_un folder with traj files

Usage:
python ./eval_script/dft_post_processor.py \
    --chgnet_csv ./20250_1dit_chgnet_results/chgnet_results.csv \
    --dft_results_folder ./20250_1dit_chgnet_results/ \
    --original_csv_file ./20250crystal_optimized_analysis/20250_1dit_crystals_results.csv \
    --mp_hull_path ./mp_02072023/2023-02-07-ppd-mp.pkl \
    --output_dir ./20250_1dit_chgnet_results/final_sun \
    --un_only  # Optional: process UN structures only
"""

import os
import sys
import json
import pickle
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
import contextlib
import io

from ase.io.trajectory import Trajectory
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram, PDEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

# Suppress warnings and output
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_all_output():
    """Completely suppress all output"""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield

class DFTResult:
    """DFT calculation result data structure"""
    def __init__(self, structure_id: int, filename: str):
        self.structure_id = structure_id
        self.filename = filename
        self.success = False
        self.error_message = None
        
        # DFT energy information
        self.initial_energy_dft = None
        self.final_energy_dft = None
        self.corrected_energy_dft = None
        self.final_structure_dft = None
        
        # Energy correction information
        self.computed_structure_entry = None
        self.phase_diagram_entry = None
        
        # Stability information
        self.e_above_hull_dft = None
        self.e_above_hull_dft_corrected = None
        self.is_stable_dft = False
        self.is_metastable_dft = False

def extract_id_from_filename(filename: str) -> int:
    """Extract numeric ID from filename"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def get_compat_params(dft_path: Path) -> Dict:
    """Get DFT compatibility parameters"""
    try:
        incar_file = dft_path / "INCAR"
        poscar_file = dft_path / "POSCAR"
        
        if not incar_file.exists() or not poscar_file.exists():
            return {"hubbards": {}, "is_hubbard": False}
        
        incar = Incar.from_file(str(incar_file))
        poscar = Poscar.from_file(str(poscar_file))
        
        param = {"hubbards": {}}
        if "LDAUU" in incar:
            param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
        param["is_hubbard"] = (
            incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
        )
        if param["is_hubbard"]:
            param["run_type"] = "GGA+U"
        
        return param
        
    except Exception:
        return {"hubbards": {}, "is_hubbard": False}

def get_energy_correction(traj_file: Path, dft_input_path: Path) -> Tuple[ComputedStructureEntry, float, PDEntry]:
    """Get DFT energy correction"""
    traj = Trajectory(str(traj_file))
    ase_atoms = traj[-1]
    
    params = get_compat_params(dft_input_path)
    struct = AseAtomsAdaptor.get_structure(ase_atoms)
    
    cse_d = {
        "structure": struct,
        "energy": ase_atoms.get_potential_energy(),
        "correction": 0.0,
        "parameters": params,
    }
    cse = ComputedStructureEntry.from_dict(cse_d)
    
    # Apply MP2020 compatibility correction
    out = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse, clean=True
    )
    
    corrected_energy = cse.energy
    if out:
        corrected_energy = out[0].energy if isinstance(out, list) else out.energy
    
    pde = PDEntry(composition=cse.composition, energy=corrected_energy)
    
    return cse, corrected_energy, pde

def get_e_hull_from_phase_diagram(phase_diagram: PatchedPhaseDiagram, structure: Structure) -> float:
    """Calculate hull energy"""
    try:
        return phase_diagram.get_hull_energy_per_atom(structure.composition)
    except (ValueError, AttributeError, ZeroDivisionError):
        return float("nan")

def get_e_hull_per_atom_from_pymatgen(phase_diagram: PatchedPhaseDiagram, pde: PDEntry) -> Tuple[Dict, float]:
    """Calculate hull energy from pymatgen"""
    try:
        out = phase_diagram.get_decomp_and_e_above_hull(pde, allow_negative=True)
    except (ValueError, AttributeError, ZeroDivisionError):
        out = ({}, float("nan"))
    return out

class DFTPostProcessor:
    """DFT post-processor main class"""
    
    def __init__(self, chgnet_csv: str, dft_results_folder: str, original_csv_file: str,
                 mp_hull_path: str, output_dir: str, un_only: bool = False,
                 stability_threshold: float = 0.0, metastability_threshold: float = 0.1):
        
        self.chgnet_csv = Path(chgnet_csv)
        self.dft_results_folder = Path(dft_results_folder)
        self.original_csv_file = Path(original_csv_file)
        self.mp_hull_path = Path(mp_hull_path)
        self.output_dir = Path(output_dir)
        self.un_only = un_only
        self.stability_threshold = stability_threshold
        self.metastability_threshold = metastability_threshold
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Check input files and calculate UN ratios
        self._validate_inputs_and_calculate_ratios()
        
        print(f"DFT post-processor initialized successfully")
        print(f"   CHGNet results: {self.chgnet_csv}")
        print(f"   DFT results folder: {self.dft_results_folder}")
        print(f"   Original CSV file: {self.original_csv_file}")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        print(f"   Dataset UN ratio: {self.un_ratio_dataset:.4f}")
        if not self.un_only:
            print(f"   Dataset non-UN ratio: {self.non_un_ratio_dataset:.4f}")
        print(f"   Output directory: {self.output_dir}")
    
    def _validate_inputs_and_calculate_ratios(self):
        """Validate input files and calculate UN/non-UN ratios"""
        if not self.chgnet_csv.exists():
            raise FileNotFoundError(f"CHGNet results CSV does not exist: {self.chgnet_csv}")
        
        if not self.dft_results_folder.exists():
            raise FileNotFoundError(f"DFT results folder does not exist: {self.dft_results_folder}")
        
        if not self.original_csv_file.exists():
            raise FileNotFoundError(f"Original CSV file does not exist: {self.original_csv_file}")
        
        if not self.mp_hull_path.exists():
            raise FileNotFoundError(f"MP hull data does not exist: {self.mp_hull_path}")
        
        # Calculate UN/non-UN ratios from original CSV
        original_df = pd.read_csv(self.original_csv_file)
        total_structures = len(original_df)
        un_structures = len(original_df[original_df['is_unique_and_novel'] == True])
        non_un_structures = total_structures - un_structures
        
        self.un_ratio_dataset = un_structures / total_structures
        self.non_un_ratio_dataset = non_un_structures / total_structures
        
        print(f"Dataset statistics (from original CSV):")
        print(f"   Total structures: {total_structures:,}")
        print(f"   UN structures: {un_structures:,} ({self.un_ratio_dataset:.1%})")
        if not self.un_only:
            print(f"   non-UN structures: {non_un_structures:,} ({self.non_un_ratio_dataset:.1%})")
    
    def load_chgnet_results(self) -> pd.DataFrame:
        """Read CHGNet evaluation results"""
        print("Reading CHGNet evaluation results...")
        
        df = pd.read_csv(self.chgnet_csv)
        print(f"   Total structures: {len(df)}")
        print(f"   Successful relaxations: {len(df[df['success'] == True])}")
        print(f"   Selected for DFT: {len(df[df['selected_for_dft'] == True])}")
        
        if self.un_only:
            un_structures = len(df[df['is_un'] == True])
            print(f"   UN structures: {un_structures}")
            print(f"   UN structures selected for DFT: {len(df[(df['selected_for_dft'] == True) & (df['is_un'] == True)])}")
        
        return df
    
    def scan_dft_results(self) -> Dict[int, Path]:
        """Scan DFT result files and build ID mapping"""
        print("Scanning DFT result files...")
        
        dft_files = {}
        
        # Select folders to scan based on mode
        if self.un_only:
            scan_folders = ["dft_un"]
        else:
            scan_folders = ["dft_un", "dft_non_un"]
        
        for subfolder in scan_folders:
            subfolder_path = self.dft_results_folder / subfolder
            if not subfolder_path.exists():
                print(f"WARNING: DFT subfolder does not exist: {subfolder_path}")
                continue
            
            # Search for .traj files
            traj_files = list(subfolder_path.glob("**/*.traj"))
            for traj_file in traj_files:
                try:
                    # Extract ID from filename
                    structure_id = extract_id_from_filename(traj_file.stem)
                    dft_files[structure_id] = traj_file
                except:
                    continue
        
        print(f"   Found DFT result files: {len(dft_files)}")
        if self.un_only:
            print(f"   Scan mode: UN structures only (dft_un/)")
        else:
            print(f"   Scan mode: UN + non-UN structures (dft_un/ + dft_non_un/)")
        
        return dft_files
    
    def process_dft_results(self, chgnet_df: pd.DataFrame, dft_files: Dict[int, Path]) -> List[DFTResult]:
        """Process DFT results and apply energy corrections"""
        print("Processing DFT results and applying energy corrections...")
        
        # Filter structures that need DFT processing
        dft_candidates = chgnet_df[chgnet_df['selected_for_dft'] == True].copy()
        
        # Further filter for UN structures only in UN_only mode
        if self.un_only:
            dft_candidates = dft_candidates[dft_candidates['is_un'] == True]
        
        print(f"   DFT structures to process: {len(dft_candidates)}")
        if self.un_only:
            print(f"   Processing mode: UN structures only")
        
        results = []
        
        for _, row in tqdm(dft_candidates.iterrows(), total=len(dft_candidates), desc="Processing DFT results"):
            structure_id = int(row['dft_folder_id'])
            result = DFTResult(structure_id, row['filename'])
            
            try:
                # Find corresponding DFT result file
                if structure_id not in dft_files:
                    result.error_message = f"DFT result file not found: {structure_id}"
                    results.append(result)
                    continue
                
                traj_file = dft_files[structure_id]
                
                # Determine DFT input folder path
                if row['is_un']:
                    dft_input_path = self.dft_results_folder / "dft_un" / f"{structure_id:06d}"
                else:
                    if self.un_only:
                        # Should not have non-UN structures in UN_only mode
                        result.error_message = f"non-UN structure in UN_only mode: {structure_id}"
                        results.append(result)
                        continue
                    dft_input_path = self.dft_results_folder / "dft_non_un" / f"{structure_id:06d}"
                
                # Read DFT trajectory
                traj = Trajectory(str(traj_file))
                result.initial_energy_dft = traj[0].get_potential_energy()
                result.final_energy_dft = traj[-1].get_potential_energy()
                result.final_structure_dft = AseAtomsAdaptor.get_structure(traj[-1])
                
                # Apply energy correction
                cse, corrected_energy, pde = get_energy_correction(traj_file, dft_input_path)
                result.corrected_energy_dft = corrected_energy
                result.computed_structure_entry = cse
                result.phase_diagram_entry = pde
                
                result.success = True
                
            except Exception as e:
                result.error_message = str(e)
                result.success = False
            
            results.append(result)
        
        successful_count = len([r for r in results if r.success])
        print(f"   Successfully processed: {successful_count} / {len(results)}")
        
        return results
    
    def calculate_dft_hull_energies(self, dft_results: List[DFTResult]) -> List[DFTResult]:
        """Calculate DFT-based hull energies"""
        print("Calculating DFT-based hull energies...")
        
        # Load hull data
        with open(self.mp_hull_path, 'rb') as f:
            ppd_mp = pickle.load(f)
        
        for result in tqdm(dft_results, desc="Calculating DFT hull energies"):
            if not result.success:
                continue
            
            try:
                # Calculate uncorrected DFT hull energy
                if result.final_structure_dft:
                    e_hull_per_atom = get_e_hull_from_phase_diagram(ppd_mp, result.final_structure_dft)
                    if not np.isnan(e_hull_per_atom):
                        e_per_atom_dft = result.final_energy_dft / len(result.final_structure_dft)
                        result.e_above_hull_dft = e_per_atom_dft - e_hull_per_atom
                
                # Calculate corrected DFT hull energy
                if result.phase_diagram_entry:
                    decomp, e_above_hull_corrected = get_e_hull_per_atom_from_pymatgen(
                        ppd_mp, result.phase_diagram_entry
                    )
                    result.e_above_hull_dft_corrected = e_above_hull_corrected
                    
                    # Determine stability (based on corrected DFT results)
                    if not np.isnan(e_above_hull_corrected):
                        result.is_stable_dft = e_above_hull_corrected <= self.stability_threshold
                        result.is_metastable_dft = e_above_hull_corrected <= self.metastability_threshold
                
            except Exception as e:
                print(f"WARNING: Failed to calculate hull energy {result.filename}: {e}")
        
        return dft_results
    
    def calculate_final_sun_metrics(self, chgnet_df: pd.DataFrame, dft_results: List[DFTResult]) -> Dict:
        """Calculate final SUN metrics and complete overall stability metrics"""
        print("Calculating final SUN metrics and overall stability...")
        
        # Create DFT result mapping
        dft_result_map = {r.structure_id: r for r in dft_results if r.success}
        
        # Statistics
        total_structures = len(chgnet_df)
        un_structures = len(chgnet_df[chgnet_df['is_un'] == True])
        non_un_structures = total_structures - un_structures
        
        # CHGNet stage statistics
        chgnet_stable_un = len(chgnet_df[(chgnet_df['is_un'] == True) & (chgnet_df['is_stable'] == True)])
        chgnet_metastable_un = len(chgnet_df[(chgnet_df['is_un'] == True) & (chgnet_df['is_metastable'] == True)])
        
        if not self.un_only:
            chgnet_stable_non_un = len(chgnet_df[(chgnet_df['is_un'] == False) & (chgnet_df['is_stable'] == True)])
            chgnet_metastable_non_un = len(chgnet_df[(chgnet_df['is_un'] == False) & (chgnet_df['is_metastable'] == True)])
        else:
            chgnet_stable_non_un = 0
            chgnet_metastable_non_un = 0
        
        # DFT stage statistics
        dft_processed = len([r for r in dft_results if r.success])
        dft_stable_total = len([r for r in dft_results if r.is_stable_dft])
        dft_metastable_total = len([r for r in dft_results if r.is_metastable_dft])
        
        # Separate UN and non-UN DFT results
        dft_stable_un = 0
        dft_metastable_un = 0
        dft_stable_non_un = 0
        dft_metastable_non_un = 0
        dft_processed_un = 0
        dft_processed_non_un = 0
        
        for _, row in chgnet_df.iterrows():
            if row['selected_for_dft'] and int(row['dft_folder_id']) in dft_result_map:
                dft_result = dft_result_map[int(row['dft_folder_id'])]
                if row['is_un']:
                    dft_processed_un += 1
                    if dft_result.is_stable_dft:
                        dft_stable_un += 1
                    if dft_result.is_metastable_dft:
                        dft_metastable_un += 1
                else:
                    if not self.un_only:
                        dft_processed_non_un += 1
                        if dft_result.is_stable_dft:
                            dft_stable_non_un += 1
                        if dft_result.is_metastable_dft:
                            dft_metastable_non_un += 1
        
        # Calculate stability rates in test set (correct logic: use test set total as denominator)
        un_stable_rate_test = (dft_stable_un / un_structures) if un_structures > 0 else 0
        un_metastable_rate_test = (dft_metastable_un / un_structures) if un_structures > 0 else 0
        
        if not self.un_only:
            non_un_stable_rate_test = (dft_stable_non_un / non_un_structures) if non_un_structures > 0 else 0
            non_un_metastable_rate_test = (dft_metastable_non_un / non_un_structures) if non_un_structures > 0 else 0
        else:
            non_un_stable_rate_test = 0
            non_un_metastable_rate_test = 0
        
        # Calculate final metrics
        # SUN/MSUN (only UN structures contribute)
        final_sun_rate = self.un_ratio_dataset * un_stable_rate_test
        final_msun_rate = self.un_ratio_dataset * un_metastable_rate_test
        
        # Overall stability metrics (weighted average of UN + non-UN structures)
        if not self.un_only:
            overall_stability_rate = (self.un_ratio_dataset * un_stable_rate_test + 
                                     self.non_un_ratio_dataset * non_un_stable_rate_test)
            overall_metastability_rate = (self.un_ratio_dataset * un_metastable_rate_test + 
                                         self.non_un_ratio_dataset * non_un_metastable_rate_test)
        else:
            # In UN_only mode, overall stability metrics are the same as UN stability metrics
            overall_stability_rate = final_sun_rate
            overall_metastability_rate = final_msun_rate
        
        metrics = {
            'processing_mode': 'un_only' if self.un_only else 'un_and_non_un',
            'dataset_info': {
                'un_ratio_dataset': self.un_ratio_dataset,
                'non_un_ratio_dataset': self.non_un_ratio_dataset if not self.un_only else 0,
                'total_structures_tested': total_structures,
                'un_structures_tested': un_structures,
                'non_un_structures_tested': non_un_structures if not self.un_only else 0
            },
            'chgnet_results': {
                'stable_un_chgnet': chgnet_stable_un,
                'metastable_un_chgnet': chgnet_metastable_un,
                'stable_non_un_chgnet': chgnet_stable_non_un,
                'metastable_non_un_chgnet': chgnet_metastable_non_un
            },
            'dft_results': {
                'dft_processed_total': dft_processed,
                'dft_processed_un': dft_processed_un,
                'dft_processed_non_un': dft_processed_non_un,
                'stable_total_dft': dft_stable_total,
                'metastable_total_dft': dft_metastable_total,
                'stable_un_dft': dft_stable_un,
                'metastable_un_dft': dft_metastable_un,
                'stable_non_un_dft': dft_stable_non_un,
                'metastable_non_un_dft': dft_metastable_non_un
            },
            'test_set_rates': {
                'un_stable_rate_test': un_stable_rate_test,
                'un_metastable_rate_test': un_metastable_rate_test,
                'non_un_stable_rate_test': non_un_stable_rate_test,
                'non_un_metastable_rate_test': non_un_metastable_rate_test
            },
            'final_metrics': {
                'final_sun_rate': final_sun_rate,
                'final_msun_rate': final_msun_rate,
                'overall_stability_rate': overall_stability_rate,
                'overall_metastability_rate': overall_metastability_rate
            },
            'thresholds': {
                'stability_threshold': self.stability_threshold,
                'metastability_threshold': self.metastability_threshold
            }
        }
        
        print(f"   Test set stability rates:")
        print(f"      UN stability rate: {un_stable_rate_test:.1%} ({dft_stable_un}/{un_structures})")
        print(f"      UN metastability rate: {un_metastable_rate_test:.1%} ({dft_metastable_un}/{un_structures})")
        if not self.un_only:
            print(f"      non-UN stability rate: {non_un_stable_rate_test:.1%} ({dft_stable_non_un}/{non_un_structures})")
            print(f"      non-UN metastability rate: {non_un_metastable_rate_test:.1%} ({dft_metastable_non_un}/{non_un_structures})")
        print(f"   Note: Denominator is test set total, failed/unselected DFT structures counted as unstable")
        
        print(f"   Final overall metrics:")
        print(f"      SUN rate: {self.un_ratio_dataset:.1%} × {un_stable_rate_test:.1%} = {final_sun_rate:.2%}")
        print(f"      MSUN rate: {self.un_ratio_dataset:.1%} × {un_metastable_rate_test:.1%} = {final_msun_rate:.2%}")
        if not self.un_only:
            print(f"      Overall stability rate: ({self.un_ratio_dataset:.1%} × {un_stable_rate_test:.1%}) + ({self.non_un_ratio_dataset:.1%} × {non_un_stable_rate_test:.1%}) = {overall_stability_rate:.2%}")
            print(f"      Overall metastability rate: ({self.un_ratio_dataset:.1%} × {un_metastable_rate_test:.1%}) + ({self.non_un_ratio_dataset:.1%} × {non_un_metastable_rate_test:.1%}) = {overall_metastability_rate:.2%}")
        else:
            print(f"      Overall stability rate (UN_only): {overall_stability_rate:.2%}")
            print(f"      Overall metastability rate (UN_only): {overall_metastability_rate:.2%}")
        
        return metrics
    
    def generate_final_report(self, chgnet_df: pd.DataFrame, dft_results: List[DFTResult], metrics: Dict):
        """Generate final complete evaluation report"""
        print("Generating final evaluation report...")
        
        # Create DFT result mapping
        dft_result_map = {r.structure_id: r for r in dft_results}
        
        # Prepare final report data
        report_data = []
        
        for _, row in chgnet_df.iterrows():
            # Skip non-UN structures in UN_only mode
            if self.un_only and not row['is_un']:
                continue
                
            # Basic information
            record = {
                'filename': row['filename'],
                'is_un': row['is_un'],
                'chgnet_success': row['success'],
                
                # Original and CHGNet structure/energy
                'original_structure_cif': row['initial_cif'],
                'original_energy': row['initial_energy'],
                'chgnet_structure_cif': row['final_cif'],
                'chgnet_energy': row['final_energy'],
                'chgnet_n_steps': row['n_steps'],
                'chgnet_converged': row['converged'],
                
                # CHGNet stability
                'e_above_hull_chgnet': row['e_above_hull_chgnet'],
                'is_stable_chgnet': row['is_stable'],
                'is_metastable_chgnet': row['is_metastable'],
                'selected_for_dft': row['selected_for_dft'],
                
                # DFT related information (initialize as empty)
                'dft_processed': False,
                'dft_structure_cif': '',
                'dft_energy_raw': None,
                'dft_energy_corrected': None,
                'e_above_hull_dft_raw': None,
                'e_above_hull_dft_corrected': None,
                'is_stable_dft': False,
                'is_metastable_dft': False,
                'dft_error_message': ''
            }
            
            # Fill DFT information if available
            if row['selected_for_dft'] and pd.notna(row['dft_folder_id']):
                structure_id = int(row['dft_folder_id'])
                if structure_id in dft_result_map:
                    dft_result = dft_result_map[structure_id]
                    record.update({
                        'dft_processed': dft_result.success,
                        'dft_structure_cif': self._structure_to_cif_string(dft_result.final_structure_dft),
                        'dft_energy_raw': dft_result.final_energy_dft,
                        'dft_energy_corrected': dft_result.corrected_energy_dft,
                        'e_above_hull_dft_raw': dft_result.e_above_hull_dft,
                        'e_above_hull_dft_corrected': dft_result.e_above_hull_dft_corrected,
                        'is_stable_dft': dft_result.is_stable_dft,
                        'is_metastable_dft': dft_result.is_metastable_dft,
                        'dft_error_message': dft_result.error_message or ''
                    })
            
            report_data.append(record)
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)
        
        # Save detailed CSV report
        csv_file = self.output_dir / "final_evaluation_report.csv"
        report_df.to_csv(csv_file, index=False)
        print(f"   Detailed CSV report: {csv_file}")
        
        # Save JSON statistics
        json_file = self.output_dir / "final_sun_metrics.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"   JSON statistics: {json_file}")
        
        # Generate simplified summary table
        summary_data = {
            'processing_mode': 'un_only' if self.un_only else 'un_and_non_un',
            'total_structures': len(report_df),
            'un_structures': len(report_df[report_df['is_un'] == True]),
            'non_un_structures': len(report_df[report_df['is_un'] == False]) if not self.un_only else 0,
            'chgnet_successful': len(report_df[report_df['chgnet_success'] == True]),
            'dft_processed': len(report_df[report_df['dft_processed'] == True]),
            'final_stable_un': len(report_df[(report_df['is_un'] == True) & (report_df['is_stable_dft'] == True)]),
            'final_metastable_un': len(report_df[(report_df['is_un'] == True) & (report_df['is_metastable_dft'] == True)]),
            'final_stable_non_un': len(report_df[(report_df['is_un'] == False) & (report_df['is_stable_dft'] == True)]) if not self.un_only else 0,
            'final_metastable_non_un': len(report_df[(report_df['is_un'] == False) & (report_df['is_metastable_dft'] == True)]) if not self.un_only else 0,
            'final_sun_rate': metrics['final_metrics']['final_sun_rate'],
            'final_msun_rate': metrics['final_metrics']['final_msun_rate'],
            'overall_stability_rate': metrics['final_metrics']['overall_stability_rate'],
            'overall_metastability_rate': metrics['final_metrics']['overall_metastability_rate']
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_csv = self.output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"   Summary results: {summary_csv}")
        
        return report_df
    
    def _structure_to_cif_string(self, structure: Structure) -> str:
        """Convert Structure to CIF string"""
        if structure is None:
            return ""
        try:
            writer = CifWriter(structure)
            return writer.__str__()
        except:
            return ""
    
    def run_complete_post_processing(self):
        """Run complete DFT post-processing workflow"""
        print("Starting complete DFT post-processing workflow")
        print("="*60)
        
        start_time = time.time()
        
        # 1. Read CHGNet results
        chgnet_df = self.load_chgnet_results()
        
        # 2. Scan DFT result files
        dft_files = self.scan_dft_results()
        
        if not dft_files:
            print("ERROR: No DFT result files found, please check if DFT calculations are complete")
            return
        
        # 3. Process DFT results and apply energy corrections
        dft_results = self.process_dft_results(chgnet_df, dft_files)
        
        # 4. Calculate DFT-based hull energies
        dft_results = self.calculate_dft_hull_energies(dft_results)
        
        # 5. Calculate final SUN metrics
        metrics = self.calculate_final_sun_metrics(chgnet_df, dft_results)
        
        # 6. Generate final evaluation report
        report_df = self.generate_final_report(chgnet_df, dft_results, metrics)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print final results
        print("="*60)
        print(f"DFT post-processing completed!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        print(f"   Processed structures: {len(chgnet_df)}")
        print(f"   DFT successful: {len([r for r in dft_results if r.success])}")
        print(f"\nFinal evaluation results:")
        print(f"   SUN rate: {metrics['final_metrics']['final_sun_rate']:.2%}")
        print(f"   MSUN rate: {metrics['final_metrics']['final_msun_rate']:.2%}")
        if not self.un_only:
            print(f"   Overall stability rate: {metrics['final_metrics']['overall_stability_rate']:.2%}")
            print(f"   Overall metastability rate: {metrics['final_metrics']['overall_metastability_rate']:.2%}")
        else:
            print(f"   Overall stability rate (UN_only): {metrics['final_metrics']['overall_stability_rate']:.2%}")
            print(f"   Overall metastability rate (UN_only): {metrics['final_metrics']['overall_metastability_rate']:.2%}")
        print(f"   Results saved in: {self.output_dir}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="DFT Post-Processor - Second Stage Evaluation")
    
    # Required parameters
    parser.add_argument("--chgnet_csv", type=str, required=True,
                       help="CHGNet evaluation results CSV file path")
    parser.add_argument("--dft_results_folder", type=str, required=True,
                       help="DFT calculation results folder path")
    parser.add_argument("--original_csv_file", type=str, required=True,
                       help="Original UN results CSV file path")
    parser.add_argument("--mp_hull_path", type=str, required=True,
                       help="MP hull data file path (.pkl)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Final results output directory path")
    
    # Optional parameters
    parser.add_argument("--un_only", action="store_true",
                       help="Process UN structures only (default: process both UN and non-UN)")
    parser.add_argument("--stability_threshold", type=float, default=0.0,
                       help="Stability threshold eV/atom (default: 0.0)")
    parser.add_argument("--metastability_threshold", type=float, default=0.1,
                       help="Metastability threshold eV/atom (default: 0.1)")
    
    args = parser.parse_args()
    
    # Create post-processor and run
    try:
        processor = DFTPostProcessor(
            chgnet_csv=args.chgnet_csv,
            dft_results_folder=args.dft_results_folder,
            original_csv_file=args.original_csv_file,
            mp_hull_path=args.mp_hull_path,
            output_dir=args.output_dir,
            un_only=args.un_only,
            stability_threshold=args.stability_threshold,
            metastability_threshold=args.metastability_threshold
        )
        
        processor.run_complete_post_processing()
        
    except Exception as e:
        print(f"ERROR: Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
