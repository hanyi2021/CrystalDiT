#!/usr/bin/env python3
"""
CHGNet Processor - First Stage Evaluation
Functions:
1. Smart sampling based on UN results
2. Multi-GPU parallel CHGNet relaxation
3. Hull stability calculation
4. DFT input file generation

Prerequisites:
Configure pymatgen VASP pseudopotential path:
    cd /path/to/your/vasp_potentials
    pmg config --add PMG_VASP_PSP_DIR $(pwd)

Usage:
python ./eval_script/chgnet_process.py \
    --cif_folder ./5w_1dit_results/5w_1dit_crystals \
    --csv_file ./5w_1dit_results_optimized_analysis/5w_1dit_crystals_results.csv \
    --mp_hull_path /your/mp_hull_path \
    --sample_size 500 \
    --max_steps 2500 \
    --dft_threshold 0.5 \
    --output_dir ./5w_chgnet_results500 \
    --un_only #optinal
"""

import os
import sys
import json
import pickle
import warnings
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from multiprocessing import Process
from functools import partial
import argparse

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import contextlib
import io

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from chgnet.model import CHGNet, StructOptimizer

# Suppress warnings and output
warnings.filterwarnings("ignore")
logging.getLogger("chgnet").setLevel(logging.CRITICAL)
logging.getLogger("pymatgen").setLevel(logging.CRITICAL)
logging.getLogger("ase").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def suppress_all_output():
    """Completely suppress all output"""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield

class CHGNetRelaxationResult:
    """CHGNet relaxation result data structure - process-safe version"""
    def __init__(self, filename: str, is_un: bool):
        self.filename = filename
        self.is_un = is_un
        self.success = False
        self.error_message = None
        
        # Structure information (stored as dict to avoid CUDA tensor issues)
        self.initial_structure_dict = None
        self.final_structure_dict = None
        self.initial_energy = None
        self.final_energy = None
        self.n_steps = None
        self.converged = False
        
        # Stability information
        self.e_hull_per_atom = None
        self.e_above_hull_per_atom = None
        self.is_stable = False
        self.is_metastable = False
        self.selected_for_dft = False
        self.dft_folder_id = None
    
    def set_initial_structure(self, structure: Structure):
        """Set initial structure (convert to dict for storage)"""
        if structure is not None:
            self.initial_structure_dict = structure.as_dict()
    
    def set_final_structure(self, structure: Structure):
        """Set final structure (convert to dict for storage)"""
        if structure is not None:
            self.final_structure_dict = structure.as_dict()
    
    def get_initial_structure(self) -> Optional[Structure]:
        """Get initial structure object"""
        if self.initial_structure_dict:
            try:
                return Structure.from_dict(self.initial_structure_dict)
            except:
                return None
        return None
    
    def get_final_structure(self) -> Optional[Structure]:
        """Get final structure object"""
        if self.final_structure_dict:
            try:
                return Structure.from_dict(self.final_structure_dict)
            except:
                return None
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dict format for CSV saving"""
        return {
            'filename': self.filename,
            'is_un': self.is_un,
            'initial_cif': self._structure_dict_to_cif_string(self.initial_structure_dict),
            'initial_energy': self.initial_energy,
            'final_cif': self._structure_dict_to_cif_string(self.final_structure_dict),
            'final_energy': self.final_energy,
            'n_steps': self.n_steps,
            'converged': self.converged,
            'e_above_hull_chgnet': self.e_above_hull_per_atom,
            'is_stable': self.is_stable,
            'is_metastable': self.is_metastable,
            'selected_for_dft': self.selected_for_dft,
            'dft_folder_id': self.dft_folder_id,
            'success': self.success,
            'error_message': self.error_message
        }
    
    def _structure_dict_to_cif_string(self, structure_dict: Dict) -> str:
        """Convert Structure dict to CIF string"""
        if structure_dict is None:
            return ""
        try:
            structure = Structure.from_dict(structure_dict)
            writer = CifWriter(structure)
            return writer.__str__()
        except:
            return ""

def gpu_worker(rank: int, structure_batch: List[Tuple], mp_hull_path: str, 
               max_steps: int, output_file: str, log_file: str):
    """GPU worker process - correct multi-GPU parallel implementation"""
    
    # Setup logging
    import logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format=f'%(asctime)s - GPU{rank} - %(message)s',
        filemode='w'
    )
    logger = logging.getLogger(f'GPU_{rank}')
    
    try:
        # Critical: First set CUDA device before loading any models
        if torch.cuda.is_available() and rank is not None:
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
            logger.info(f"Set GPU device: {device}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.current_device()}")
        else:
            device = "cpu"
            rank = "CPU"
            logger.info("Using CPU device")
        
        # Load model and data (after device setup)
        logger.info("Loading CHGNet model...")
        with suppress_all_output():
            chgnet = CHGNet.load()
            if device != "cpu":
                chgnet = chgnet.to(device)
                logger.info(f"CHGNet model loaded to {device}")
            else:
                logger.info("CHGNet model loaded to CPU")
        
        logger.info("Loading MP hull data...")
        with open(mp_hull_path, 'rb') as f:
            ppd_mp = pickle.load(f)
        logger.info("MP hull data loaded")
        
        # Verify GPU usage
        if device != "cpu":
            logger.info(f"GPU memory usage: {torch.cuda.memory_allocated(rank) / 1024**3:.2f} GB")
            logger.info(f"GPU memory cache: {torch.cuda.memory_reserved(rank) / 1024**3:.2f} GB")
        
        # Process assigned structures
        logger.info(f"Processing {len(structure_batch)} structures")
        
        # Statistics for assigned structures
        un_count = sum(1 for _, _, is_un in structure_batch if is_un)
        non_un_count = len(structure_batch) - un_count
        logger.info(f"Structure assignment: UN={un_count}, non-UN={non_un_count}")
        
        results = []
        
        for i, (cif_file, cif_path, is_un) in enumerate(structure_batch):
            if (i + 1) % 10 == 0:  # Progress every 10 structures
                logger.info(f"Progress: {i+1}/{len(structure_batch)} ({(i+1)/len(structure_batch)*100:.1f}%)")
            
            result = CHGNetRelaxationResult(cif_file, is_un)
            
            try:
                # Read CIF file
                parser = CifParser(str(cif_path))
                initial_structure = parser.get_structures()[0]
                result.set_initial_structure(initial_structure)
                
                logger.info(f"Starting relaxation {cif_file}: {initial_structure.num_sites} atoms, type: {'UN' if is_un else 'non-UN'}")
                
                start_relax_time = time.time()
                
                # CHGNet relaxation
                with suppress_all_output():
                    # Initial energy prediction
                    prediction = chgnet.predict_structure(initial_structure)
                    result.initial_energy = prediction["e"] * initial_structure.num_sites
                    
                    # Structure relaxation
                    relaxer = StructOptimizer(model=chgnet)
                    relaxation = relaxer.relax(initial_structure, steps=max_steps)
                    
                    final_structure = relaxation["final_structure"]
                    result.set_final_structure(final_structure)
                    result.final_energy = relaxation["trajectory"].energies[-1]
                    result.n_steps = len(relaxation["trajectory"].energies)
                    result.converged = result.n_steps < max_steps
                
                relax_time = time.time() - start_relax_time
                logger.info(f"Relaxation completed {cif_file}: {result.n_steps} steps, converged: {result.converged}, time: {relax_time:.2f}s")
                
                # Calculate hull energy (using final structure)
                e_hull_per_atom = get_e_hull_from_phase_diagram(ppd_mp, final_structure)
                result.e_hull_per_atom = e_hull_per_atom
                
                if not np.isnan(e_hull_per_atom):
                    e_per_atom = result.final_energy / len(final_structure)
                    result.e_above_hull_per_atom = e_per_atom - e_hull_per_atom
                    logger.info(f"Hull energy calculated {cif_file}: {result.e_above_hull_per_atom:.4f} eV/atom")
                
                result.success = True
                
            except Exception as e:
                error_msg = str(e)
                result.error_message = error_msg
                result.success = False
                logger.error(f"Processing failed {cif_file}: {error_msg}")
            
            results.append(result)
        
        logger.info(f"GPU {rank} processing completed: {len([r for r in results if r.success])}/{len(results)} successful")
        
        # Separate statistics for UN and non-UN processing results
        un_success = len([r for r in results if r.success and r.is_un])
        un_total = len([r for r in results if r.is_un])
        non_un_success = len([r for r in results if r.success and not r.is_un])
        non_un_total = len([r for r in results if not r.is_un])
        
        logger.info(f"UN structure processing: {un_success}/{un_total} successful")
        logger.info(f"non-UN structure processing: {non_un_success}/{non_un_total} successful")
        
        # Save results to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to: {output_file}")
        
        # Clean GPU memory
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
                logger.info(f"GPU {rank} memory cleaned")
            except:
                pass
                
    except Exception as e:
        error_msg = f"GPU {rank} worker failed: {str(e)}"
        logger.error(error_msg)
        
        # Save error result
        error_result = CHGNetRelaxationResult("ERROR", False)
        error_result.error_message = error_msg
        with open(output_file, 'wb') as f:
            pickle.dump([error_result], f)

def get_e_hull_from_phase_diagram(phase_diagram: PatchedPhaseDiagram, structure: Structure) -> float:
    """FlowMM compatible hull energy calculation"""
    try:
        return phase_diagram.get_hull_energy_per_atom(structure.composition)
    except (ValueError, AttributeError, ZeroDivisionError):
        return float("nan")

def extract_id_from_filename(filename: str) -> int:
    """Extract numeric ID from filename"""
    import re
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

class CHGNetProcessor:
    """CHGNet processor main class"""
    
    def __init__(self, cif_folder: str, csv_file: str, mp_hull_path: str, 
                 sample_size: int, output_dir: str, un_only: bool = False,
                 max_steps: int = 1500, stability_threshold: float = 0.0,
                 metastability_threshold: float = 0.1, dft_threshold: float = 0.1):
        
        self.cif_folder = Path(cif_folder)
        self.csv_file = Path(csv_file)
        self.mp_hull_path = Path(mp_hull_path)
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.un_only = un_only
        self.max_steps = max_steps
        self.stability_threshold = stability_threshold
        self.metastability_threshold = metastability_threshold
        self.dft_threshold = dft_threshold
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "dft_un").mkdir(exist_ok=True)
        if not self.un_only:
            (self.output_dir / "dft_non_un").mkdir(exist_ok=True)
        
        # Check input files and pymatgen configuration
        self._validate_inputs()
        
        # Detect GPUs
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_ids = list(range(self.num_gpus))
            print(f"   Detected {self.num_gpus} GPUs: {[f'cuda:{i}' for i in self.gpu_ids]}")
        else:
            self.num_gpus = 1
            self.gpu_ids = [None]
            print(f"   No GPUs detected, will use CPU")
        
        print(f"CHGNet processor initialized successfully")
        print(f"   CIF folder: {self.cif_folder}")
        print(f"   CSV file: {self.csv_file}")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        if self.un_only:
            print(f"   Sample size: {self.sample_size} (UN only)")
        else:
            print(f"   Sample size: {self.sample_size} (UN) + {self.sample_size} (non-UN)")
        print(f"   GPU count: {self.num_gpus}")
        print(f"   Output directory: {self.output_dir}")
    
    def _validate_inputs(self):
        """Validate input files, directories and pymatgen configuration"""
        if not self.cif_folder.exists():
            raise FileNotFoundError(f"CIF folder does not exist: {self.cif_folder}")
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_file}")
        
        if not self.mp_hull_path.exists():
            raise FileNotFoundError(f"MP hull data does not exist: {self.mp_hull_path}")
        
        # Check pymatgen VASP pseudopotential configuration
        from pymatgen.core import SETTINGS
        
        if "PMG_VASP_PSP_DIR" not in SETTINGS:
            print("ERROR: VASP pseudopotential configuration not detected")
            print("Please run the following commands to configure pymatgen:")
            print("   cd /path/to/your/vasp_potentials")
            print("   pmg config --add PMG_VASP_PSP_DIR $(pwd)")
            print("")
            print("The pseudopotential directory should contain POT_GGA_PAW_PBE/ subdirectories")
            raise RuntimeError("pymatgen VASP pseudopotential path not configured")
        
        potcar_dir = Path(SETTINGS["PMG_VASP_PSP_DIR"])
        if not potcar_dir.exists():
            raise FileNotFoundError(f"pymatgen configured pseudopotential directory does not exist: {potcar_dir}")
        
        # Check pseudopotential structure
        pot_dir = potcar_dir / "POT_GGA_PAW_PBE"
        if not pot_dir.exists():
            raise FileNotFoundError(f"Missing pseudopotential subdirectory: {pot_dir}")
        
        print(f"pymatgen VASP pseudopotential configuration correct: {potcar_dir}")
    
    def load_and_sample_structures(self) -> List[Tuple]:
        """Load CSV and intelligently sample structures, return mixed structure list"""
        print("Reading CSV file and sampling structures...")
        
        # Read CSV
        df = pd.read_csv(self.csv_file)
        print(f"   Total structures: {len(df)}")
        
        # Separate UN and non-UN structures
        un_df = df[df['is_unique_and_novel'] == True]
        non_un_df = df[df['is_unique_and_novel'] == False]
        
        print(f"   UN structures: {len(un_df)}")
        print(f"   non-UN structures: {len(non_un_df)}")
        
        # Sampling
        un_sample_size = min(self.sample_size, len(un_df))
        un_sampled = un_df.sample(n=un_sample_size, random_state=42) if un_sample_size > 0 else pd.DataFrame()
        
        if self.un_only:
            non_un_sampled = pd.DataFrame()
            non_un_sample_size = 0
            print(f"   Sampling result (UN only mode): {len(un_sampled)} UN")
        else:
            non_un_sample_size = min(self.sample_size, len(non_un_df))
            non_un_sampled = non_un_df.sample(n=non_un_sample_size, random_state=42) if non_un_sample_size > 0 else pd.DataFrame()
            print(f"   Sampling result: {len(un_sampled)} UN + {len(non_un_sampled)} non-UN = {len(un_sampled) + len(non_un_sampled)} total")
        
        # Build file path lists
        un_structures = []
        for _, row in un_sampled.iterrows():
            cif_path = self.cif_folder / row['filename']
            if cif_path.exists():
                un_structures.append((row['filename'], cif_path, True))
            else:
                print(f"WARNING: File does not exist: {cif_path}")
        
        non_un_structures = []
        if not self.un_only:
            for _, row in non_un_sampled.iterrows():
                cif_path = self.cif_folder / row['filename']
                if cif_path.exists():
                    non_un_structures.append((row['filename'], cif_path, False))
                else:
                    print(f"WARNING: File does not exist: {cif_path}")
        
        if self.un_only:
            print(f"   Actually processable (UN only mode): {len(un_structures)} UN")
        else:
            print(f"   Actually processable: {len(un_structures)} UN + {len(non_un_structures)} non-UN")
        
        # Mix structures for load balancing
        all_structures = un_structures + non_un_structures
        
        # Randomly shuffle all structures to ensure mixed workload for each GPU
        random.shuffle(all_structures)
        
        if self.un_only:
            print(f"   UN structures prepared")
        else:
            print(f"   Structures randomly mixed for GPU load balancing")
        
        print(f"   Expected structures per GPU: ~{len(all_structures)//self.num_gpus}")
        
        return all_structures
    
    def run_multi_gpu_relaxation(self, all_structures: List[Tuple]) -> List[CHGNetRelaxationResult]:
        """Multi-GPU parallel CHGNet relaxation - correct multi-GPU implementation"""
        if not all_structures:
            return []
        
        print(f"Starting multi-GPU parallel CHGNet relaxation...")
        print(f"   Total structures: {len(all_structures)}")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        print(f"   GPU count: {self.num_gpus}")
        print(f"   Max relaxation steps: {self.max_steps}")
        
        # Create log directory
        log_dir = self.output_dir / "gpu_logs"
        log_dir.mkdir(exist_ok=True)
        
        # Distribute structures to GPUs
        structures_per_gpu = len(all_structures) // self.num_gpus
        remainder = len(all_structures) % self.num_gpus
        
        processes = []
        output_files = []
        log_files = []
        start_idx = 0
        
        print(f"GPU task distribution (load balanced):")
        for rank in range(self.num_gpus):
            batch_size = structures_per_gpu + (1 if rank < remainder else 0)
            if batch_size == 0:
                continue
            
            batch = all_structures[start_idx:start_idx + batch_size]
            output_file = self.output_dir / f"gpu_{rank}_results.pkl"
            log_file = log_dir / f"gpu_{rank}.log"
            
            # Count UN and non-UN structures assigned to each GPU
            un_count = sum(1 for _, _, is_un in batch if is_un)
            non_un_count = len(batch) - un_count
            
            output_files.append(output_file)
            log_files.append(log_file)
            
            if self.un_only:
                print(f"   GPU {rank}: {len(batch)} structures (UN: {un_count}) → log: {log_file}")
            else:
                print(f"   GPU {rank}: {len(batch)} structures (UN: {un_count}, non-UN: {non_un_count}) → log: {log_file}")
            
            p = Process(
                target=gpu_worker,
                args=(rank, batch, str(self.mp_hull_path), self.max_steps, 
                      str(output_file), str(log_file))
            )
            p.start()
            processes.append(p)
            start_idx += batch_size
        
        print(f"\nMonitoring GPU process status...")
        print(f"   Log files location: {log_dir}")
        print(f"   Use 'tail -f {log_dir}/gpu_*.log' to view real-time progress")
        
        # Monitor processes
        import time
        while True:
            alive_count = sum(1 for p in processes if p.is_alive())
            if alive_count == 0:
                break
            print(f"   Running GPU processes: {alive_count}/{len(processes)}")
            time.sleep(30)  # Check every 30 seconds
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"All GPU processes completed")
        
        # Collect results
        all_results = []
        for output_file in output_files:
            if output_file.exists():
                try:
                    with open(output_file, 'rb') as f:
                        results = pickle.load(f)
                    all_results.extend(results)
                    # Clean temporary files
                    output_file.unlink()
                except Exception as e:
                    print(f"WARNING: Failed to read result file {output_file}: {e}")
        
        successful_count = len([r for r in all_results if r.success])
        un_processed = len([r for r in all_results if r.is_un])
        non_un_processed = len([r for r in all_results if not r.is_un])
        
        print(f"CHGNet relaxation completed:")
        print(f"   Successfully processed: {successful_count} / {len(all_results)} structures")
        print(f"   Success rate: {successful_count/len(all_results)*100:.1f}%")
        if self.un_only:
            print(f"   Processed structure types: UN={un_processed}")
        else:
            print(f"   Processed structure types: UN={un_processed}, non-UN={non_un_processed}")
        
        if self.un_only:
            print(f"   UN only mode: Structure processing completed")
        else:
            print(f"   Load balancing strategy: Structures mixed distributed to GPUs")
        
        return all_results
    
    def calculate_stability_and_filter(self, results: List[CHGNetRelaxationResult]) -> List[CHGNetRelaxationResult]:
        """Calculate stability and filter DFT candidate structures"""
        print("Calculating stability and filtering DFT candidate structures...")
        
        stable_count = 0
        metastable_count = 0
        dft_candidates = []
        
        for result in results:
            if result.success and result.e_above_hull_per_atom is not None:
                # Determine stability
                result.is_stable = result.e_above_hull_per_atom <= self.stability_threshold
                result.is_metastable = result.e_above_hull_per_atom <= self.metastability_threshold
                
                if result.is_stable:
                    stable_count += 1
                if result.is_metastable:
                    metastable_count += 1
                
                # Filter DFT candidates
                if result.e_above_hull_per_atom <= self.dft_threshold:
                    result.selected_for_dft = True
                    result.dft_folder_id = extract_id_from_filename(result.filename)
                    dft_candidates.append(result)
        
        print(f"   Stable structures: {stable_count}")
        print(f"   Metastable structures: {metastable_count}")
        print(f"   DFT candidate structures: {len(dft_candidates)}")
        
        if self.un_only:
            un_candidates = len([r for r in dft_candidates if r.is_un])
            print(f"   DFT candidate structures (UN only): {un_candidates}")
        
        return dft_candidates
    
    def generate_dft_inputs(self, dft_candidates: List[CHGNetRelaxationResult]):
        """Generate DFT input files"""
        if not dft_candidates:
            print("No qualifying DFT candidate structures")
            return
        
        print(f"Generating DFT input files for {len(dft_candidates)} structures...")
        
        un_count = 0
        non_un_count = 0
        
        for result in tqdm(dft_candidates, desc="Generating DFT inputs"):
            try:
                # Select output directory
                if result.is_un:
                    output_dir = self.output_dir / "dft_un"
                    un_count += 1
                else:
                    if self.un_only:
                        # Should not have non-UN structures in UN_only mode
                        print(f"WARNING: non-UN structure in UN_only mode: {result.filename}")
                        continue
                    output_dir = self.output_dir / "dft_non_un"
                    non_un_count += 1
                
                # Create DFT input directory
                dft_dir = output_dir / f"{result.dft_folder_id:06d}"
                dft_dir.mkdir(exist_ok=True)
                
                # Generate VASP inputs using relaxed structure
                final_structure = result.get_final_structure()
                if final_structure is None:
                    print(f"WARNING: Cannot get structure {result.filename}")
                    result.selected_for_dft = False
                    continue
                
                relax_set = MPRelaxSet(structure=final_structure)
                relax_set.write_input(
                    output_dir=str(dft_dir),
                    make_dir_if_not_present=True,
                    potcar_spec=False  # Generate actual POTCAR files
                )
                
            except Exception as e:
                print(f"WARNING: Failed to generate DFT input {result.filename}: {e}")
                result.selected_for_dft = False
        
        print(f"DFT input file generation completed:")
        print(f"   UN structures: {un_count} → {self.output_dir}/dft_un/")
        if not self.un_only:
            print(f"   non-UN structures: {non_un_count} → {self.output_dir}/dft_non_un/")
    
    def save_results(self, results: List[CHGNetRelaxationResult]):
        """Save results to CSV and JSON"""
        print("Saving results...")
        
        # Convert to DataFrame
        data = [result.to_dict() for result in results]
        df = pd.DataFrame(data)
        
        # Save CSV
        csv_file = self.output_dir / "chgnet_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"   CSV results: {csv_file}")
        
        # Read original CSV for large dataset statistics
        original_df = pd.read_csv(self.csv_file)
        total_structures_dataset = len(original_df)
        un_structures_dataset = len(original_df[original_df['is_unique_and_novel'] == True])
        un_ratio_dataset = un_structures_dataset / total_structures_dataset
        
        # Calculate statistics
        total_structures = len(results)
        successful_structures = len([r for r in results if r.success])
        un_structures = len([r for r in results if r.is_un])
        non_un_structures = total_structures - un_structures
        
        stable_count = len([r for r in results if r.is_stable])
        metastable_count = len([r for r in results if r.is_metastable])
        dft_count = len([r for r in results if r.selected_for_dft])
        
        # Separate UN and non-UN statistics
        un_stable = len([r for r in results if r.is_un and r.is_stable])
        un_metastable = len([r for r in results if r.is_un and r.is_metastable])
        un_dft = len([r for r in results if r.is_un and r.selected_for_dft])
        
        non_un_stable = len([r for r in results if not r.is_un and r.is_stable])
        non_un_metastable = len([r for r in results if not r.is_un and r.is_metastable])
        non_un_dft = len([r for r in results if not r.is_un and r.selected_for_dft])
        
        # Calculate UN stability and metastability rates in test set
        un_stable_rate_test = (un_stable / un_structures) if un_structures > 0 else 0
        un_metastable_rate_test = (un_metastable / un_structures) if un_structures > 0 else 0
        
        # Key: Estimate overall SUN and MSUN rates
        estimated_overall_sun_rate = un_ratio_dataset * un_stable_rate_test
        estimated_overall_msun_rate = un_ratio_dataset * un_metastable_rate_test
        
        statistics = {
            'processing_mode': 'un_only' if self.un_only else 'un_and_non_un',
            'dataset_info': {
                'total_structures_dataset': total_structures_dataset,
                'un_structures_dataset': un_structures_dataset,
                'non_un_structures_dataset': total_structures_dataset - un_structures_dataset,
                'un_ratio_dataset': un_ratio_dataset
            },
            'sampling_info': {
                'total_structures_tested': total_structures,
                'successful_relaxations': successful_structures,
                'failed_relaxations': total_structures - successful_structures,
                'un_structures_tested': un_structures,
                'non_un_structures_tested': non_un_structures
            },
            'stability_results': {
                'total_stable': stable_count,
                'total_metastable': metastable_count,
                'total_selected_for_dft': dft_count,
                'un_stable': un_stable,
                'un_metastable': un_metastable,
                'un_selected_for_dft': un_dft,
                'non_un_stable': non_un_stable,
                'non_un_metastable': non_un_metastable,
                'non_un_selected_for_dft': non_un_dft
            },
            'test_set_rates': {
                'un_stable_rate_test': un_stable_rate_test,
                'un_metastable_rate_test': un_metastable_rate_test,
                'note': 'UN structure stability rates based on test set'
            },
            'estimated_overall_metrics': {
                'estimated_overall_sun_rate': estimated_overall_sun_rate,
                'estimated_overall_msun_rate': estimated_overall_msun_rate,
                'calculation_method': 'UN ratio × test set UN stability rate',
                'note': 'Preliminary estimate based on CHGNet results, final results require DFT correction'
            },
            'thresholds': {
                'stability_threshold': self.stability_threshold,
                'metastability_threshold': self.metastability_threshold,
                'dft_threshold': self.dft_threshold,
                'max_relaxation_steps': self.max_steps
            },
            'user_guidance': {
                'next_step': 'Run DFT calculations, then use dft_post_processor.py for final evaluation',
                'dft_un_folder': str(self.output_dir / "dft_un"),
                'dft_non_un_folder': str(self.output_dir / "dft_non_un") if not self.un_only else "Not generated (UN_only mode)",
                'final_command': f'python dft_post_processor.py --chgnet_csv {self.output_dir}/chgnet_results.csv --dft_results_folder ./dft_results --original_csv_file {self.csv_file} --mp_hull_path {self.mp_hull_path} --output_dir ./final_results' + (' --un_only' if self.un_only else ''),
                'note': 'Second step will automatically calculate UN ratio from original CSV, no manual input needed',
                'vasp_config': 'VASP pseudopotentials configured via pymatgen, POTCAR files correctly generated'
            }
        }
        
        # Save JSON
        json_file = self.output_dir / "chgnet_statistics.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"   JSON statistics: {json_file}")
        
        # Print summary
        print(f"\nProcessing results summary:")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        if self.un_only:
            print(f"   Total structures: {total_structures} (UN: {un_structures})")
        else:
            print(f"   Total structures: {total_structures} (UN: {un_structures}, non-UN: {non_un_structures})")
        print(f"   Successful relaxations: {successful_structures}")
        if self.un_only:
            print(f"   Stable structures: {stable_count} (UN: {un_stable})")
            print(f"   Metastable structures: {metastable_count} (UN: {un_metastable})")
            print(f"   DFT candidates: {dft_count} (UN: {un_dft})")
        else:
            print(f"   Stable structures: {stable_count} (UN: {un_stable}, non-UN: {non_un_stable})")
            print(f"   Metastable structures: {metastable_count} (UN: {un_metastable}, non-UN: {non_un_metastable})")
            print(f"   DFT candidates: {dft_count} (UN: {un_dft}, non-UN: {non_un_dft})")
        
        print(f"\nOverall SUN/MSUN estimation (based on CHGNet results):")
        print(f"   Sample test results:")
        print(f"      - UN structures tested: {un_structures}")
        print(f"      - Stable UN: {un_stable} (stability rate: {un_stable_rate_test:.1%})")
        print(f"      - Metastable UN: {un_metastable} (metastability rate: {un_metastable_rate_test:.1%})")
        print(f"   Large dataset statistics:")
        print(f"      - Total structures: {total_structures_dataset:,}")
        print(f"      - UN structures: {un_structures_dataset:,}")
        print(f"      - UN ratio: {un_ratio_dataset:.1%}")
        print(f"   Estimated overall metrics:")
        print(f"      - Overall SUN rate: {un_ratio_dataset:.1%} × {un_stable_rate_test:.1%} = {estimated_overall_sun_rate:.2%}")
        print(f"      - Overall MSUN rate: {un_ratio_dataset:.1%} × {un_metastable_rate_test:.1%} = {estimated_overall_msun_rate:.2%}")
        print(f"   Interpretation: Expected ~{estimated_overall_sun_rate*100:.0f} SUN and ~{estimated_overall_msun_rate*100:.0f} MSUN per 100 generated structures")
        
        print(f"\nNext steps:")
        print(f"   1. Run DFT calculations (using generated input files)")
        print(f"      - UN structures: {self.output_dir}/dft_un/")
        if not self.un_only:
            print(f"      - non-UN structures: {self.output_dir}/dft_non_un/")
        print(f"   2. Collect DFT results (.traj files)")
        print(f"   3. Run final evaluation:")
        print(f"      python dft_post_processor.py \\")
        print(f"        --chgnet_csv {self.output_dir}/chgnet_results.csv \\")
        print(f"        --dft_results_folder ./dft_results \\")
        print(f"        --original_csv_file {self.csv_file} \\")
        print(f"        --mp_hull_path {self.mp_hull_path} \\")
        print(f"        --output_dir ./final_results \\")
        if self.un_only:
            print(f"        --un_only")
        print(f"   POTCAR files automatically generated via pymatgen configuration")
    
    def run_complete_evaluation(self):
        """Run complete evaluation workflow"""
        print("Starting complete CHGNet evaluation workflow")
        print("="*60)
        
        start_time = time.time()
        
        # 1. Load and sample structures (mixed for load balancing)
        all_structures = self.load_and_sample_structures()
        
        if not all_structures:
            print("ERROR: No processable structures, exiting")
            return
        
        # 2. Multi-GPU parallel CHGNet relaxation (load balanced)
        results = self.run_multi_gpu_relaxation(all_structures)
        
        # 3. Calculate stability and filter DFT candidates
        dft_candidates = self.calculate_stability_and_filter(results)
        
        # 4. Generate DFT input files
        self.generate_dft_inputs(dft_candidates)
        
        # 5. Save results
        self.save_results(results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("="*60)
        print(f"CHGNet evaluation workflow completed!")
        print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"   Processing mode: {'UN structures only' if self.un_only else 'UN + non-UN structures'}")
        print(f"   Results saved in: {self.output_dir}")
        print("="*60)

def main():
    # Critical: Set multiprocessing start method to spawn (CUDA requirement)
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="CHGNet Processor - First Stage Evaluation")
    
    # Required parameters
    parser.add_argument("--cif_folder", type=str, required=True,
                       help="Path to folder containing CIF files")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to CSV file containing UN results")
    parser.add_argument("--mp_hull_path", type=str, required=True,
                       help="Path to MP hull data file (.pkl)")
    parser.add_argument("--sample_size", type=int, required=True,
                       help="Number of structures to sample from UN (UN_only mode samples UN only, otherwise samples this number from both UN and non-UN)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory path")
    
    # Optional parameters
    parser.add_argument("--un_only", action="store_true",
                       help="Process UN structures only (default: process both UN and non-UN)")
    parser.add_argument("--max_steps", type=int, default=1500,
                       help="Maximum CHGNet relaxation steps (default: 1500)")
    parser.add_argument("--stability_threshold", type=float, default=0.0,
                       help="Stability threshold eV/atom (default: 0.0)")
    parser.add_argument("--metastability_threshold", type=float, default=0.1,
                       help="Metastability threshold eV/atom (default: 0.1)")
    parser.add_argument("--dft_threshold", type=float, default=0.1,
                       help="DFT selection threshold eV/atom (default: 0.1)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create processor and run
    try:
        processor = CHGNetProcessor(
            cif_folder=args.cif_folder,
            csv_file=args.csv_file,
            mp_hull_path=args.mp_hull_path,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            un_only=args.un_only,
            max_steps=args.max_steps,
            stability_threshold=args.stability_threshold,
            metastability_threshold=args.metastability_threshold,
            dft_threshold=args.dft_threshold
        )
        
        processor.run_complete_evaluation()
        
    except Exception as e:
        print(f"ERROR: Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
