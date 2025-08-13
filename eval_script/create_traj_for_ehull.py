#!/usr/bin/env python3
"""
Create ASE trajectory files suitable for ehull_correction.py from VASP output files
Simplified version: Use the same name as DFT directory (no method prefix added)

python create_traj_for_ehull.py --input-dir ./stable_results --output-dir ./clean_outputs
"""

import os
import sys
import glob
import re
import numpy as np
from pathlib import Path
import argparse
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

def read_vasp_energy(folder_path):
    """
    Read energy information from VASP output files
    Priority order: OSZICAR > OUTCAR > vasprun.xml
    """
    # Try reading from OSZICAR
    oszicar_path = os.path.join(folder_path, 'OSZICAR')
    if os.path.exists(oszicar_path):
        try:
            with open(oszicar_path, 'r') as f:
                lines = f.readlines()
                # Search backwards from the last line
                for line in reversed(lines):
                    if 'F=' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'F=':
                                return float(parts[i+1])
        except Exception as e:
            print(f"Warning: Cannot read energy from OSZICAR: {e}")
    
    # Try reading from OUTCAR
    outcar_path = os.path.join(folder_path, 'OUTCAR')
    if os.path.exists(outcar_path):
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
                # Try reading energy(sigma->0)
                energy_match = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.]+)', content)
                if energy_match:
                    return float(energy_match.group(1))
                # Try reading TOTEN
                toten_match = re.findall(r'TOTEN\s*=\s*([-\d.]+)', content)
                if toten_match:
                    return float(toten_match[-1])  # Take the last one
        except Exception as e:
            print(f"Warning: Cannot read energy from OUTCAR: {e}")
    
    # Energy information not found
    print(f"Warning: Cannot find energy information in {folder_path}")
    return None

def read_vasp_forces(folder_path):
    """Read force information from OUTCAR file"""
    outcar_path = os.path.join(folder_path, 'OUTCAR')
    if not os.path.exists(outcar_path):
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
            forces = []
            read_forces = False
            
            for i, line in enumerate(lines):
                if 'POSITION' in line and 'TOTAL-FORCE' in line:
                    read_forces = True
                    continue
                
                if read_forces:
                    if '------' in line:
                        continue
                    if line.strip() == '':
                        read_forces = False
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                            forces.append([fx, fy, fz])
                        except (ValueError, IndexError):
                            read_forces = False
            
            if forces:
                return np.array(forces)
        
    except Exception as e:
        print(f"Warning: Cannot read force information from OUTCAR: {e}")
    
    return None

def read_vasp_stress(folder_path):
    """Read stress information from OUTCAR file"""
    outcar_path = os.path.join(folder_path, 'OUTCAR')
    if not os.path.exists(outcar_path):
        return None
    
    try:
        with open(outcar_path, 'r') as f:
            content = f.read()
            
            # Find stress tensor information
            stress_matches = re.findall(r'in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', content)
            if stress_matches:
                # Take the last match
                stress_match = stress_matches[-1]
                # Convert to ASE format (xx, yy, zz, yz, xz, xy)
                stress = [float(stress_match[i]) for i in range(6)]
                # Convert units: kB -> eV/Å³ (factor is 0.0006241509)
                stress = [-0.0006241509 * s for s in stress]
                return stress
    except Exception as e:
        print(f"Warning: Cannot read stress information from OUTCAR: {e}")
    
    return None

def create_traj_file(folder_path, output_file):
    """
    Create trajectory file from VASP output
    
    Parameters:
    - folder_path: VASP calculation folder
    - output_file: Output trajectory file path
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Read initial structure
        poscar_path = os.path.join(folder_path, 'POSCAR')
        if not os.path.exists(poscar_path):
            print(f"Error: POSCAR file not found in {folder_path}")
            return False
        
        initial_atoms = read(poscar_path)
        
        # Set initial energy (use 0 as initial energy)
        initial_energy = 0.0
        initial_calc = SinglePointCalculator(initial_atoms, energy=initial_energy)
        initial_atoms.calc = initial_calc
        
        # Read final structure
        contcar_path = os.path.join(folder_path, 'CONTCAR')
        if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
            final_atoms = read(contcar_path)
        else:
            # If CONTCAR doesn't exist or is empty, try reading from OUTCAR
            outcar_path = os.path.join(folder_path, 'OUTCAR')
            if os.path.exists(outcar_path):
                try:
                    final_atoms = read(outcar_path, index=-1)  # Read last frame
                except Exception:
                    print(f"Error: Cannot read final structure from {outcar_path}")
                    return False
            else:
                print(f"Error: Cannot find valid final structure in {folder_path}")
                return False
        
        # Read final energy
        final_energy = read_vasp_energy(folder_path)
        if final_energy is None:
            print(f"Error: Cannot read energy from {folder_path}")
            return False
        
        # Read forces and stress information
        forces = read_vasp_forces(folder_path)
        stress = read_vasp_stress(folder_path)
        
        # Create calculator and attach to final structure
        final_calc = SinglePointCalculator(
            final_atoms, 
            energy=final_energy,
            forces=forces,
            stress=stress
        )
        final_atoms.calc = final_calc
        
        # Write trajectory file
        with Trajectory(output_file, 'w') as traj:
            traj.write(initial_atoms)
            traj.write(final_atoms)
        
        print(f"Success: Created trajectory file {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: Failed to create trajectory file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create ASE trajectory files from VASP output files')
    parser.add_argument('--input-dir', required=True, help='VASP calculation results directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for trajectory files')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of VASP calculation directories
    vasp_folders = []
    for item in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, item)
        if os.path.isdir(folder_path) and re.match(r'^\d+$', item):
            vasp_folders.append(folder_path)
    
    if not vasp_folders:
        print(f"Error: No VASP calculation directories found in {args.input_dir}")
        return
    
    print(f"Found {len(vasp_folders)} VASP calculation directories")
    
    # Create trajectory files
    success_count = 0
    for folder in tqdm(sorted(vasp_folders), desc="Processing VASP outputs"):
        # Use the same name as DFT directory (no method prefix added)
        folder_id = os.path.basename(folder)
        output_file = os.path.join(args.output_dir, f"{folder_id}.traj")
        
        if create_traj_file(folder, output_file):
            success_count += 1
    
    print(f"Completed: Successfully created {success_count}/{len(vasp_folders)} trajectory files")
    
    # Verify one trajectory file (if exists)
    if success_count > 0:
        first_folder_id = os.path.basename(vasp_folders[0])
        first_traj = os.path.join(args.output_dir, f"{first_folder_id}.traj")
        if os.path.exists(first_traj):
            print("\nVerifying first trajectory file:", first_traj)
            try:
                from ase.io.trajectory import Trajectory
                traj = Trajectory(first_traj)
                print(f"Number of frames: {len(traj)}")
                print(f"Initial energy: {traj[0].get_potential_energy()} eV")
                print(f"Final energy: {traj[-1].get_potential_energy()} eV")
                print(f"Number of atoms: {traj[-1].get_global_number_of_atoms()}")
                if traj[-1].calc.results.get('forces') is not None:
                    print(f"Force information: Available")
                else:
                    print(f"Force information: Not available")
                print("Verification successful: Trajectory file contains required information")
                
                # Show command for running ehull_correction.py
                print("\nNext, run the following command:")
                print("python scripts_analysis/ehull_correction.py \"eval_for_dft.json\" \"ehulls_corrected.json\" --root_dft_clean_outputs \".\" --method \"diffcsp_mp20\"")
            except Exception as e:
                print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
