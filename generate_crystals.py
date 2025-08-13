import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
import torch.multiprocessing as mp
from functools import partial
import time
import json
from scipy.stats import norm

from crystal_dit import CrystalDiT
from crystal_diffusion import CrystalGaussianDiffusion
from crystal_representation import ATOMIC_NUMBER_TO_POSITION, POSITION_TO_ATOMIC_NUMBER, normalize_row, normalize_column, denormalize_row, denormalize_column

# Rare gas elements list
RARE_GAS_ELEMENTS = [2, 10, 18, 36, 54, 86]

def map_to_element(row_value, column_value, sigma=0.01, filter_rare_gases=True, max_atomic_number=94):
    """Map predicted continuous row/column values to periodic table elements using DDPM discretization"""
    from scipy.stats import norm
    
    candidates = [(0, 0, 0)]  # Invalid atom as candidate
    
    for z, (elem_row, elem_column) in ATOMIC_NUMBER_TO_POSITION.items():
        if z == 0:
            continue
        if filter_rare_gases and z in RARE_GAS_ELEMENTS:
            continue
        if z > max_atomic_number:
            continue
        candidates.append((z, elem_row, elem_column))
    
    row_step = 1/7     # 7 periods
    column_step = 1/18  # 18 groups
    
    element_probs = []
    for z, elem_row, elem_column in candidates:
        norm_row = normalize_row(elem_row)
        norm_column = normalize_column(elem_column)
        
        row_upper = norm_row + row_step if norm_row < 1 else np.inf
        row_lower = norm_row - row_step if norm_row > -1 else -np.inf
        column_upper = norm_column + column_step if norm_column < 1 else np.inf
        column_lower = norm_column - column_step if norm_column > -1 else -np.inf
        
        row_prob = norm.cdf(row_upper, loc=row_value, scale=sigma) - norm.cdf(row_lower, loc=row_value, scale=sigma)
        column_prob = norm.cdf(column_upper, loc=column_value, scale=sigma) - norm.cdf(column_lower, loc=column_value, scale=sigma)
        
        joint_prob = row_prob * column_prob
        element_probs.append((z, joint_prob))
    
    best_z, best_prob = max(element_probs, key=lambda x: x[1])
    
    if best_prob < 1e-6:
        return 0
        
    return best_z

def load_model(checkpoint_path, max_atoms, device, hidden_size=512, depth=18, num_heads=8):
    """Load trained model"""
    model = CrystalDiT(
        max_atoms=max_atoms,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'module.' in list(checkpoint['model'].keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    else:
        state_dict = checkpoint['model']
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, checkpoint.get('args', None)
    
def convert_to_structure(lattice_vectors, atom_features, max_length=46.7425, filter_rare_gases=True):
    """Convert model output to crystal structures"""
    batch_size = lattice_vectors.shape[0]
    structures = []
    
    for i in range(batch_size):
        lattice_matrix = lattice_vectors[i].cpu().numpy() * max_length
        atoms = atom_features[i].cpu().numpy()
        
        atomic_numbers = []
        frac_coords = []
        
        for atom in atoms:
            row_value = atom[0]
            column_value = atom[1]
            coords = atom[2:5]
            
            atom_z = map_to_element(
                row_value, 
                column_value, 
                sigma=0.1, 
                filter_rare_gases=filter_rare_gases, 
                max_atomic_number=94
            )
            
            if atom_z > 0:
                atomic_numbers.append(atom_z)
                frac_coords.append(coords % 1.0)
        
        if not atomic_numbers:
            print(f"Warning: Sample {i} has no valid atoms, skipping")
            continue
            
        elements = [Element.from_Z(z) for z in atomic_numbers]
        
        try:
            structure = Structure(
                lattice=lattice_matrix,
                species=elements,
                coords=frac_coords,
                coords_are_cartesian=False
            )
            structures.append(structure)
        except Exception as e:
            print(f"Error creating structure {i}: {e}")
    
    return structures


def save_structures(structures, output_dir, prefix="gen", start_idx=0):
    """Save generated crystal structures"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for i, structure in enumerate(structures):
        cif_path = os.path.join(output_dir, f"{prefix}_{start_idx+i+1}.cif")
        structure.to(filename=cif_path)
        saved_count += 1
    
    return saved_count


def get_available_gpus():
    """Get all available GPU devices"""
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(gpu_count)]


def analyze_generated_structures(structures):
    """Analyze generated crystal structures"""
    if not structures:
        return None
        
    element_counter = {}
    for struct in structures:
        for site in struct:
            element = site.specie.symbol
            z = site.specie.Z
            if z not in element_counter:
                element_counter[z] = 0
            element_counter[z] += 1
    
    total_atoms = sum(element_counter.values())
    element_percentages = {z: count/total_atoms*100 for z, count in element_counter.items()}
    sorted_elements = sorted(element_percentages.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nGenerated structure analysis - Total atoms: {total_atoms}")
    print("Element\tAtomic#\tCount\tPercentage(%)")
    print("-" * 40)
    for z, percentage in sorted_elements[:20]:
        element_symbol = Element.from_Z(z).symbol
        count = element_counter[z]
        print(f"{element_symbol}\t{z}\t{count}\t{percentage:.2f}")
    
    rare_gases_present = [z for z in element_counter if z in RARE_GAS_ELEMENTS]
    if rare_gases_present:
        print("\nWarning: Generated structures contain rare gas elements:")
        for z in rare_gases_present:
            element_symbol = Element.from_Z(z).symbol
            print(f"{element_symbol}(Z={z}): {element_counter[z]} atoms, {element_percentages[z]:.2f}%")
    
    return {
        "total_atoms": total_atoms,
        "element_counts": element_counter,
        "element_percentages": element_percentages
    }


def generate_crystals_on_gpu(gpu_id, args, samples_per_gpu, start_idx):
    """Generate crystal structures on specific GPU"""
    device = torch.device(gpu_id)
    
    process_id = gpu_id.split(":")[-1]
    print(f"[GPU {process_id}] Starting generation of {samples_per_gpu} crystal structures on {gpu_id}...")
    
    model, _ = load_model(args.checkpoint, args.max_atoms, device, 
                         hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads)
    
    diffusion = CrystalGaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule="linear"
    )
    
    all_structures = []
    num_batches = (samples_per_gpu + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches), desc=f"GPU {process_id} Generation Progress"):
        current_batch_size = min(args.batch_size, samples_per_gpu - i * args.batch_size)
        if current_batch_size <= 0:
            break
            
        with torch.no_grad():
            lattice_vectors, atom_features = diffusion.p_sample_loop(
                model,
                batch_size=current_batch_size,
                max_atoms=args.max_atoms,
                device=device,
                progress=False
            )
        
        structures = convert_to_structure(
            lattice_vectors, 
            atom_features, 
            max_length=args.max_length,
            filter_rare_gases=args.filter_rare_gases
        )
        
        all_structures.extend(structures)
    
    print(f"[GPU {process_id}] Successfully generated {len(all_structures)} crystal structures")
    
    if process_id == '0':
        analyze_generated_structures(all_structures)
    
    save_count = save_structures(all_structures, args.output_dir, start_idx=start_idx)
    
    return save_count


def run_parallel_generation(args):
    """Run parallel generation on multiple GPUs"""
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No available GPUs detected, using CPU generation")
        device = torch.device("cpu")
        args.device = "cpu"
        return generate_crystals_single_device(args)
    
    print(f"Detected {len(available_gpus)} available GPUs: {available_gpus}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_samples = args.num_samples
    samples_per_gpu = total_samples // len(available_gpus)
    remainder = total_samples % len(available_gpus)
    
    mp.set_start_method('spawn', force=True)
    
    processes = []
    start_idx = 0
    
    for i, gpu_id in enumerate(available_gpus):
        if i == len(available_gpus) - 1:
            gpu_samples = samples_per_gpu + remainder
        else:
            gpu_samples = samples_per_gpu
        
        p = mp.Process(
            target=generate_crystals_on_gpu,
            args=(gpu_id, args, gpu_samples, start_idx)
        )
        
        processes.append(p)
        start_idx += gpu_samples
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
    
    print(f"All GPU work completed, generated structures saved to {args.output_dir}")


def generate_crystals_single_device(args):
    """Generate crystal structures on single device"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.checkpoint}...")
    model, _ = load_model(args.checkpoint, args.max_atoms, device,
                         hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads)
    
    diffusion = CrystalGaussianDiffusion(
        timesteps=args.diffusion_steps,
        beta_schedule="linear"
    )
    
    all_structures = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    print(f"Starting generation of {args.num_samples} crystal structures...")
    for i in tqdm(range(num_batches)):
        current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        with torch.no_grad():
            lattice_vectors, atom_features = diffusion.p_sample_loop(
                model,
                batch_size=current_batch_size,
                max_atoms=args.max_atoms,
                device=device,
                progress=True
            )
        
        structures = convert_to_structure(
            lattice_vectors, 
            atom_features, 
            max_length=args.max_length,
            filter_rare_gases=args.filter_rare_gases
        )
        
        all_structures.extend(structures)
    
    print(f"Successfully generated {len(all_structures)} crystal structures")
    
    analyze_generated_structures(all_structures)
    save_structures(all_structures, args.output_dir)
    
    return all_structures


def main():
    parser = argparse.ArgumentParser(description="Generate crystal structures")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--max_atoms", type=int, default=20, help="Maximum number of atoms")
    parser.add_argument("--max_length", type=float, default=46.7425, help="Maximum lattice vector length")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--depth", type=int, default=18, help="Number of DiT layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Diffusion parameters
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./generated_crystals", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_multi_gpu", action="store_true", help="Use multi-GPU parallel generation")
    parser.add_argument("--filter_rare_gases", action="store_true", default=True, help="Filter rare gas elements")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        args.use_multi_gpu = True
    
    if args.use_multi_gpu:
        run_parallel_generation(args)
    else:
        generate_crystals_single_device(args)


if __name__ == "__main__":
    main()
