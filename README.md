# CrystalDiT: A Diffusion Transformer for Crystal Generation

This repository contains the official implementation of **CrystalDiT**, a simplified diffusion transformer architecture for crystal structure generation that achieves state-of-the-art performance by treating lattice and atomic properties as a single, interdependent system.

## 📄 Paper

**CrystalDiT: A Diffusion Transformer for Crystal Generation**  
Xiaohan Yi, Guikun Xu, Xi Xiao, Zhong Zhang, Liu Liu, Yatao Bian, Peilin Zhao

## 🚀 Coming Soon

- **🤖 Pre-trained Models**: Model checkpoints will be released upon paper acceptance
- **💎 Generated Structures**: Complete set of generated CIF files for reproducibility


## 🌟 Key Features

- **Simplified Architecture**: Unified diffusion transformer with joint attention processing
- **Chemical Representation**: Two-dimensional atomic encoding using periodic table positions  
- **Balanced Evaluation**: Novel Balance Score for optimizing discovery potential vs. generation quality
- **State-of-the-art Results**: 9.62% SUN rate on MP-20, outperforming FlowMM (4.38%) and MatterGen (3.42%)

## 🏗️ Project Structure

```
.
├── baseline/                           # Baseline method implementations
├── crystal_diffusion.py               # Gaussian diffusion process for crystals
├── crystal_dit.py                     # CrystalDiT model architecture
├── crystal_representation.py          # Crystal data processing and representation
├── datasets/
│   └── mp_20/                         # MP-20 dataset (to be downloaded)
├── diffusion/                         # DiT diffusion utilities (from Meta DiT)
│   ├── __init__.py
│   ├── diffusion_utils.py
│   ├── gaussian_diffusion.py
│   ├── respace.py
│   ├── models.py
│   └── timestep_sampler.py
├── eval_script/                       # Evaluation and analysis scripts
│   ├── balance_score_calculator.py    # Balance Score calculation
│   ├── batch_eval_metrics.py         # Batch evaluation metrics
│   ├── batch_generatefor.sh          # Batch generation script
│   ├── chgnet_process.py             # CHGNet relaxation and DFT preparation
│   ├── create_traj_for_ehull.py      # VASP trajectory creation
│   └── dft_post_processor.py         # DFT post-processing and final metrics
├── generate_crystals.py              # Crystal structure generation
├── requirements.txt                  # Python dependencies
└── train_crystal_dit.py              # Training script
```

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- torch==2.5.1
- numpy==1.26.4
- pandas==2.2.3
- pymatgen==2024.8.9
- ase==3.22.1
- chgnet==0.3.1
- timm==1.0.15

## 📂 Data Preparation

Due to supplement material size limitations, the following data files need to be downloaded separately:

### 1. MP-20 Dataset
Download the MP-20 dataset and place it in `datasets/mp_20/`:
```
datasets/mp_20/
├── train.csv
├── val.csv
└── test.csv
```

**Download Source**: [CDVAE Repository](https://github.com/txie-93/cdvae) or [Materials Project](https://materialsproject.org/)

### 2. Materials Project Hull Data
Download the Materials Project convex hull data for stability evaluation:


**Download Source**: [Matbench Discovery](https://matbench-discovery.materialsproject.org/)

### 3. VASP Pseudopotentials
Configure pymatgen for VASP calculations:
```bash
# Download VASP pseudopotentials and configure pymatgen
cd /path/to/your/vasp_potentials
pmg config --add PMG_VASP_PSP_DIR $(pwd)
```

**Required**: POT_GGA_PAW_PBE pseudopotential directory

## 🚀 Complete Workflow

Our complete evaluation pipeline consists of the following stages:

### Stage 1: Model Training

Train CrystalDiT for 50,000 epochs with checkpoints saved every 250 epochs:

```bash
# Multi-GPU training (8x V100, ~4 days)
python train_crystal_dit.py \
    --data_dir ./datasets/mp_20 \
    --output_dir ./output/crystal_dit \
    --batch_size 256 \
    --epochs 50000 \
    --save_every 250 \
    --hidden_size 512 \
    --depth 18 \
    --num_heads 8
```

### Stage 2: Checkpoint Selection via Balance Score

Generate 1000 structures from each checkpoint and calculate Balance Score:

```bash
# Batch generation for all checkpoints
./eval_script/batch_generatefor.sh ./output/crystal_dit 1000 32

# Calculate balance scores and select best 3 models
python eval_script/balance_score_calculator.py \
    ./checkpoint_evaluation_results \
    --alpha 1.0 \
    --max_epoch 50000
```

### Stage 3: Large-Scale Generation

Generate 10,000 structures from the best selected models:

```bash
# Generate 10k structures
python generate_crystals.py \
    --checkpoint ./output/crystal_dit/best_model.pt \
    --num_samples 10000 \
    --batch_size 32 \
    --output_dir ./generated_crystals \
    --use_multi_gpu
```

### Stage 4: Comprehensive Evaluation

Evaluate generated structures using multiple metrics:

```bash
# Calculate validity, uniqueness, novelty metrics
python eval_script/batch_eval_metrics.py \
    ./generated_crystals \
    --csv_folder ./datasets/mp_20
```

### Stage 5: Stability Assessment

#### 5.1 CHGNet Pre-relaxation and DFT Preparation

```bash
# CHGNet relaxation and DFT input generation
python eval_script/chgnet_process.py \
    --cif_folder ./generated_crystals \
    --csv_file ./generated_crystals_results.csv \
    --mp_hull_path /your/mp_hull_path \
    --sample_size 500 \
    --output_dir ./chgnet_results
```

#### 5.2 VASP DFT Calculations

Run VASP calculations using the generated input files:
```bash
# Run VASP calculations (user-specific cluster setup)
# Input files are in: ./chgnet_results/dft_un/ and ./chgnet_results/dft_non_un/
```

#### 5.3 Create Trajectory Files for Post-processing
、
```bash
# Convert VASP outputs to ASE trajectory files
python eval_script/create_traj_for_ehull.py \
    --input-dir ./vasp_results \
    --output-dir ./clean_outputs
```

#### 5.4 Final SUN/MSUN Calculation

```bash
# Calculate final SUN and MSUN rates
python eval_script/dft_post_processor.py \
    --chgnet_csv ./chgnet_results/chgnet_results.csv \
    --dft_results_folder ./clean_outputs \
    --original_csv_file ./generated_crystals_results.csv \
    --mp_hull_path /your/mp_hull_path \
    --output_dir ./final_results
```

## 📊 Evaluation Metrics

Our evaluation framework includes:

- **Validity Metrics**: Structural and compositional validity
- **Distribution Metrics**: Density and element distribution distances
- **Discovery Metrics**: Uniqueness, Novelty, UN Rate
- **Stability Metrics**: SUN Rate (Stable, Unique, Novel), MSUN Rate (Metastable, Unique, Novel)

## 🔧 Key Components

### CrystalDiT Architecture

The core model (`crystal_dit.py`) implements:
- Unified attention mechanism processing both lattice and atomic features
- Two-dimensional periodic table-based atomic representation
- Time-conditional adaptive layer normalization (AdaLN)

### Balance Score

Our novel model selection metric (`balance_score_calculator.py`):
```
Balance Score = UN Rate × (Quality Composite)^α
```

Where Quality Composite combines structural validity, compositional validity, density distribution, and element distribution scores.

## 📈 Results

| Method | SUN (%) | MSUN (%) | UN Rate (%) |
|---------|---------|----------|-------------|
| FlowMM | 4.38 | 20.16 | 87.66 |
| MatterGen | 3.42 | 23.91 | 89.89 |
| ADiT | 2.74 | 13.50 | 37.08 |
| **CrystalDiT** | **9.62** | **25.94** | **63.28** |

## 🎯 Baseline Comparison

Due to supplement size limitations, baseline method implementations and generated structures can be provided upon request. The `baseline/` directory contains implementations for:
- DiffCSP
- FlowMM  
- DiffCSP++
- MatterGen
- ADiT

## 🔄 Ablation Studies

Ablation study results (including 1D atomic representation variant) are available in the evaluation results. Generated CIF files for ablation studies can be provided upon request due to size constraints.

## 📜 License and Attribution

### CrystalDiT Code
This project is licensed under the MIT License.

### DiT Components
The `diffusion/` directory contains code adapted from Meta's DiT (Diffusion Transformers) project:
- **Original Repository**: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- **License**: MIT License
- **Citation**: Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." ICCV 2023.

We acknowledge and thank the DiT authors for making their code available under the MIT License.
