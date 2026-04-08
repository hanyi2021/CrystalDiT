# CrystalDiT: A Diffusion Transformer for Crystal Generation


[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.16614)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/hanyi2021/CrystalDiT)

> **🎉 This work has been accepted by AAAI 2026!**

## Overview

CrystalDiT is a diffusion transformer for crystal structure generation that achieves state-of-the-art performance by challenging the trend of architectural complexity. Instead of intricate, multi-stream designs, CrystalDiT employs a unified transformer that treats lattice and atomic properties as a single, interdependent system.

**Key Features:**
- **Simplified Architecture**: Unified attention mechanism for joint lattice-atom processing
- **Chemical Representation**: Two-dimensional atomic encoding using periodic table positions
- **Balanced Evaluation**: Novel checkpoint selection optimizing quality-discovery trade-off
- **State-of-the-art Performance**: 8.78±0.74% SUN rate on MP-20, substantially outperforming recent methods

## Main Results

### Performance on MP-20 Dataset

| Method | Struct. Valid (%) | Chem. Valid (%) | UN Rate (%) | SUN (%) | MSUN (%) |
|--------|-------------------|-----------------|-------------|---------|----------|
| DiffCSP | 99.90 | 82.52 | 87.17 | 3.49 | 20.75 |
| FlowMM | 99.22 | 82.09 | 87.66 | 4.21±0.18 | 20.77±0.09 |
| DiffCSP++ | 99.96 | 84.74 | 87.62 | 3.33 | 19.10 |
| MatterGen | 99.99 | 83.62 | 89.89 | 3.66±0.28 | 24.18±0.63 |
| ADiT | 99.58 | 90.83 | 37.08 | 2.74 | 13.50 |
| **CrystalDiT (Simple)** | **97.79** | **87.02** | **63.28** | **8.78±0.74** | **25.90±0.95** |

> **Note**: Results with ±std are based on 3 independent bootstrap samples of 500 UN structures for DFT evaluation. Other baseline results are from single runs.

### Scaling to Larger Structures (MPTS-52)

CrystalDiT demonstrates strong generalization to larger crystal structures:

| Dataset | Max Atoms | UN Rate (%) | SUN (%) | MSUN (%) | Degradation |
|---------|-----------|-------------|---------|----------|-------------|
| MP-20 | 20 | 63.28 | 8.78±0.74 | 25.90±0.95 | - |
| MPTS-52 | 52 | 61.45 | 6.73 | 20.19 | -2.05% SUN |

Despite a 2.6× increase in maximum structure size, the performance degradation is only ~2%, demonstrating excellent scalability.

## Architecture

### Unified Diffusion Transformer

- **Model Size**: 330MB (18 layers, d=512, 8 attention heads)
- **Input Representation**: 23-token sequence (3 lattice vectors + 20 atoms)
- **Processing**: Unified self-attention treating all crystal components as interdependent
- **Training**: 50,000 epochs, batch size 256, learning rate 1e-4

### Two-Dimensional Atomic Representation

Instead of atomic numbers, we encode atoms using periodic table positions:
- **Period (row)**: r ∈ [0, 7], normalized to [-1, 1]
- **Group (column)**: c ∈ [0, 18], normalized to [-1, 1]
- Naturally captures chemical similarity through spatial proximity

## Installation
```bash
# Clone the repository
git clone https://github.com/hanyi2021/CrystalDiT.git
cd CrystalDiT

# Create conda environment
conda create -n crystaldit python=3.9
conda activate crystaldit

# Install dependencies
pip install torch torchvision torchaudio
pip install pymatgen
pip install smact
pip install chgnet
pip install -r requirements.txt
```

## Quick Start

### Generate Crystal Structures
```python
from crystaldit import CrystalDiT

# Load pretrained model
model = CrystalDiT.from_pretrained('checkpoints/crystaldit_simple.pt')

# Generate 100 structures
structures = model.generate(n_samples=100)

# Save to CIF files
for i, structure in enumerate(structures):
    structure.to(filename=f'generated_structure_{i}.cif')
```

### Training
```bash
# Train on MP-20 dataset
python train.py \
    --dataset mp20 \
    --model_type simple \
    --num_layers 18 \
    --hidden_dim 512 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --num_epochs 50000
```

### Evaluation
```bash
# Generate and evaluate structures
python evaluate.py \
    --checkpoint checkpoints/crystaldit_simple.pt \
    --n_samples 10000 \
    --output_dir results/

# Run DFT stability assessment
python evaluate_stability.py \
    --structures_dir results/ \
    --n_samples 500
```

## Dataset

We use the MP-20 dataset from the Materials Project:
- **Training set**: 27,136 structures
- **Test set**: 18,095 structures  
- **Max atoms**: 20
- **Elements**: 89 types

Download and preprocess:
```bash
python scripts/prepare_mp20.py --output_dir data/mp20/
```

## Ablation Studies

### Architecture Depth

| Depth | UN Rate (%) | SUN (%) | MSUN (%) |
|-------|-------------|---------|----------|
| 6 layers | 82.4 | 5.78 | 23.48 |
| 12 layers | 73.2 | 6.95 | 24.89 |
| **18 layers** | **63.3** | **8.78** | **25.90** |
| 24 layers | 56.8 | 7.10 | 26.41 |

### Atomic Representation

| Representation | UN Rate (%) | SUN (%) | MSUN (%) |
|----------------|-------------|---------|----------|
| 1D (atomic number) | 78.47 | 6.28 | 24.33 |
| **2D (period, group)** | **63.28** | **8.78** | **25.90** |

### Normalization

| Normalization | UN Rate (%) | SUN (%) | MSUN (%) |
|---------------|-------------|---------|----------|
| Without | 49.4 | 4.70 | 18.04 |
| **With** | **63.28** | **8.78** | **25.90** |

## Model Checkpoints

Download pretrained models from [Hugging Face](https://huggingface.co/xiaohan-yi/CrystalDiT):

- `crystaldit_simple.pt` - Main model (330MB)

## Project Structure
```
CrystalDiT/
├── crystaldit/
│   ├── models/
│   │   ├── diffusion_transformer.py
│   │   ├── atomic_encoder.py
│   │   └── lattice_encoder.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── balance_score.py
│   └── evaluation/
│       ├── metrics.py
│       └── stability.py
├── scripts/
│   ├── prepare_mp20.py
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── default.yaml
├── checkpoints/
├── requirements.txt
└── README.md
```

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{yi2026crystaldit,
  title={CrystalDiT: A Diffusion Transformer for Crystal Generation},
  author={Yi, Xiaohan and Xu, Guikun and Zhang, Zhong and Liu, Liu and Bian, Yatao and Xiao, Xi and Zhao, Peilin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  year={2026}
}
```

## Acknowledgments

This work was supported by:
- Natural Science Foundation of Guangdong Province (grant no. 2025A1515011946)
- National University of Singapore School of Computing (grant no. A-0010308-00-00)

Part of this work was conducted when authors Xiaohan Yi and Guikun Xu were at Tencent AI Lab. We acknowledge computational resources from Tencent and thank Tao Chen for insightful discussions.

## Contact

For questions and feedback:
- **Xiaohan Yi**: yxh24@mails.tsinghua.edu.cn
- **Xi Xiao**: xiaox@sz.tsinghua.edu.cn
- **Peilin Zhao**: peilinzhao@sjtu.edu.cn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Work

- [DiffCSP](https://github.com/jiaor17/DiffCSP) - Diffusion approach for crystal structure prediction
- [FlowMM](https://github.com/facebookresearch/flowmm) - Riemannian flow matching for materials
- [MatterGen](https://github.com/microsoft/mattergen) - Joint diffusion with equivariant networks
- [CDVAE](https://github.com/txie-93/cdvae) - Crystal diffusion variational autoencoder

---

⭐ **Star this repo if you find it useful!**
