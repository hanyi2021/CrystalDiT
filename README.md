# CrystalDiT: A Diffusion Transformer for Crystal Generation

![Paper](https://img.shields.io/badge/Paper-arXiv-red)

![Code](https://img.shields.io/badge/Code-GitHub-green)

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

| Method                  | Struct. Valid (%) | Chem. Valid (%) | UN Rate (%) | SUN (%)       | MSUN (%)       |
| ----------------------- | ----------------- | --------------- | ----------- | ------------- | -------------- |
| DiffCSP                 | 99.90             | 82.52           | 87.17       | 3.49          | 20.75          |
| FlowMM                  | 99.22             | 82.09           | 87.66       | 4.21±0.18     | 20.77±0.09     |
| DiffCSP++               | 99.96             | 84.74           | 87.62       | 3.33          | 19.10          |
| MatterGen               | 99.99             | 83.62           | 89.89       | 3.66±0.28     | 24.18±0.63     |
| ADiT                    | 99.58             | 90.83           | 37.08       | 2.74          | 13.50          |
| **CrystalDiT (Simple)** | **97.79**         | **87.02**       | **63.28**   | **8.78±0.74** | **25.90±0.95** |

> **Note**: Results with ±std are based on 3 independent bootstrap samples of 500 UN structures for DFT evaluation. Other baseline results are from single runs.

### Scaling to Larger Structures (MPTS-52)

CrystalDiT demonstrates strong generalization to larger crystal structures:

| Dataset | Max Atoms | UN Rate (%) | SUN (%)   | MSUN (%)   | Degradation |
| ------- | --------- | ----------- | --------- | ---------- | ----------- |
| MP-20   | 20        | 63.28       | 8.78±0.74 | 25.90±0.95 | -           |
| MPTS-52 | 52        | 61.45       | 6.73      | 20.19      | -2.05% SUN  |

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

Run the commands below from the repository root. The dependency versions in  
`requirements.txt` are the released environment specification.

```bash
git clone https://github.com/hanyi2021/CrystalDiT.git
cd CrystalDiT
conda create -n crystaldit python=3.10
conda activate crystaldit
python -m pip install -r requirements.txt
```

Training requires an NVIDIA GPU and a CUDA-compatible PyTorch installation.  
Generation also supports CPU execution, but is substantially slower.

## Quick Start

### Generate Crystal Structures

Download the published checkpoint (the actual filename is `best_model.pt`):

```bash
mkdir -p checkpoints
curl -fL https://huggingface.co/xiaohan-yi/CrystalDiT/resolve/main/best_model.pt \
    -o checkpoints/best_model.pt

CUDA_VISIBLE_DEVICES=0 python generate_crystals.py \
    --checkpoint checkpoints/best_model.pt \
    --max_atoms 20 \
    --hidden_size 512 \
    --depth 18 \
    --num_heads 8 \
    --num_samples 100 \
    --batch_size 32 \
    --output_dir results/mp20
```

The script saves CIF files as `results/mp20/gen_1.cif`, etc. Samples that cannot  
be converted to structures are skipped, so the saved count can be smaller than  
`--num_samples`. To use CPU, set `CUDA_VISIBLE_DEVICES=""` and add `--device cpu`.  
With multiple visible GPUs, generation automatically distributes samples across  
them. Model dimensions must match the checkpoint. There is no packaged  
`crystaldit` module or `CrystalDiT.from_pretrained()` API in this release.

### Training

The MP-20 CSV splits are already included in `datasets/mp_20/`.

```bash
CUDA_VISIBLE_DEVICES=0 python train_crystal_dit.py \
    --data_dir datasets/mp_20 \
    --max_atoms 20 \
    --hidden_size 512 \
    --depth 18 \
    --num_heads 8 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --epochs 50000 \
    --output_dir output/crystal_dit
```

The training script launches one process per visible GPU itself; run it with  
`python`. `--batch_size` is **per GPU**. Rank 0 automatically preprocesses  
`train.csv` and creates `datasets/mp_20/preprocessed/train_preprocessed_ma20.pkl`.  
The output directory contains `config.json`, `train.log`, `latest_checkpoint.pt`,  
`best_model.pt`, and periodic `checkpoint_epoch_<epoch>.pt` files. Resume with  
`--resume output/crystal_dit/latest_checkpoint.pt`. The training script selects  
`best_model.pt` by training loss; the paper's balance-score selection is a separate  
evaluation step using `eval_script/balance_score_calculator.py`.

### Structural and Chemical Evaluation

The evaluator expects a parent directory containing one or more subdirectories  
of CIF files. The generation example above creates this layout:

```text
results/
└── mp20/
    ├── gen_1.cif
    └── ...
```

```bash
python eval_script/batch_eval_metrics.py results --csv_folder datasets/mp_20
```

Outputs are written to `results_flowmm_aligned_analysis/` in the current working  
directory, including `mp20_results.json`, `mp20_results.csv`, and  
`summary_report.csv`. This step evaluates validity, uniqueness, novelty, and  
other structural/compositional metrics. DFT-based SUN/MSUN additionally require  
CHGNet relaxation, external VASP calculations, and MP hull reference data; see  
[the evaluation and data notes](docs/reproducibility.md).

## Dataset

The committed MP-20 files contain:

| Split      | File                       | Structures |
| ---------- | -------------------------- | ---------- |
| Training   | `datasets/mp_20/train.csv` | 27,136     |
| Validation | `datasets/mp_20/val.csv`   | 9,047      |
| Test       | `datasets/mp_20/test.csv`  | 9,046      |

The model consumes CIF strings from the `cif` column. Preprocessing is implemented  
by `preprocess_dataset()` and `CrystalDataset` in `crystal_representation.py`;  
no separate download or `scripts/prepare_mp20.py` is needed. See  
[preprocessing details and a standalone command](docs/reproducibility.md#mp-20-preprocessing).

### MPTS-52

We use the same MPTS-52 dataset as DiffCSP++. The upstream repository provides  
[all three CSV splits](https://github.com/jiaor17/DiffCSP-PP/tree/main/data/mpts_52).  
Download a fixed revision from the repository root:

```bash
mkdir -p datasets/mpts_52
for split in train val test; do
    curl -fL "https://raw.githubusercontent.com/jiaor17/DiffCSP-PP/e82e7ffa7cb2a383bde69b067a343ca137d73e47/data/mpts_52/${split}.csv" \
        -o "datasets/mpts_52/${split}.csv"
done
```

For training, use `--data_dir datasets/mpts_52 --max_atoms 52` with  
`train_crystal_dit.py`; the CSV preprocessing is the same as for MP-20. Generation  
requires a matching 52-atom checkpoint and `--max_atoms 52`. The published MP-20  
checkpoint should not be used as an MPTS-52 checkpoint.

### Materials Project Hull Reference

Download the precomputed Materials Project convex hull reference from  
[this link](https://figshare.com/files/48241624) (provided by Matbench Discovery).  
Decompress the `.pkl.gz` file and pass the resulting `.pkl` file to the evaluation  
scripts with `--mp_hull_path`. See the  
[download commands](docs/reproducibility.md#materials-project-hull-reference).

## Ablation Studies

### Architecture Depth

| Depth         | UN Rate (%) | SUN (%)  | MSUN (%)  |
| ------------- | ----------- | -------- | --------- |
| 6 layers      | 82.4        | 5.78     | 23.48     |
| 12 layers     | 73.2        | 6.95     | 24.89     |
| **18 layers** | **63.3**    | **8.78** | **25.90** |
| 24 layers     | 56.8        | 7.10     | 26.41     |

### Atomic Representation

| Representation         | UN Rate (%) | SUN (%)  | MSUN (%)  |
| ---------------------- | ----------- | -------- | --------- |
| 1D (atomic number)     | 78.47       | 6.28     | 24.33     |
| **2D (period, group)** | **63.28**   | **8.78** | **25.90** |

### Normalization

| Normalization | UN Rate (%) | SUN (%)  | MSUN (%)  |
| ------------- | ----------- | -------- | --------- |
| Without       | 49.4        | 4.70     | 18.04     |
| **With**      | **63.28**   | **8.78** | **25.90** |

## Model Checkpoints

The [Hugging Face repository](https://huggingface.co/xiaohan-yi/CrystalDiT/tree/main)  
contains `best_model.pt` and `generate_crystals.tar`. Use the checkpoint download  
and generation command above; `crystaldit_simple.pt` is not a published filename.

## Project Structure

```text
CrystalDiT/
├── crystal_dit.py                 # CrystalDiT model
├── crystal_diffusion.py           # Crystal diffusion wrapper
├── crystal_representation.py      # CIF preprocessing and dataset
├── train_crystal_dit.py           # Training entry point
├── generate_crystals.py           # Generation entry point
├── datasets/mp_20/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── diffusion/                    # Diffusion and transformer utilities
├── eval_script/
│   ├── batch_generatefor.sh
│   ├── batch_eval_metrics.py
│   ├── balance_score_calculator.py
│   ├── chgnet_process.py
│   ├── create_traj_for_ehull.py
│   └── dft_post_processor.py
├── docs/reproducibility.md
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

The model card declares the MIT license. A standalone `LICENSE` file is not
included in this source repository.

## Related Work

- [DiffCSP](https://github.com/jiaor17/DiffCSP) - Diffusion approach for crystal structure prediction
- [FlowMM](https://github.com/facebookresearch/flowmm) - Riemannian flow matching for materials
- [MatterGen](https://github.com/microsoft/mattergen) - Joint diffusion with equivariant networks
- [CDVAE](https://github.com/txie-93/cdvae) - Crystal diffusion variational autoencoder

---

⭐ **Star this repo if you find it useful!**
