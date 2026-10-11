# Rubik's Cube ML Solver

An ultra-fast, stateless representation learning framework and machine learning solver for the Rubik's Cube.

---

## Highlights

- **Stateless Permutation Engine**: Fast Python permutation engine using `operator.itemgetter(*perm)` and immutable 54-element tuples (~1.7M–2.9M moves/sec).
- **Bidirectional BFS Search**: Fast graph search and neighborhood verification for exact move distances up to $d=5$ in < 0.9 ms.
- **Transformer Cube Encoder**: Pre-LN spatial Multi-Head Attention transformer with discrete color embeddings and learnable 54-coordinate positional encodings.
- **InfoNCE Metric Learning**: Contrastive representation learning configured with:
  - **Positive pair**: Exactly 1 move ahead ($\text{dist} = 1$).
  - **Excluded buffer zone**: $2 \le \text{dist} \le 5$ strictly omitted ($k=5$).
  - **Negatives**: High-volume random scrambles with $\text{dist} > 5$ ($K=100$ negatives per anchor).
- **Zero Disk Files**: Pure infinite streaming generator synthesizing fresh random scrambles directly in RAM (pure step-based training, no epochs).
- **Multi-GPU Distributed Training**: Built-in support for PyTorch DDP via `torchrun`, Automatic Mixed Precision (AMP FP16), and multi-worker DataLoader prefetching.

---

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
git clone https://github.com/luisportillo91/rubik-ml-solver.git
cd rubik-ml-solver
uv sync
```

Or using standard `pip`:
```bash
pip install -e .
```

Run test suite:
```bash
pytest -v tests/
```

---

## Local Training

Run step-based InfoNCE contrastive training:
```bash
python scripts/train_infonce.py \
    --steps 1000 \
    --batch-size 16 \
    --k-negatives 100 \
    --distance-threshold 5 \
    --log-every 25 \
    --save-every 250
```

---

## Kaggle Multi-GPU Training (2x NVIDIA T4)

To train on Kaggle using dual T4 GPUs:

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. In the right panel under **Notebook Settings**:
   - Set **Accelerator** to **GPU T4 x 2**.
   - Turn **Internet** to **On**.
3. Import or upload `notebooks/kaggle_train_2xt4.ipynb` (`File -> Import Notebook` or paste the GitHub URL).
4. Or launch directly via `torchrun` in a notebook cell:

```bash
!torchrun --nproc_per_node=2 scripts/train_infonce.py \
    --steps 2000 \
    --batch-size 32 \
    --k-negatives 100 \
    --distance-threshold 5 \
    --d-model 128 \
    --nhead 4 \
    --num-layers 4 \
    --d-embed 128 \
    --lr 3e-4 \
    --num-workers 2 \
    --save-dir /kaggle/working/checkpoints \
    --history-file training_history.json
```

This runs:
- 2 distributed processes mapped to `cuda:0` and `cuda:1`.
- Global batch size = 64 ($32 \times 2$).
- 6,400 negatives verified & evaluated per step ($64 \times 100$).
- Fast FP16 Tensor Cores acceleration via AMP.
- Checkpoints and training plots saved to `/kaggle/working/` for direct download.
