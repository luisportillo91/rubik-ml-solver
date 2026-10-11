# Rubik's Cube ML Solver

An ultra-fast, stateless representation learning framework and machine learning solver for the Rubik's Cube.

---

## Highlights

- **Stateless Permutation Engine**: Fast Python permutation engine using `operator.itemgetter(*perm)` and immutable 54-element tuples (~1.7M–2.9M moves/sec).
- **Transformer Cube Encoder**: Pre-LN spatial Multi-Head Attention transformer with discrete color embeddings and learnable 54-coordinate positional encodings.
- **InfoNCE Metric Learning**: Contrastive representation learning with:
  - **Positive pair**: Exactly 1 move ahead ($\text{dist} = 1$).
  - **Excluded buffer zone**: $2 \le \text{dist} \le 5$ strictly omitted ($k=5$).
  - **Negatives**: Random scrambles strictly with $\text{dist} > 5$.
- **Vectorized $M \times N$ Batch Generator**: Group-theoretic Cayley graph translation invariance enables generating **millions of verified negatives per second** via native PyTorch tensor indexing (no hashmaps, no per-anchor BFS).
- **Interactive Notebook Trainer**: Run training directly inside Python / Jupyter / Kaggle notebook cells with an **enriched progress bar (`tqdm`)** showing live Loss, Top-1/Top-5 accuracy, Positive/Negative similarity, and Margin.
- **Multi-GPU Acceleration**: Automatic `DataParallel` across multiple GPUs in notebooks, or `torchrun` DDP for distributed clusters.

---

## Installation

### From GitHub directly:
```bash
pip install git+https://github.com/luisportillo91/rubik-ml-solver.git
```

### Local Development:
Using [uv](https://github.com/astral-sh/uv) (recommended):
```bash
git clone https://github.com/luisportillo91/rubik-ml-solver.git
cd rubik-ml-solver
uv sync
```

Or standard `pip`:
```bash
pip install -e .
```

Run test suite:
```bash
pytest -v tests/
```

---

## Interactive Python & Notebook Usage

Train directly inside a Jupyter or Kaggle notebook with live metrics:

```python
import rubik

# Run training directly in a notebook cell
result = rubik.train_infonce(
    steps=2000,
    batch_size=32,          # M random anchors
    k_negatives=64,         # N canonical negatives (M x N = 2,048 negatives per step)
    distance_threshold=5,   # Excluded buffer zone [2, 5]
    d_model=128,
    nhead=4,
    num_layers=4,
    d_embed=128,
    lr=3e-4,
    device="auto",          # Automatically detects CUDA / 2x T4 GPUs via DataParallel
    save_dir="checkpoints",
    show_progress_bar=True,
)

# Plot training curves directly from the result
result.plot_curves(save_path="training_curves.png")
```

---

## Kaggle Notebook (2x NVIDIA T4 GPUs)

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Under **Notebook Settings** (right panel):
   - Set **Accelerator** to **GPU T4 x 2**.
   - Turn **Internet** to **On**.
3. Import the prebuilt notebook from GitHub:
   - Click **File** $\rightarrow$ **Import Notebook** $\rightarrow$ **GitHub**.
   - Paste: `https://github.com/luisportillo91/rubik-ml-solver/blob/master/notebooks/kaggle_train_2xt4.ipynb`
4. Click **Run All**. The training cell displays the live enriched progress bar and exports checkpoints to `/kaggle/working/checkpoints/`.
