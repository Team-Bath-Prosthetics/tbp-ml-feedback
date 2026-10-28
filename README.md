# tbp-ml-feedback

Tools for turning prosthetic-hand tactile sensor data into vibration feedback:

- **Dataset generator**: produces synthetic tactile recordings, since the real hardware isn't built yet.
- **Autoencoder benchmark**: trains and compares AE variants (denoising, variational, contractive) on those recordings, with sweeps logged to Weights & Biases.

Research and planning documents live on Teams. This repository is for code only.

## Setup

Requires Python 3.11+.

```bash
git clone <repo-url>
cd tbp-ml-feedback
python3 -m venv .venv               # Windows: py -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"          # If you're not doing ML, `pip install -e ".[dev]"` skips the large PyTorch download.
wandb login                         # ONLY if you're using wandb for training; never commit your API key
```

Check your Python version with `python3 --version` (Windows: `py --version`). It must be 3.11 or newer. On a Mac, get a newer one from python.org or with `brew install python@3.11`.

If you don't have an NVIDIA GPU, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the `pip install -e` step. That gets the much smaller CPU-only PyTorch.

Check everything works:

```bash
pytest
```

## Adding a dependency

1. Add the package to `pyproject.toml`: under `dependencies` if everyone needs it, or under the `ml` / `dev` extras if only training or testing does. Use `>=` with the version you tested on (`pip show <package>`).
2. Reinstall: `pip install -e ".[dev,ml]"`.
3. Commit `pyproject.toml` in the same pull request as the code that uses it.

Note: if a dependency isn't in `pyproject.toml`, it'll break for everyone else and in CI.

## Layout

```
src/tactile/
  data_format.py   shared save/load format
  generator/       synthetic dataset generator
  viz/             visualising recordings
  noise.py         noise utilities
  models/          autoencoder variants
  training/        training/eval loops and W&B sweeps
scripts/           Run-only scripts
configs/           W&B sweep configs
tests/             pytest tests
data/              generated datasets (git-ignored)
```

## Workflow

1. Create a branch. 
2. Keep changes small. Run `pytest` before pushing.
3. Open a pull request into `main`. It needs passing CI and one approval, including from the owner of the folder you changed (see `.github/CODEOWNERS`).

You can use AI tools, but you must understand and be able to explain every line you submit.
