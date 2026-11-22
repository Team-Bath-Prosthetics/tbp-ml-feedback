# Configs

W&B sweep configs. Each YAML file defines one hyperparameter sweep.

| File | Sweeps |
|---|---|
| `mlp_ae_sweep.yaml` | `MLPAutoencoder` via `src/tactile/training/mlp_ae_train.py` |

## Logging in to W&B

Run `wandb login` once and paste your API key from https://wandb.ai/authorize. It's saved to `~/.netrc`, so you won't need it again on that machine.

Keep the key secret: never commit it, paste it into code or configs, or share it in chat. `.env` and `.netrc` are git-ignored. If a key leaks, revoke it on the same page and make a new one.

Runs should go to the team's W&B workspace (`<entity>` below). If you don't have access, contact Toby (tdv28@bath.ac.uk).

## Running a sweep

From the repo root (`data_dir` is relative to where the agent runs):

```bash
wandb sweep configs/mlp_ae_sweep.yaml         # creates the sweep, prints the command to run the agent
wandb agent --count 20 <entity>/tbp-ml-feedback/<sweep-id> # Note: you can copy this ^^ and add the --count 20 flag instead
```

Run `wandb agent` on more machines (or terminals) with the same ID to run trials in parallel. Results show up under the project's **Sweeps** tab.

## Editing or adding a sweep

- Parameter types: `values: [...]` (pick from a list), `value: x` (fixed), or `distribution` + `min`/`max` (e.g. `log_uniform_values` for learning rates).
- `method`: `bayes` (default here), `random`, or `grid` (grid only works with `values`/`value`, no distributions).
- `metric.name` must be something the script logs (`val_loss`, `val_adc_mae`, `train_loss`).
- For a new model, copy `mlp_ae_sweep.yaml`, point `program` at its training script, and make sure that script reads its hyperparameters from `run.config`.

Full syntax: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
