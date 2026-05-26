# Notebooks

Analysis scripts in **Jupytext "percent" format** (`# %% [markdown]` for prose,
`# %%` for code). Each file:

- runs end-to-end as a plain Python script — `python notebooks/01_verification.py`
  produces plots + a Markdown report on stdout, no Jupyter required.
- opens as a Jupyter notebook directly in VS Code or JupyterLab (with the
  `jupytext` extension installed).
- is diff-friendly in git — no embedded JSON, no execution-count noise.

| File | Purpose |
|---|---|
| `_lib.py`               | Shared loaders: KNMI HDF5 parsers + synthetic data generator. |
| `01_verification.py`    | Pair forecast → observation, compute MAE/RMSE/POD/FAR/CSI/HSS by lead-time, plot skill curves. Uses real KNMI data if available, falls back to a deterministic synthetic dataset. |

## Running

```bash
# from research/
source .venv/bin/activate
python notebooks/01_verification.py                       # synthetic fallback
python notebooks/01_verification.py --data ../data/knmi   # real KNMI files
```

Output is written to `research/output/` (plots, summary `.md`).

## Why `.py` instead of `.ipynb`

- `git diff` actually works.
- Reviewers can read it in any editor.
- Still opens as a notebook in IDEs that speak Jupytext (VS Code: built-in).
- Can be imported as a module from other scripts when needed.
