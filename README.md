# pylearn
Python programming from zero to hero

## Using the notebooks

This repository contains several Jupyter notebooks under the `notebooks/` folder that demonstrate Python basics and advanced topics. Below are the observed prerequisites and step-by-step instructions to get started with the notebooks on a macOS (zsh) environment.

### Checklist (user requirements)
- Identify prerequisites and steps required to run notebooks locally.  
- Update `README.md` with those prerequisites and steps.  
- Do not include unverifiable information (no hallucinations).

### Prerequisites
- Python 3.8 or newer installed and available as `python3`.
- pip (Python package installer).
- A virtual environment tool (built-in `venv`) is recommended.
- Jupyter (Notebook or JupyterLab) to open and run `.ipynb` files.

Observed Python packages used directly in the notebooks:
- pandas (dataframe creation and CSV save/load) — used in `01_hands_on.ipynb`.
- matplotlib (plotting) — used in `01_hands_on.ipynb`.

Other notebook content relies only on Python standard library features (generators, decorators, context managers, OO concepts) and does not require additional third-party packages.

### Quick start (recommended)
1. Open a terminal (zsh) and change to the repository root:

```bash
cd /path/to/pylearn
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages (a minimal `requirements.txt` is included):

```bash
pip install -r requirements.txt
```

4. Start Jupyter Notebook or JupyterLab:

```bash
# start classic notebook
python -m notebook
# or start JupyterLab if you prefer
jupyter lab
```

5. In the Jupyter UI, open a notebook from the `notebooks/` folder (for example `01_hands_on.ipynb`, `python_basic.ipynb`, or `python_advance.ipynb`) and run the cells in order.

### Important notes and observed issues
- `01_hands_on.ipynb` creates a CSV at `uploaded_files/sample_products1.csv` when the data-creation cell is executed. A later plotting cell attempts to read `./uploaded_files/sample_products.csv`. To avoid a FileNotFoundError either:
  - Run the data-creation cell, then rename `uploaded_files/sample_products1.csv` to `uploaded_files/sample_products.csv`, or
  - Edit the plotting cell to read `sample_products1.csv` instead of `sample_products.csv`.

- Notebook magics like `%pip install pandas` may appear inside notebook cells. Prefer installing packages into the virtual environment (step 3) rather than repeatedly running magics inside notebooks.

- When using VS Code: install the official Python and Jupyter extensions, then select the `.venv` interpreter/kernel before running cells.

### Troubleshooting
- If `ImportError` occurs for pandas or matplotlib: ensure the virtualenv is activated and `pip install -r requirements.txt` completed successfully.
- If plots don't render in your frontend, add `%matplotlib inline` at the top of plotting cells or use the Jupyter frontend's plot renderer settings.

### Next steps I already applied
- I created a minimal `requirements.txt` containing the packages observed in the notebooks (pandas, matplotlib, jupyter). (See repository root.)
- I reviewed the notebooks to extract the prerequisites and usage steps included above.

If you want, I can also:
- Update `01_hands_on.ipynb` to consistently write/read the same CSV filename, or
- Add a brief CONTRIBUTING or DEVSETUP script to automate venv creation and install (e.g., `scripts/setup.sh`).
