# Incremental Capacity Curves

Computation of incremental capacity curves using various smoothing methods, and using a histogram-based method, inspired by Feng et al. [DOI: 10.1016/j.jpowsour.2013.01.018](http://doi.org/10.1016/j.jpowsour.2013.01.018)

This repository accompanies the publication Flores et al 2026 J. Electrochem. Soc. at [10.1149/1945-7111/ae7e5c](https://doi.org/10.1149/1945-7111/ae7e5c)

The datasets used can be found at the Zenodo repoisory at [10.5281/zenodo.20086298](https://doi.org/10.5281/zenodo.20086298).

# Citation

If you use any of the functions or scripts, please cite the publication at [10.1149/1945-7111/ae7e5c](https://doi.org/10.1149/1945-7111/ae7e5c)

# Structure
* `dqdvs.py`: functions to compute incremental capacity curves using various methods.
* `dqdv_metrics.py`: functions to compute metrics to quantify the quality of voltage reconstruciton.
* `fig_dqdvs_*.ipynb`: notebooks with figures for the manuscript.
* `/figures`: figures produced by the notebooks, used in the manuscript.

# Quickstart
1. Clone this repository
2. Navigate to the root of your local clone, create a virtual environment and install all dependencies.

Using uv (preferred)
```bash
uv venv create

source .venv/bin/activate

uv install
```     

Instead, using pip

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```
3. Run the notebooks to reproduce the figures of the main article. OR
4. Re-use the code in `dqdvs.py` and `dqdv_metrics.py` to compute IC curves and reconstruction metrics, respectively.

# License
MIT License

# Aknowledgements
The authors acknowledge funding from the European Union’s Horizon Europe research and innovation programme under [IntelLiGent (grant number No. 101069765)](https://doi.org/10.3030/101069765).
