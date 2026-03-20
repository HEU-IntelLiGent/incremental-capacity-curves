# Incremental Capacity Curves

Computation of incremental capacity curves using various smoothing methods, and using a histogram-based method, inspired by Feng et al. [DOI: 10.1016/j.jpowsour.2013.01.018](http://doi.org/10.1016/j.jpowsour.2013.01.018)

This repository accompanies the publication by Flores and Clark [DOI: XXXX](YYYYY)

# Structure
* `dqdvs.py`: functions to compute incremental capacity curves using various methods.

* `fig_dqdvs_*.ipynb`: notebooks with figures for the manuscript.

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

# License
MIT License

# Aknowledgements
The authors acknowledge funding from the European Union’s Horizon Europe research and innovation programme under [IntelLiGent (grant number No. 101069765)](https://doi.org/10.3030/101069765).
