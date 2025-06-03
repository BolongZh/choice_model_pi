# Choice Models and Permutation Invariance

This repository contains the official code for the paper  
**"Choice Models and Permutation Invariance: Demand Estimation in Differentiated Products Markets"**  
[SSRN Working Paper #4508227](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4508227)

---

## Overview

To reproduce the results, you only need to run the Jupyter notebook:

📓 **`Inference.ipynb`** — This notebook provides a tutorial that replicates the main experiment from Section 3.4 of the paper. It covers both the **plug-in** and **debiased** estimation methods.

All necessary functions are defined in the following supporting scripts:

| File / folder | Description |
|---------------|-------------|
| `prediction.py` | Contains data-generation routines, the permutation-invariant neural network, training logic, and plug-in evaluation helpers. |
| `debiase.py`    | Implements the debiasing procedure: moment construction, α-network training, and final θ inference. |


In addition,
  - `requirements.txt` lists all required Python packages. 
  - `results/` is a directory to store intermediate results from 100 random simulation draws. 

---

## Quick start

```bash
git clone https://github.com/yeliu929/choice_model_pi.git
cd choice_model_pi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab inference.ipynb   # Launch the tutorial
```

---

If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{singh2023choice,
  title     = {Choice Models and Permutation Invariance: Demand Estimation in Differentiated Products Markets},
  author    = {Singh, Amandeep and Liu, Ye and Yoganarasimhan, Hema},
  journal   = {arXiv preprint arXiv:2307.07090},
  year      = {2025}
}
```
---
Contact: yeliu@uw.edu
