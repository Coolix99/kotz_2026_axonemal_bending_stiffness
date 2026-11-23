# Axonemal bending stiffness of *Chlamydomonas* cilia implies single-motor forces above 5 pN

**Maximilian Kotz, Veikko F. Geyer, and Benjamin M. Friedrich**

Code and data to reproduce Figures 2–4 and S2–S6 of the paper.

Preprint: [https://arxiv.org/abs/2609.39562](https://arxiv.org/abs/2609.39562)

## Usage

Requires Python with `numpy` and `matplotlib`.

```bash
pip install numpy matplotlib
python make_figures.py
```

Figures are saved as PNG and PDF in `output_png/` and `output_pdf/`. Numerical results are printed to the terminal and saved in `analysis_report.txt`.

## Data

All required preprocessed data are included in `data/`, primarily originating from:

- **Sharma et al. (2024), PNAS:** Experimental axoneme beat patterns ([Zenodo](https://zenodo.org/records/13881397)).
- **Striegler et al. (2025), Nature Physics:** Three-dimensional axoneme beat patterns ([Zenodo](https://zenodo.org/records/13909287)).

The folder also contains derived phase-averaged waveforms, idealized beat patterns, and precomputed hydrodynamic force moments used in the analysis.
