# LymphXPress

XGBoost ensemble classifier for DLBCL molecular subtypes (MCD, A53, N1, BN2, EZB, ST2, Other) based on gene expression data.

## Install locally

Requires Python 3.10–3.12. Clone the repo and install in editable mode:

```bash
git clone https://github.com/arberqoku/lymphxpress
cd lymphxpress
pip install -e .
```

## CLI usage

```bash
lymphxpress path/to/expression.csv [OPTIONS]

Options:
  -m, --models INT          Number of ensemble models (default: 100)
  -s, --scaler TEXT         Scaler: rnaseq_dge | rnapanel (default: rnaseq_dge)
      --no-scaler           Skip scaling
  -r, --seed INT            Random seed (default: 0)
      --out TEXT            Output CSV path (default: results.csv)
```

**Input format:** CSV with samples as rows and gene symbols as columns (first column = sample ID).

```
sample_id,GENE1,GENE2,...
S1,1.23,4.56,...
S2,0.78,2.34,...
```

## Python API

```python
import pandas as pd
from lymphxpress import LymphxpressClassifier

X_test = pd.read_csv("expression.csv", index_col=0)

clf = LymphxpressClassifier(n_models=100)
clf.add_test_data(X_test, scaler="rnaseq_dge")
clf.fit(seed=0)

predictions = clf.predict()       # pandas Series: sample → subtype
probabilities = clf.predict_proba()  # DataFrame: sample × subtype
```

## Web server (Docker)

Requires Docker and Docker Compose.

```bash
cd web
docker compose up --build
```

The app is served at `http://localhost`. Set `SMTP_ENABLED=true` and configure SMTP env vars in `docker-compose.yml` to enable email verification in production.

## Citation

> Manuscript in preparation.
