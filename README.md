# MLOps linear regression exercise

The current model is ordinary least squares using `x3` and `x4`, selected as the best two-predictor combination among `x1`–`x4` on a fixed validation split of the 5,394-row `sampregdata.csv`; its untouched test-set R² is **0.5239** and RMSE is **7.0729**. The previous model, preserved at Git tag `v1-one-feature`, uses only `x4` and achieves test R² **0.2686** and RMSE **8.7666** on the same split. Thus the two-predictor version improves test RMSE by **1.6937** (19.3%). The `Unnamed: 0` column is an exported row index and is excluded from candidates. The current fitted model is `y ≈ 4.8132 + 0.5846·x3 − 1.3870·x4`; both versions choose features on validation data, refit using training plus validation data, and evaluate once on the held-out test set.

## Run

```bash
python -m venv .venv
# Activate .venv using your operating system's shell command.
python -m pip install -r requirements.txt
python src/train.py --features 2
```

The committed dataset is `data/sampregdata.csv`. Running the script writes `models/current.joblib` and `results/metrics.json` locally. For the earlier version, run `git checkout v1-one-feature`, then `python src/train.py --features 1`; return with `git switch main` (or the branch printed by `git branch --show-current`). Only load model files from a trusted source. Git tracks code and the dataset; generated model binaries are reproducible and ignored.

To give a classmate access, push this repository to a private GitHub repository and invite their verified GitHub username under **Settings → Collaborators**. After they accept, they can clone the repository and run the commands above. No classmate username or repository destination was supplied, so that invitation has not been sent.
