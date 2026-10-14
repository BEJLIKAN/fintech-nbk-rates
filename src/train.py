"""Обучение из зафиксированного протокола третьего этапа."""
from pathlib import Path
import runpy
import statsmodels.api as sm

def fit_baseline(y, X):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":3,"use_correction":True}, use_t=False)

def train():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/03_boosting.py"))

if __name__ == "__main__":
    train()
