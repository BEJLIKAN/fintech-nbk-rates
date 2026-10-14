"""Повторяет зафиксированный анализ без изменения сетки параметров."""
from pathlib import Path
import os, runpy, shutil
import matplotlib
matplotlib.use("Agg")
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
from src.train import train
for name in ["01_eda", "02_baseline"]:
    print("RUN", name)
    runpy.run_path(str(ROOT / "scripts" / (name + ".py")))
train()
runpy.run_path(str(ROOT / "scripts/04_evaluation.py"))
print("Результаты: results/reproduced/stage4_results")
