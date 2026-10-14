from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if not (ROOT / "src").exists():
    raise RuntimeError("Сначала скачайте весь репозиторий и перейдите в его папку")
sys.path.insert(0, str(ROOT))
WORK = ROOT / "results/reproduced"
WORK.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import zipfile
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import lightgbm as lgb
import shap
import yaml
import joblib

from IPython.display import display
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

OUTPUT = (WORK / "stage4_results")
OUTPUT.mkdir(exist_ok=True)

print("Библиотеки подключены")

csv_path = (ROOT / "data/NBK_BNS_variant6_2016_2025.csv")
zip_path = (WORK / "stage3_results.zip")

assert csv_path.exists(), "Загрузите CSV."
assert zip_path.exists(), "Загрузите stage3_results.zip."

stage3_dir = (WORK / "stage3_loaded")
stage3_dir.mkdir(exist_ok=True)

required_files = [
    "config.yaml",
    "stage3_models.joblib",
    "validation_comparison.csv",
    "requirements_stage3.txt",
]

with zipfile.ZipFile(zip_path) as archive:
    for name in required_files:
        assert name in archive.namelist(), f"В архиве нет {name}"
        (stage3_dir / name).write_bytes(archive.read(name))

print("Файлы третьего этапа извлечены")

saved = joblib.load(stage3_dir / "stage3_models.joblib")

config = saved["config"]
baseline_models = saved["baseline_models"]
boosting_models = saved["boosting_models"]

assert config["test_used"] is False
assert config["refit_before_test"] is False

FEATURE_LAG = int(config["feature_lag_months"])

TRAIN_END = pd.Timestamp(config["train_end"])
VALID_END = pd.Timestamp(config["validation_end"])

targets = {
    "business": "Y_business_change_pp",
    "households": "Y_households_change_pp",
}

group_names = {
    "business": "Юридические лица",
    "households": "Физические лица, включая ИП",
}

for group in targets:
    print(
        group_names[group],
        "— модель, выбранная на validation:",
        config["selected_models"][group]
    )

data = pd.read_csv(csv_path, encoding="utf-8-sig")

data["month"] = pd.to_datetime(
    data["month"], format="%Y-%m", errors="raise"
)

data = data.sort_values("month").reset_index(drop=True)

numeric_columns = data.columns.drop("month")
data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric, errors="raise"
)

assert np.isfinite(data[numeric_columns].to_numpy()).all()

assert pd.DatetimeIndex(data["month"]).equals(
    pd.date_range("2016-01-01", "2025-12-01", freq="MS")
)

print("Данные проверены")

from src.features import make_macro_features

from src.features import make_features

evaluation_data = {}

for group, target in targets.items():
    X = make_features(data, group, FEATURE_LAG)

    assert X.columns.tolist() == config["feature_columns"]

    usable = X.notna().all(axis=1)
    train_mask = usable & (data["month"] <= TRAIN_END)
    test_mask = usable & (data["month"] > VALID_END)

    assert train_mask.sum() == 80
    assert test_mask.sum() == 18

    evaluation_data[group] = {
        "X_train": X.loc[train_mask].copy(),
        "X_test": X.loc[test_mask].copy(),
        "y_test": data.loc[test_mask, target].copy(),
        "dates_test": data.loc[test_mask, "month"].copy(),
    }

    print(group_names[group], "— test: 18 месяцев")

def predict_model(group, name, X):
    if name == "zero_change":
        return np.zeros(len(X))

    if name == "lightgbm":
        model = boosting_models[group]
        return np.asarray(
            model.predict(X, num_iteration=model.best_iteration_)
        )

    columns = config["baseline_columns"][name]

    X_regression = sm.add_constant(
        X[columns], has_constant="add"
    )

    return np.asarray(
        baseline_models[group][name].predict(X_regression)
    )

print("Функция прогнозирования готова")

test_predictions = {}
test_rows = []

for group, parts in evaluation_data.items():
    test_predictions[group] = {}

    for name in [
        "zero_change", "symmetric", "asymmetric", "lightgbm"
    ]:
        prediction = predict_model(group, name, parts["X_test"])
        test_predictions[group][name] = prediction

        test_rows.append({
            "group": group,
            "model": name,
            "selected_on_validation":
                name == config["selected_models"][group],
            "RMSE": float(np.sqrt(mean_squared_error(
                parts["y_test"], prediction
            ))),
            "MAE": float(mean_absolute_error(
                parts["y_test"], prediction
            )),
            "R2": float(r2_score(
                parts["y_test"], prediction
            )),
        })

test_metrics = pd.DataFrame(test_rows)

display(test_metrics.round(4))

test_metrics.to_csv(
    OUTPUT / "test_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ax, group in zip(axes, targets):
    parts = evaluation_data[group]
    selected = config["selected_models"][group]

    ax.plot(
        parts["dates_test"], parts["y_test"],
        color="black", marker="o", label="Факт"
    )

    ax.plot(
        parts["dates_test"],
        test_predictions[group][selected],
        color="darkorange", marker="o",
        label=f"Выбранная модель: {selected}"
    )

    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title(group_names[group])
    ax.set_ylabel("Изменение ставки, п.п.")
    ax.legend()

axes[1].set_xlabel("Месяц")
plt.tight_layout()

plt.savefig(OUTPUT / "selected_models_test.png", dpi=160)
plt.show()

prediction_tables = []

for group, parts in evaluation_data.items():
    table = pd.DataFrame({
        "month": parts["dates_test"].to_numpy(),
        "group": group,
        "actual_change_pp": parts["y_test"].to_numpy(),
    })

    for name, prediction in test_predictions[group].items():
        table[f"prediction_{name}"] = prediction

    prediction_tables.append(table)

test_prediction_table = pd.concat(
    prediction_tables, ignore_index=True
)

test_prediction_table.to_csv(
    OUTPUT / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

display(test_prediction_table.head())

wald_rows = []

for group in targets:
    model = baseline_models[group]["asymmetric"]

    result = model.wald_test(
        "inflation_up_lag2 = inflation_down_lag2",
        use_f=False,
        scalar=True
    )

    p_value = float(result.pvalue)

    wald_rows.append({
        "group": group,
        "beta_up": float(model.params["inflation_up_lag2"]),
        "beta_down": float(model.params["inflation_down_lag2"]),
        "wald_p_value": p_value,
        "reject_symmetry_at_5pct": p_value < 0.05,
    })

wald_results = pd.DataFrame(wald_rows)
display(wald_results.round(4))

wald_results.to_csv(
    OUTPUT / "wald_train.csv",
    index=False,
    encoding="utf-8-sig"
)

boost_explanations = {}
boost_train_explanations = {}

for group, parts in evaluation_data.items():
    explainer = shap.TreeExplainer(
        boosting_models[group],
        feature_perturbation="tree_path_dependent"
    )

    boost_explanations[group] = explainer(parts["X_test"])
    boost_train_explanations[group] = explainer(parts["X_train"])

    explanation = boost_explanations[group]

    reconstructed = (
        explanation.base_values +
        explanation.values.sum(axis=1)
    )

    assert np.allclose(
        reconstructed,
        test_predictions[group]["lightgbm"],
        atol=1e-6
    )

print("SHAP рассчитан, совпадение с прогнозами проверено")

for group in targets:
    shap.plots.beeswarm(
        boost_explanations[group],
        max_display=10,
        show=False
    )

    plt.title(f"SHAP LightGBM — {group_names[group]}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT / f"shap_lightgbm_{group}.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.show()

selected_explanations = {}

for group, parts in evaluation_data.items():
    name = config["selected_models"][group]
    X_test = parts["X_test"]
    X_train = parts["X_train"]

    if name == "lightgbm":
        explanation = boost_explanations[group]

    else:
        coefficients = pd.Series(0.0, index=X_test.columns)
        intercept = 0.0

        if name != "zero_change":
            model = baseline_models[group][name]
            columns = config["baseline_columns"][name]

            coefficients.loc[columns] = model.params[columns]
            intercept = float(model.params["const"])

        train_mean = X_train.mean()

        contributions = (
            (X_test - train_mean) * coefficients
        ).to_numpy()

        base_value = float(
            intercept + (train_mean * coefficients).sum()
        )

        explanation = shap.Explanation(
            values=contributions,
            base_values=np.full(len(X_test), base_value),
            data=X_test.to_numpy(),
            feature_names=X_test.columns.tolist()
        )

    assert np.allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        test_predictions[group][name],
        atol=1e-6
    )

    selected_explanations[group] = explanation

print("Объяснения выбранных моделей готовы")

case_rows = []

for group, parts in evaluation_data.items():
    name = config["selected_models"][group]
    explanation = selected_explanations[group]
    positions = [0, len(parts["X_test"]) // 2, len(parts["X_test"]) - 1]

    for position in positions:
        month = parts["dates_test"].iloc[position].strftime("%Y-%m")

        print(
            group_names[group], month,
            "| Модель:", name,
            "| Факт:", round(float(parts["y_test"].iloc[position]), 3),
            "| Прогноз:", round(
                float(test_predictions[group][name][position]), 3
            )
        )

        if name == "zero_change":
            print("Прогноз всегда 0; все вклады признаков равны нулю.")
        else:
            shap.plots.waterfall(
                explanation[position],
                max_display=8,
                show=False
            )

            plt.title(f"{group_names[group]} — {month}")
            plt.savefig(
                OUTPUT / f"shap_case_{group}_{month}.png",
                dpi=160,
                bbox_inches="tight"
            )
            plt.show()

        contributions = pd.Series(
            explanation.values[position],
            index=parts["X_test"].columns
        )

        top = contributions.abs().nlargest(3).index

        for feature in top:
            case_rows.append({
                "group": group,
                "month": month,
                "model": name,
                "feature": feature,
                "feature_value": float(
                    parts["X_test"].iloc[position][feature]
                ),
                "shap_contribution_pp": float(contributions[feature]),
            })

case_details = pd.DataFrame(case_rows)
display(case_details.round(4))

case_details.to_csv(
    OUTPUT / "three_cases_per_group.csv",
    index=False,
    encoding="utf-8-sig"
)

from src.metrics import calculate_psi

psi_rows = []

for group, parts in evaluation_data.items():
    selected = config["selected_models"][group]

    for name in sorted({selected, "lightgbm"}):
        train_prediction = predict_model(
            group, name, parts["X_train"]
        )
        test_prediction = test_predictions[group][name]

        psi_rows.append({
            "group": group,
            "item": f"prediction_{name}",
            "PSI": calculate_psi(train_prediction, test_prediction),
        })

    importance = pd.Series(
        np.abs(boost_train_explanations[group].values).mean(axis=0),
        index=parts["X_train"].columns
    )

    top_features = importance.sort_values(
        ascending=False, kind="stable"
    ).head(5).index

    if (importance > 0).sum() < 5:
        print(group, "— менее пяти признаков с ненулевым SHAP.")

    for feature in top_features:
        psi_rows.append({
            "group": group,
            "item": feature,
            "PSI": calculate_psi(
                parts["X_train"][feature],
                parts["X_test"][feature]
            ),
        })

psi_results = pd.DataFrame(psi_rows)
psi_results["flag_above_025"] = psi_results["PSI"] > 0.25

display(psi_results.round(4))

psi_results.to_csv(
    OUTPUT / "psi_results.csv",
    index=False,
    encoding="utf-8-sig"
)

ASSUMED_LOAN_AMOUNT_KZT = 1_000_000_000
ASSUMED_HORIZON_YEARS = 1

money_results = test_metrics[
    ["group", "model", "selected_on_validation", "MAE"]
].copy()

money_results["annual_income_error_equivalent_KZT"] = (
    money_results["MAE"] / 100
    * ASSUMED_LOAN_AMOUNT_KZT
    * ASSUMED_HORIZON_YEARS
)

display(money_results.round(2))

money_results.to_csv(
    OUTPUT / "scenario_money_equivalent.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Сценарий: 1 млрд ₸, постоянный остаток, срок 1 год.")
print("Не учтены фондирование, дефолты, расходы и изменение спроса.")
print("Это масштаб ошибки, а не фактическая прибыль.")

final_bundle = {
    "baseline_models": baseline_models,
    "boosting_models": boosting_models,
    "config": config,
}

joblib.dump(final_bundle, OUTPUT / "model.pkl")

evaluation_config = dict(config)
evaluation_config["test_used"] = True
evaluation_config["money_scenario"] = {
    "loan_amount_KZT": ASSUMED_LOAN_AMOUNT_KZT,
    "horizon_years": ASSUMED_HORIZON_YEARS,
    "meaning": "Annual interest-income error equivalent, not realized profit"
}

with open(OUTPUT / "config.yaml", "w", encoding="utf-8") as file:
    yaml.safe_dump(
        evaluation_config,
        file,
        allow_unicode=True,
        sort_keys=False
    )

limitations = """
Ограничения исследования:
1. Совместная модель использует 2016–2025 вместо полного периода 2010–2025.
2. Физические лица включают ИП и не равны строго группе «население».
3. Доступность данных t−2 принята как допущение; календарь публикаций
   не проверен. Нужно учитывать и доступность целевых значений train
   на дату первого валидационного прогноза.
4. Используется актуальная историческая выгрузка, а не архив первых
   публикаций: возможны пересмотры.
5. Период 2024–2025 уже рассматривался в прежней учебной работе.
6. Test содержит только 18 месяцев; оценки качества и PSI неустойчивы.
7. SHAP объясняет модель, а не доказывает причинность.
8. Денежная оценка — условный эквивалент ошибки, не фактическая прибыль.
9. После просмотра test параметры и выбор модели не менялись
   в данном блокноте.
"""

(OUTPUT / "limitations.txt").write_text(
    limitations.strip(), encoding="utf-8"
)

print("model.pkl, config.yaml и ограничения сохранены")

def predict(df):
    frame = df.copy()
    frame["month"] = pd.to_datetime(frame["month"])
    frame = frame.sort_values("month").reset_index(drop=True)

    assert not frame["month"].duplicated().any()

    expected_dates = pd.date_range(
        frame["month"].min(),
        frame["month"].max(),
        freq="MS"
    )
    assert pd.DatetimeIndex(frame["month"]).equals(expected_dates)

    result = frame[["month"]].copy()

    for group in targets:
        X = make_features(frame, group, FEATURE_LAG)
        valid = X.notna().all(axis=1)

        result[f"predicted_{group}_change_pp"] = np.nan

        if valid.any():
            name = config["selected_models"][group]

            result.loc[valid, f"predicted_{group}_change_pp"] = (
                predict_model(group, name, X.loc[valid])
            )

    return result

# Проверяем совпадение с уже рассчитанными прогнозами.
prediction_check = predict(data)
test_mask = prediction_check["month"] > VALID_END

for group in targets:
    name = config["selected_models"][group]

    assert np.allclose(
        prediction_check.loc[
            test_mask, f"predicted_{group}_change_pp"
        ],
        test_predictions[group][name]
    )

print("predict(df) работает и воспроизводит сохранённые прогнозы")

shutil.copy2(ROOT / "predict.py", OUTPUT / "predict.py")

import importlib.metadata


packages = [
    "numpy", "pandas", "matplotlib", "statsmodels",
    "scikit-learn", "lightgbm", "PyYAML", "joblib", "shap"
]

requirements = "\n".join(
    f"{package}=={importlib.metadata.version(package)}"
    for package in packages
)

(OUTPUT / "requirements.txt").write_text(
    requirements + "\n",
    encoding="utf-8"
)

shutil.copy2(
    stage3_dir / "validation_comparison.csv",
    OUTPUT / "validation_comparison.csv"
)

archive = shutil.make_archive(
    str(WORK / "stage4_results"),
    "zip",
    root_dir=OUTPUT
)

print("Архив:", archive)