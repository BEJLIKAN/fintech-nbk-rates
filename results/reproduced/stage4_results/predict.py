from src.features import make_features, make_macro_features

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib

saved = joblib.load(Path(__file__).with_name("model.pkl"))
config = saved["config"]
# Этот артефакт уже оценивался на test.
config["test_used"] = True
baseline_models = saved["baseline_models"]
boosting_models = saved["boosting_models"]
FEATURE_LAG = int(config["feature_lag_months"])

targets = {
    "business": "Y_business_change_pp",
    "households": "Y_households_change_pp",
}






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
