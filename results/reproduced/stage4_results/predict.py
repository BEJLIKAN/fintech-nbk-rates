
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


def make_macro_features(frame, lag=2):
    frame = frame.sort_values("month").reset_index(drop=True)
    features = pd.DataFrame(index=frame.index)

    inflation = frame["inflation_yoy_pct"]
    base = frame["base_rate_pct"]
    change = inflation.diff().round(10)

    features["inflation_yoy_lag2"] = inflation.shift(lag)
    features["inflation_mom_lag2"] = (
        frame["inflation_mom_pct"].shift(lag)
    )

    features["inflation_change_lag2"] = change.shift(lag)
    features["inflation_up_lag2"] = change.clip(lower=0).shift(lag)
    features["inflation_down_lag2"] = change.clip(upper=0).shift(lag)

    features["base_rate_lag2"] = base.shift(lag)
    features["base_change_lag2"] = base.diff().shift(lag)

    features["inflation_mean3_lag2"] = (
        inflation.rolling(3).mean().shift(lag)
    )
    features["inflation_std3_lag2"] = (
        inflation.rolling(3).std().shift(lag)
    )

    features["real_base_proxy_lag2"] = (base - inflation).shift(lag)

    return features


def make_features(frame, group, lag=2):
    frame = frame.sort_values("month").reset_index(drop=True)
    features = make_macro_features(frame, lag)

    rate = frame[f"loan_{group}_pct"]
    change = frame[targets[group]]

    features["loan_rate_lag2"] = rate.shift(lag)
    features["loan_change_lag2"] = change.shift(lag)
    features["loan_change_lag3"] = change.shift(lag + 1)

    features["loan_change_mean3_lag2"] = (
        change.rolling(3).mean().shift(lag)
    )
    features["loan_change_std3_lag2"] = (
        change.rolling(3).std().shift(lag)
    )

    features["loan_base_spread_lag2"] = (
        rate - frame["base_rate_pct"]
    ).shift(lag)

    return features


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
