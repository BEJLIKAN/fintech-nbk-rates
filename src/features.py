import pandas as pd

targets = {"business":"Y_business_change_pp", "households":"Y_households_change_pp"}

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
    features["inflation_up_lag2"] = (
        change.clip(lower=0).shift(lag)
    )
    features["inflation_down_lag2"] = (
        change.clip(upper=0).shift(lag)
    )

    features["base_rate_lag2"] = base.shift(lag)
    features["base_change_lag2"] = base.diff().shift(lag)

    features["inflation_mean3_lag2"] = (
        inflation.rolling(3).mean().shift(lag)
    )
    features["inflation_std3_lag2"] = (
        inflation.rolling(3).std().shift(lag)
    )

    features["real_base_proxy_lag2"] = (
        base - inflation
    ).shift(lag)

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
