import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_psi(expected, actual, bins=5):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()

    if np.unique(expected).size == 1:
        value = expected[0]

        cuts = np.array([
            -np.inf,
            value,
            np.nextafter(value, np.inf),
            np.inf
        ])
    else:
        quantiles = np.unique(
            np.quantile(expected, np.linspace(0, 1, bins + 1))
        )
        cuts = np.r_[-np.inf, quantiles, np.inf]

    expected_counts = np.histogram(expected, bins=cuts)[0]
    actual_counts = np.histogram(actual, bins=cuts)[0]

    smoothing = 0.5
    number_bins = len(expected_counts)

    e = (
        (expected_counts + smoothing) /
        (len(expected) + smoothing * number_bins)
    )
    a = (
        (actual_counts + smoothing) /
        (len(actual) + smoothing * number_bins)
    )

    return float(np.sum((a - e) * np.log(a / e)))
