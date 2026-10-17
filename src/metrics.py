import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def calculate_psi(expected, actual, bins=5):
    """PSI: границы корзин определяются только по train."""
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()

    if expected.size == 0 or actual.size == 0:
        raise ValueError("Для PSI нужны две непустые выборки.")

    if not np.isfinite(expected).all():
        raise ValueError("В train есть пропуски или бесконечности.")

    if not np.isfinite(actual).all():
        raise ValueError("В test есть пропуски или бесконечности.")

    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins должен быть целым числом не меньше 2.")

    if np.unique(expected).size == 1:
        value = expected[0]

        # Отдельная корзина для постоянного значения train.
        cuts = np.array([
            -np.inf,
            value,
            np.nextafter(value, np.inf),
            np.inf,
        ])
    else:
        quantiles = np.unique(
            np.quantile(expected, np.linspace(0, 1, bins + 1))
        )
        cuts = np.r_[-np.inf, quantiles, np.inf]

    expected_counts = np.histogram(expected, bins=cuts)[0]
    actual_counts = np.histogram(actual, bins=cuts)[0]

    # Сначала доли: размер выборки не создаёт искусственный дрейф.
    e = expected_counts / expected.size
    a = actual_counts / actual.size

    # Одинаковый минимальный уровень для нулевых долей.
    epsilon = 1e-6
    e = np.maximum(e, epsilon)
    a = np.maximum(a, epsilon)

    e = e / e.sum()
    a = a / a.sum()

    return float(np.sum((a - e) * np.log(a / e)))
