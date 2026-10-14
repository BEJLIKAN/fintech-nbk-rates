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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

V6_RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)

print("Библиотеки подключены")

v6_path = (ROOT / "data/NBK_BNS_variant6_2016_2025.csv")

print("CSV-файлы в Colab:")
for file in (ROOT / "data").glob("*.csv"):
    print(file.name)

assert v6_path.exists(), (
    "Файл не найден. Загрузите NBK_BNS_variant6_2016_2025.csv "
    "в раздел «Файлы» слева."
)

print("\nНужный файл найден!")

v6_raw = pd.read_csv(v6_path, encoding="utf-8-sig")
v6_data = v6_raw.copy()

print("Количество строк:", len(v6_data))
print("Количество столбцов:", len(v6_data.columns))
print("\nНазвания столбцов:")
print(v6_data.columns.tolist())

v6_expected_columns = [
    "month",
    "loan_business_pct",
    "loan_households_pct",
    "inflation_mom_pct",
    "inflation_yoy_pct",
    "base_rate_pct",
    "Y_business_change_pp",
    "Y_households_change_pp",
]

assert set(v6_data.columns) == set(v6_expected_columns), (
    "Состав столбцов отличается от ожидаемого."
)

v6_data = v6_data[v6_expected_columns].copy()

print("Все необходимые столбцы присутствуют")

v6_data["month"] = pd.to_datetime(
    v6_data["month"],
    format="%Y-%m",
    errors="raise"
)

v6_data = (
    v6_data
    .sort_values("month")
    .reset_index(drop=True)
)

print("Начало периода:", v6_data["month"].min().strftime("%Y-%m"))
print("Конец периода:", v6_data["month"].max().strftime("%Y-%m"))

v6_numeric_columns = [
    column for column in v6_expected_columns
    if column != "month"
]

for column in v6_numeric_columns:
    v6_data[column] = pd.to_numeric(
        v6_data[column],
        errors="raise"
    )

assert np.isfinite(
    v6_data[v6_numeric_columns].to_numpy()
).all(), "Обнаружены пропуски или бесконечные значения."

print("Все показатели имеют корректный числовой формат")

v6_expected_months = pd.date_range(
    start="2016-01-01",
    end="2025-12-01",
    freq="MS"
)

v6_actual_months = pd.DatetimeIndex(v6_data["month"])

assert len(v6_data) == 120, "Ожидалось 120 месяцев."
assert not v6_data["month"].duplicated().any(), "Есть повторные месяцы."
assert v6_actual_months.equals(v6_expected_months), (
    "Есть пропущенные или неожиданные месяцы."
)

print("120 последовательных месяцев")
print("Повторных месяцев нет")
print("Пропущенных месяцев нет")

v6_n = len(v6_data)

v6_train_end = int(v6_n * 0.70)
v6_valid_end = int(v6_n * 0.85)

v6_train = v6_data.iloc[:v6_train_end].copy()
v6_valid = v6_data.iloc[v6_train_end:v6_valid_end].copy()
v6_test_locked = v6_data.iloc[v6_valid_end:].copy()

assert v6_train["month"].max() < v6_valid["month"].min()
assert v6_valid["month"].max() < v6_test_locked["month"].min()

print("Разделение выполнено без перемешивания")
print("Тестовая часть отложена до этапа 4")

v6_split_table = pd.DataFrame([
    {
        "Часть": name,
        "Месяцев": len(part),
        "Доля, %": len(part) / v6_n * 100,
        "Начало": part["month"].min().strftime("%Y-%m"),
        "Конец": part["month"].max().strftime("%Y-%m"),
    }
    for name, part in [
        ("Train — обучение", v6_train),
        ("Validation — настройка", v6_valid),
        ("Test — итоговая проверка", v6_test_locked),
    ]
])

display(v6_split_table)

v6_meanings = {
    "month": "Месяц наблюдения; не дата публикации",
    "loan_business_pct": "Ставка новых кредитов юрлицам в тенге, % годовых",
    "loan_households_pct": "Ставка новых кредитов физлицам, включая ИП, % годовых",
    "inflation_mom_pct": "Инфляция к предыдущему месяцу, %",
    "inflation_yoy_pct": "Инфляция к тому же месяцу прошлого года, %",
    "base_rate_pct": "Средняя базовая ставка за календарный месяц, % годовых",
    "Y_business_change_pp": "Изменение ставки юрлицам за месяц, п.п.",
    "Y_households_change_pp": "Изменение ставки физлицам за месяц, п.п.",
}

v6_dictionary = pd.DataFrame({
    "Столбец": v6_train.columns,
    "Тип": [str(v6_train[c].dtype) for c in v6_train.columns],
    "Пропуски в train, %": [
        v6_train[c].isna().mean() * 100
        for c in v6_train.columns
    ],
    "Экономический смысл": [
        v6_meanings[c] for c in v6_train.columns
    ],
})

display(v6_dictionary)

v6_targets = {
    "Юридические лица": "Y_business_change_pp",
    "Физические лица, включая ИП": "Y_households_change_pp",
}

v6_leakage_audit = pd.DataFrame([
    [
        "Кредитная ставка текущего месяца",
        "Входит в формулу текущего Y",
        "Не включать в X текущего месяца"
    ],
    [
        "Инфляция текущего месяца",
        "До публикации ещё неизвестна",
        "Использовать прошлые опубликованные значения"
    ],
    [
        "Средняя базовая ставка текущего месяца",
        "Полное среднее известно после окончания месяца",
        "Для прогноза использовать доступную прошлую информацию"
    ],
    [
        "Скользящие средние",
        "Могут случайно включить текущий или будущий месяц",
        "Рассчитывать только по прошлым доступным данным"
    ],
    [
        "Test",
        "Подбор модели по тесту завысит результат",
        "Использовать только на этапе 4"
    ],
], columns=["Показатель", "Риск", "Правило"])

display(v6_leakage_audit)

display(
    v6_train.head(10).round(3)
)

v6_descriptive = (
    v6_train[v6_numeric_columns]
    .describe()
    .T
    .rename(columns={
        "count": "Количество",
        "mean": "Среднее",
        "std": "Станд. отклонение",
        "min": "Минимум",
        "25%": "25-й процентиль",
        "50%": "Медиана",
        "75%": "75-й процентиль",
        "max": "Максимум",
    })
)

display(v6_descriptive.round(3))

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    v6_train["month"], v6_train["loan_business_pct"],
    label="Юридические лица", linewidth=2
)

ax.plot(
    v6_train["month"], v6_train["loan_households_pct"],
    label="Физические лица, включая ИП", linewidth=2
)

ax.plot(
    v6_train["month"], v6_train["base_rate_pct"],
    label="Базовая ставка", linewidth=2, linestyle="--"
)

ax.set_title("Динамика процентных ставок — обучающая выборка")
ax.set_xlabel("Месяц")
ax.set_ylabel("% годовых")
ax.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(
    v6_train["month"],
    v6_train["inflation_yoy_pct"],
    color="darkorange",
    linewidth=2
)
axes[0].set_title("Годовая инфляция — обучающая выборка")
axes[0].set_ylabel("% к тому же месяцу\nпрошлого года")

axes[1].bar(
    v6_train["month"],
    v6_train["inflation_mom_pct"],
    width=22,
    color="steelblue"
)
axes[1].set_title("Месячная инфляция — обучающая выборка")
axes[1].set_ylabel("% к предыдущему\nмесяцу")
axes[1].set_xlabel("Месяц")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

for ax, (group, target) in zip(axes, v6_targets.items()):
    ax.plot(
        v6_train["month"],
        v6_train[target],
        linewidth=1.5,
        marker="o",
        markersize=3
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(group)
    ax.set_ylabel("Изменение ставки, п.п.")

axes[1].set_xlabel("Месяц")
fig.suptitle("Целевые переменные — обучающая выборка")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (group, target) in zip(axes, v6_targets.items()):
    sns.histplot(
        data=v6_train,
        x=target,
        bins=15,
        ax=ax,
        color="steelblue"
    )
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(group)
    ax.set_xlabel("Изменение ставки, п.п.")
    ax.set_ylabel("Количество месяцев")

plt.tight_layout()
plt.show()

v6_level_labels = {
    "loan_business_pct": "Ставка юрлицам",
    "loan_households_pct": "Ставка физлицам",
    "inflation_yoy_pct": "Годовая инфляция",
    "base_rate_pct": "Базовая ставка",
}

v6_level_corr = (
    v6_train[list(v6_level_labels)]
    .rename(columns=v6_level_labels)
    .corr()
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    v6_level_corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True
)

plt.title("Корреляция уровней — только train")
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

v6_changes = pd.DataFrame({
    "Изменение ставки юрлицам":
        v6_train["Y_business_change_pp"],

    "Изменение ставки физлицам":
        v6_train["Y_households_change_pp"],

    "Изменение годовой инфляции":
        v6_train["inflation_yoy_pct"].diff(),

    "Изменение базовой ставки":
        v6_train["base_rate_pct"].diff(),
}).dropna()

plt.figure(figsize=(9, 6))

sns.heatmap(
    v6_changes.corr(),
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True
)

plt.title("Корреляция месячных изменений — только train")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

for rate, target in [
    ("loan_business_pct", "Y_business_change_pp"),
    ("loan_households_pct", "Y_households_change_pp"),
]:
    calculated = v6_train[rate].diff()

    assert np.allclose(
        calculated.iloc[1:],
        v6_train[target].iloc[1:],
        atol=1e-8
    ), f"Ошибка расчёта {target}"

print("Проверка целевых переменных пройдена.")
print()
print("ЭТАП 1:")
print("1. Данные загружены и проверены.")
print("2. Типы, пропуски и смысл столбцов описаны.")
print("3. Определены две целевые переменные.")
print("4. Выполнено разделение 70% / 15% / 15% по времени.")
print("5. Графики и корреляции рассчитаны только на train.")
print("6. Риски утечки перечислены.")
print()
print("До моделирования необходимо уточнить доступность данных")
print("на дату прогноза и зафиксировать допустимые лаги.")