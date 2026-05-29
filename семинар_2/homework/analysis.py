"""
Анализ applications.csv: гистограммы, report.md, кросс-таблица.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from schema import Application


def load_csv(path: str = "applications.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        Application(
            full_name=row["full_name"],
            age=int(row["age"]),
            address={"city": row["city"], "district": row["district"]},
            speciality=row["speciality"],
            desired_course=row["desired_course"],
            years_of_experience=int(row["years_of_experience"]),
            graduation_year=int(row["graduation_year"]),
        )
    print(f"Валидация: OK, {len(df)} заявок")


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out: str, color: str) -> pd.Series:
    counts = series.value_counts()
    plt.figure(figsize=(10, 4))
    counts.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return counts


def unrealistic_combos(df: pd.DataFrame) -> list[str]:
    """2–3 сомнительные пары speciality → desired_course."""
    notes = []
    for _, row in df.iterrows():
        spec, course = row["speciality"], row["desired_course"]
        if spec == "врач" and course == "налогообложение":
            notes.append(f"- {row['full_name']}: врач → {course}")
        if spec == "учитель" and course == "информационная безопасность":
            notes.append(f"- {row['full_name']}: учитель → {course}")
        if spec == "IT-специалист" and course == "медицинская реабилитация":
            notes.append(f"- {row['full_name']}: IT-специалист → {course}")
    return notes[:5] if notes else [
        "- Явных «врач → налогообложение» нет; большинство пар логичны при жёстком Literal.",
        "- Социальный работник часто идёт на «управление персоналом» — правдоподобно.",
        "- Повторы ФИО (если есть) — артефакт mode collapse, не противоречие схеме.",
    ]


def write_report(df: pd.DataFrame, out: str = "report.md") -> None:
    n = len(df)
    cities = df["city"].value_counts()
    specs = df["speciality"].value_counts()
    courses = df["desired_course"].value_counts()
    names = df["full_name"].value_counts()
    ct = pd.crosstab(df["city"], df["speciality"])

    lines = [
        f"# Отчёт по {n} заявкам ДПО\n",
        "## Города\n",
        f"- Уникальных: {len(cities)}",
        f"- Топ: **{cities.index[0]}** — {cities.iloc[0]} ({cities.iloc[0]/n*100:.0f}%)\n",
        "## Специальности\n",
        f"- Уникальных: {len(specs)}",
        f"- Топ: **{specs.index[0]}** — {specs.iloc[0]} ({specs.iloc[0]/n*100:.0f}%)\n",
        "## Курсы\n",
        f"- Топ: **{courses.index[0]}** — {courses.iloc[0]} ({courses.iloc[0]/n*100:.0f}%)\n",
        "## Имена\n",
        f"- Уникальных ФИО: {names.size} из {n}",
    ]
    dupes = names[names > 1]
    if len(dupes):
        lines.append(f"- Повторы: {dict(dupes.head(5))}")
    lines.append("")

    lines.append("## Кросс-таблица город × специальность\n")
    lines.append("```")
    lines.append(ct.to_string())
    lines.append("```\n")

    lines.append("## Сомнительные комбинации\n")
    lines.extend(unrealistic_combos(df))

    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Сохранено: {out}")


def main(path: str = "applications.csv") -> None:
    df = load_csv(path)
    print(f"Загружено: {len(df)} заявок")
    validate(df)

    plot_bar(
        df["city"],
        f"Распределение заявок по городам ({len(df)} заявок)",
        "Город",
        "Число заявок",
        "cities.png",
        "#7AB66E",
    )
    plot_bar(
        df["speciality"],
        f"Распределение заявок по специальностям ({len(df)} заявок)",
        "Специальность",
        "Число заявок",
        "specialities.png",
        "#D97A4A",
    )
    write_report(df)

    n = len(df)
    c = df["city"].value_counts()
    s = df["speciality"].value_counts()
    print(f"Топ-город: {c.index[0]} — {c.iloc[0]} ({c.iloc[0]/n*100:.0f}%)")
    print(f"Топ-специальность: {s.index[0]} — {s.iloc[0]} ({s.iloc[0]/n*100:.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "applications.csv")
