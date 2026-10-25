"""Воспроизводит таблицу 1 тезисов: доля итоговых ошибок при точности ИИ 95 %.

Запуск: python reproduce_table1.py  (около минуты на 4 ядрах)
Результат печатается и сохраняется в table1.csv.
Балл потребности (шкала 1–6) переводится в параметр модели nfc = (балл − 1) / 5.
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from nfc_ai_model import run, threshold

N = 60_000
SCORES = [1, 2, 3, 4, 5, 6]
NFC = [(s - 1) / 5 for s in SCORES]
BASE = dict(accuracy=0.95, stakes=0.5)

CONDITIONS = {
    "categorical_dialogue": dict(style="categorical"),
    "calibrated_q0.5": dict(style="calibrated", calib_quality=0.5),
    "calibrated_q1": dict(style="calibrated", calib_quality=1.0),
    "categorical_high_stakes": dict(style="categorical", stakes=1.0),
    "independent_reference": dict(style="categorical", sycophancy=0.0, cue_false=0.0, verify=1.0),
}
SEEDS = {name: 20261025 + i for i, name in enumerate(CONDITIONS)}


def job(name):
    kw = {**BASE, **CONDITIONS[name]}
    res = run(NFC, n=N, seed=SEEDS[name], **kw)
    return name, res


def main():
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(job, CONDITIONS))
    rows = []
    for j, s in enumerate(SCORES):
        row = {"score": s, "threshold_1to7": round(1 + 6 * float(threshold(NFC[j], 0.5)), 2)}
        for name in CONDITIONS:
            row[name] = round(100 * float(out[name][j]["final_error"]), 1)
        row["categorical_after_dialogue"] = round(100 * float(out["categorical_dialogue"][j]["accepted_error"]), 1)
        rows.append(row)
    with open("table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
