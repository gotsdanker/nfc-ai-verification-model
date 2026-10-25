"""Проверка устойчивости вывода 1: коэффициенты меняются по одному (точность ИИ 95 %,
средняя цена ошибки, категоричный стиль). Печатает долю итоговых ошибок при баллах 1 и 6
и долю ошибок сразу после диалога (до пересмотра). Результат — sensitivity.csv.
"""
import csv
from concurrent.futures import ProcessPoolExecutor
from nfc_ai_model import run

N = 60_000
NFC = [0.0, 1.0]  # баллы 1 и 6
BASE = dict(accuracy=0.95, stakes=0.5, style="categorical")
VARIANTS = {"base": {}}
for v in (0.90, 0.98):
    VARIANTS[f"accuracy={v}"] = {"accuracy": v}
for v in (0.2, 0.8):
    VARIANTS[f"verify={v}"] = {"verify": v}
for v in (0.0, 0.05, 0.1, 0.15, 0.5):
    VARIANTS[f"sycophancy={v}"] = {"sycophancy": v}
for v in (0.05, 0.1, 0.25):
    VARIANTS[f"cue_false={v}"] = {"cue_false": v}
for v, lab in ((5 / 12, "0.5"), (1 / 3, "0.4")):
    VARIANTS[f"threshold_per_score={lab}"] = {"th_nfc": v}  # 0,6 балла уверенности на балл шкалы по умолчанию
SEEDS = {name: 31000 + i for i, name in enumerate(VARIANTS)}


def job(name):
    res = run(NFC, n=N, seed=SEEDS[name], **{**BASE, **VARIANTS[name]})
    return name, res


def main():
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(job, VARIANTS))
    rows = []
    for name, res in out.items():
        rows.append(dict(variant=name,
                         final_score1=round(100 * float(res[0]["final_error"]), 1),
                         final_score6=round(100 * float(res[1]["final_error"]), 1),
                         after_dialogue_score1=round(100 * float(res[0]["accepted_error"]), 1)))
    with open("sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
