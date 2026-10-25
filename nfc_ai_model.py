"""
Модель: потребность в когнитивной завершённости (NFC) пользователя и закрытие вопроса,
заданного ИИ-ассистенту. Демонстрационная формализация теории (Kruglanski & Webster 1996:
тенденции безотлагательности — seizing — и постоянства — freezing), не эмпирические данные.

Один эпизод = один пользователь задаёт один вопрос с бинарным верным ответом.
"""
import numpy as np

DEFAULTS = dict(
    accuracy=0.75,      # доля верных первых ответов ИИ
    style="categorical",# categorical | hedged | calibrated
    stakes=0.5,         # цена ошибки для пользователя (0..1): повышает порог (страх неверного закрытия)
    verify=0.5,         # вероятность, что проверка пользователя вскроет ошибку, если ответ неверен
    sycophancy=0.3,     # вероятность, что ИИ под нажимом сменит ответ, если ошибка не вскрыта
    max_rounds=6,
    cue_wrong=0.5,      # после закрытия: вероятность, что появится верное опровержение неверного ответа
    cue_false=0.15,     # после закрытия: вероятность ложного опровержения верного ответа
    openness=0.8,       # максимальная готовность пересмотреть закрытый вопрос (при NFC = 0)
    noise=0.08,         # шум восприятия уверенности ответа
    calib_quality=0.5,
    th_base=0.70, th_stakes=0.30, th_nfc=0.50,   # порог = th_base + th_stakes·цена − th_nfc·NFC
    urg_base=0.05, urg_nfc=0.15,                 # снижение порога за раунд = urg_base + urg_nfc·NFC  # калиброванный стиль: 0 — формулировка не связана с правильностью, 1 — ИИ всегда знает, когда неправ
)

# Средняя уверенность людей в ответе LLM при разных формулировках; значения сняты с рис. 4
# Steyvers et al. (2025), Nature Machine Intelligence 7, 221–231 (приблизительно, по графику).
CONF = dict(sure=0.80, somewhat=0.70, notsure=0.53)

def expressed_conf(correct, p, rng):
    style = p["style"]
    if style == "categorical":
        return CONF["sure"]
    if style == "hedged":
        return CONF["somewhat"]
    # calibrated: формулировка совпадает с правильностью с вероятностью 0.5 + 0.5·качество
    match = rng.random() < 0.5 + 0.5 * p["calib_quality"]
    return CONF["sure"] if (correct == match) else CONF["notsure"]

def threshold(nfc, stakes, p=None):
    # порог субъективной уверенности, при котором пользователь считает вопрос решённым
    p = p or DEFAULTS
    return np.clip(p["th_base"] + p["th_stakes"] * stakes - p["th_nfc"] * nfc, 0.05, 0.99)

def urgency(nfc, p=None):
    # скорость, с которой порог снижается с каждым раундом (безотлагательность)
    p = p or DEFAULTS
    return p["urg_base"] + p["urg_nfc"] * nfc

def episode(nfc, p, rng, trace=False):
    correct = rng.random() < p["accuracy"]
    conf = expressed_conf(correct, p, rng)
    th0 = threshold(nfc, p["stakes"], p)
    log = []
    rounds = 0
    syc_flip_to_wrong = False
    while True:
        perceived = conf + rng.normal(0, p["noise"])
        th = max(0.05, th0 - urgency(nfc, p) * rounds)
        if trace:
            log.append(dict(round=rounds, correct=bool(correct), conf=round(float(conf), 3),
                            perceived=round(float(perceived), 3), threshold=round(float(th), 3)))
        if perceived >= th or rounds >= p["max_rounds"]:
            break
        rounds += 1
        # проверка / уточнение
        if (not correct) and rng.random() < p["verify"]:
            # ошибка вскрыта: новый ответ ИИ
            correct = rng.random() < p["accuracy"]
            conf = expressed_conf(correct, p, rng)
            if trace: log[-1]["action"] = "проверка вскрыла ошибку, новый ответ"
        elif rng.random() < p["sycophancy"]:
            # ИИ уступает нажиму
            if correct:
                syc_flip_to_wrong = True
            correct = not correct
            conf = CONF["somewhat"]
            if trace: log[-1]["action"] = "ИИ сменил ответ под нажимом"
        else:
            conf = min(0.99, conf + 0.10)  # ответ устоял под нажимом
            if trace: log[-1]["action"] = "ответ устоял"
    accepted_correct = correct
    # постоянство (freezing): пересмотр после закрытия
    revise = p["openness"] * (1 - nfc)
    if correct and rng.random() < p["cue_false"] and rng.random() < revise:
        correct = False
    elif (not correct) and rng.random() < p["cue_wrong"] and rng.random() < revise:
        correct = True
    out = dict(final_correct=correct, accepted_correct=accepted_correct, rounds=rounds,
               syc_to_wrong=syc_flip_to_wrong)
    if trace:
        out["log"] = log
    return out

def run(nfc_values, n=4000, seed=0, **kw):
    p = {**DEFAULTS, **kw}
    rng = np.random.default_rng(seed)
    res = []
    for v in nfc_values:
        eps = [episode(v, p, rng) for _ in range(n)]
        res.append(dict(
            nfc=v,
            final_error=1 - np.mean([e["final_correct"] for e in eps]),
            accepted_error=1 - np.mean([e["accepted_correct"] for e in eps]),
            rounds=np.mean([e["rounds"] for e in eps]),
            syc_to_wrong=np.mean([e["syc_to_wrong"] for e in eps]),
        ))
    return res
