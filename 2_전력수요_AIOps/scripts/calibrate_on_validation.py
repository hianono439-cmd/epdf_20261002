"""
🔧 [전력수요 변경] (신규 파일) 검증 기간(2024)으로 하이퍼파라미터·기준값을 정하는 스크립트.

2025년은 마지막 평가에만 써야 하므로, 아래 결정은 모두 2024년 검증 데이터로 합니다.
    학습 2021~2023  →  검증 2024  (스케일러도 2021~2023으로만 fit)
    1) 모델 크기 비교   : LSTM (32,32,16) vs (64,32,16)
    2) 기준값(RMSE)     : 2024년 28일 구간별 RMSE 최댓값을 5 GWh 단위로 올림
                          → drift_detector.RMSE_THRESHOLD, train_and_register.RMSE_GATE 에 사용
    3) 드리프트 시나리오: 2024년 42일 구간에 시나리오를 적용했을 때 기준값을 넘는지 확인하고,
                          ramp 크기는 검증 구간 100%에서 감지되는 가장 작은 값으로 정함
    4) 냉·난방도일 기준온도: 2021~2023 만으로 다시 골라도 14℃/19℃ 인지 확인

결정이 끝나면 최종 모델은 scripts/train_baseline_v1.py, serving_app/train_and_register.py 가
2021~2024 전체로 다시 학습하고, 2025년으로 한 번만 평가합니다.

실행: python scripts/calibrate_on_validation.py   (결과는 docs/검증기간_보정결과.json 에도 저장)
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tensorflow import keras

from data import features
from data.drift_scenarios import apply_scenario
from data.features import SEQ_LEN, N_FEATURES, PowerScaler, build_sequences, load_rows

TRAIN_END = "2024-01-01"  # 이 날짜 전 = 학습
VAL_END = "2025-01-01"  # 이 날짜 전 = 검증 (2025는 읽기만 하고 쓰지 않음)
WINDOW = 28
EPOCHS = 100
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "검증기간_보정결과.json")


def build(units):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(units[0], return_sequences=True),
            keras.layers.LSTM(units[1], return_sequences=True),
            keras.layers.LSTM(units[2]),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a - b) / a) * 100)


def choose_degree_day_bases(rows):
    """2021~2023 만으로 냉·난방도일 기준온도를 다시 고른다 (요일·휴일 더미 + HDD + CDD 선형회귀 R²)."""
    from datetime import date

    y = np.array([r["Demand"] for r in rows])
    t = np.array([r["Temp"] for r in rows])
    dow = np.array([date.fromisoformat(r["Date"]).weekday() for r in rows])
    hol = np.array([r["Holiday"] for r in rows])
    base = np.column_stack([np.ones(len(rows))] + [(dow == k).astype(float) for k in range(1, 7)] + [hol])
    best = None
    for hb in np.arange(10, 20.5, 0.5):
        for cb in np.arange(18, 26.5, 0.5):
            X = np.column_stack([base, np.clip(hb - t, 0, None), np.clip(t - cb, 0, None), np.arange(len(rows))])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if best is None or r2 > best[0]:
                best = (float(r2), float(hb), float(cb))
    return best


def main():
    np.seterr(all="ignore")
    rows_all = load_rows("data/sample_power_daily.csv")
    rows = [r for r in rows_all if r["Date"] < VAL_END]  # 2025 행은 여기서 버림
    train_rows = [r for r in rows if r["Date"] < TRAIN_END]

    r2, hb, cb = choose_degree_day_bases(train_rows)
    print(f"[4] 냉·난방도일 기준온도 (2021~2023 만): HDD {hb}℃ / CDD {cb}℃ (R²={r2:.4f}) "
          f"| 현재 설정 HDD {features.HDD_BASE} / CDD {features.CDD_BASE}")

    scaler = PowerScaler().fit(train_rows)
    X, y, dates = build_sequences(rows, scaler)
    split = next(k for k, d in enumerate(dates) if d >= TRAIN_END)
    X = np.array(X, dtype="float32")
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = np.array(y[:split]), np.array(y[split:])
    y_tr_s = np.array([scaler.scale_demand(v) for v in y_tr], dtype="float32")
    print(f"학습 {len(X_tr)}개 ({dates[0]}~{dates[split - 1]}) / 검증 {len(X_val)}개 ({dates[split]}~{dates[-1]})")

    # 1) 모델 크기 비교
    results, models = {}, {}
    for units in [(32, 32, 16), (64, 32, 16)]:
        keras.utils.set_random_seed(42)
        m = build(units)
        m.fit(X_tr, y_tr_s, epochs=EPOCHS, verbose=0)
        pred = np.array([scaler.inverse_demand(v) for v in m.predict(X_val, verbose=0).flatten()])
        results[str(units)] = {"rmse": rmse(y_val, pred), "mape": mape(y_val, pred)}
        models[units] = (m, pred)
        print(f"[1] LSTM {units}: 검증 RMSE {results[str(units)]['rmse']:.1f} GWh, MAPE {results[str(units)]['mape']:.2f}%")

    all_y = np.array([r["Demand"] for r in rows])
    n_val = len(y_val)
    naive = {}
    for name, lag in [("어제와 같음", 1), ("지난주 같은 요일", 7)]:
        p = all_y[len(all_y) - n_val - lag : len(all_y) - lag]
        naive[name] = {"rmse": rmse(y_val, p), "mape": mape(y_val, p)}
        print(f"    기준선 {name}: 검증 RMSE {naive[name]['rmse']:.1f} GWh, MAPE {naive[name]['mape']:.2f}%")

    # 채택: 검증 RMSE 차이가 1 GWh 미만이면 더 작은(원본) 모델
    small, big = (32, 32, 16), (64, 32, 16)
    chosen = big if results[str(big)]["rmse"] < results[str(small)]["rmse"] - 1.0 else small
    print(f"    → 채택: {chosen}")
    m, pred = models[chosen]

    # 2) 기준값: 검증 기간 28일 구간별 RMSE
    err2 = (y_val - pred) ** 2
    roll = np.sqrt(np.convolve(err2, np.ones(WINDOW) / WINDOW, mode="valid"))
    threshold = float(math.ceil(roll.max() / 5) * 5)
    worst_end = dates[split + WINDOW - 1 + int(roll.argmax())]
    print(f"[2] 2024년 28일 구간 RMSE: 중앙값 {np.median(roll):.1f} / 95% {np.percentile(roll, 95):.1f} / "
          f"최댓값 {roll.max():.1f} (~{worst_end}) → 기준값 {threshold:.0f} GWh")

    # 3) 드리프트 시나리오 확인 (2024년 42일 구간, 7일 간격)
    val_rows = rows[split:]  # 정답이 2024인 첫 샘플의 창문 시작 행부터
    scen = {}
    cases = [("normal", None)] + [(f"ramp+{int(m * 100)}%", m) for m in (0.20, 0.25, 0.30, 0.35)] + [("level", None), ("temp_bias", None)]
    for name, ramp_max in cases:
        vals = []
        for k in range(0, len(val_rows) - (WINDOW + SEQ_LEN) + 1, 7):
            sl = [dict(r) for r in val_rows[k : k + WINDOW + SEQ_LEN]]
            if name.startswith("ramp"):
                apply_scenario(sl, "ramp", ramp_max)
            elif name != "normal":
                apply_scenario(sl, name)
            Xs, ys, _ = build_sequences(sl, scaler)
            ps = [scaler.inverse_demand(v) for v in m.predict(np.array(Xs, dtype="float32"), verbose=0).flatten()]
            vals.append(rmse(ys, ps))
        vals = np.array(vals)
        detected = float((vals > threshold).mean() * 100)
        scen[name] = {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max()),
                      "over_threshold_pct": detected, "n": int(len(vals))}
        print(f"[3] {name:10s}: RMSE 중앙값 {np.median(vals):.1f} / 최소 {vals.min():.1f} / 최대 {vals.max():.1f} "
              f"→ 기준값 초과 {detected:.0f}% ({len(vals)}구간)")

    ramp_ok = [m for n_, m in cases if m is not None and scen[n_]["over_threshold_pct"] == 100.0]
    print(f"    → ramp 크기: 검증 100% 감지되는 최소값 = +{int(min(ramp_ok) * 100)}%" if ramp_ok else "    → 100% 감지되는 ramp 없음")

    out = {
        "split": {"train": f"{dates[0]}~{dates[split - 1]}", "validation": f"{dates[split]}~{dates[-1]}",
                  "test(미사용)": "2025-01-01~2025-12-31"},
        "degree_day_bases_2021_2023": {"HDD": hb, "CDD": cb, "r2": r2},
        "model_size": results, "chosen_units": list(chosen), "naive": naive,
        "rolling28_rmse": {"median": float(np.median(roll)), "p95": float(np.percentile(roll, 95)),
                           "max": float(roll.max()), "max_window_end": worst_end},
        "threshold_gwh": threshold, "scenarios": scen,
        "ramp_max_chosen": min(ramp_ok) if ramp_ok else None,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved -> {os.path.normpath(OUT_JSON)}")


if __name__ == "__main__":
    main()
