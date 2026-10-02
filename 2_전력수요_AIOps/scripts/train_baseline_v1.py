"""
[Day1 사전 준비] 로컬 baseline 모델 만들기  —  scripts/train_baseline_v1.py

■ 이 파일이 하는 일 (한 줄 요약)
   업로드된 전력수요 CSV로 LSTM 모델을 학습시키고,
   서버가 읽어 갈 파일 2개(스케일러, 모델)를 만들어 둡니다.

■ 만들어지는 파일 (이름·위치를 바꾸지 마세요 — 서버가 이 경로로 찾습니다)
   serving_app/models/scaler.pkl      ← 숫자를 0~1로 바꿔 주는 '자' (Day1~3 내내 계속 사용)
   serving_app/models/power_v1.keras  ← 학습된 LSTM 모델

■ 실행 순서
   1) 서버 실행       : uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
   2) 데이터 업로드   : 대시보드(http://localhost:8077)에서 data/sample_power_daily.csv 업로드
   3) 이 파일 실행    : (다른 터미널에서) python scripts/train_baseline_v1.py

■ 빈칸 정답 위치 : [빈칸 1]

🔧 [전력수요 변경] 요약
   - 학습 2021~2024 / 평가 2025 (날짜 기준 분할, 원본은 앞 80% / 뒤 20%)
   - 스케일러를 "학습 기간(2025 이전)" 데이터로만 fit
     (원본은 전체 데이터로 fit → 평가 구간의 최소·최대가 학습에 새어 들어감)
   - RMSE 단위 달러 → GWh, 게이트 $4.00 → 75 GWh, MAPE(%)와 기준선(어제/지난주 같은 요일)도 함께 출력
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 이 파일에서 쓰는 도구들 (모두 이미 만들어져 있습니다)
#   load_rows        : CSV 파일 → [{"Date":..., "Demand":..., "Temp":..., "Holiday":...}, ...]
#   build_sequences  : 행 목록 → 학습용 문제(X)와 정답(y), 정답 날짜(dates)
#   train_test_split : 2025-01-01 이전은 공부용(train), 이후는 시험용(test)으로 나누기
#   PowerScaler      : 0~1로 바꾸기(transform_point, scale_demand) / GWh로 되돌리기(inverse_demand)
#   latest_upload    : data/uploads/ 에서 가장 최근 올린 CSV 경로
#   build_model      : LSTM 모델 뼈대 만들기
from data.features import load_rows, build_sequences, train_test_split, PowerScaler, TEST_START  # 🔧 [전력수요 변경]
from data.storage import latest_upload
from serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/power_v1.keras"  # 🔧 [전력수요 변경] haic_v1.keras → power_v1.keras
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 전체 문제를 100번 반복해서 학습
RMSE_GATE = 75.0  # 🔧 [전력수요 변경] $4.00 → 75 GWh (2024 검증으로 정한 값, train_and_register.py 와 같음)


def rmse(y_true, y_pred) -> float:
    """예측이 실제보다 '평균 몇 GWh' 틀렸는지 계산합니다."""
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def mape(y_true, y_pred) -> float:
    """🔧 [전력수요 변경] (신규) 평균 몇 % 틀렸는지 — 단위가 없어 기준선과 비교하기 쉬움."""
    return sum(abs(a - b) / a for a, b in zip(y_true, y_pred)) / len(y_true) * 100


def main():
    import numpy as np
    from tensorflow import keras

    keras.utils.set_random_seed(42)  # 🔧 [전력수요 변경] 재현성을 위해 시드 고정 (train_and_register.py 와 동일)

    # STEP 1. 데이터 읽기 — 가장 최근 업로드한 CSV를 행 목록으로 (2021~2025, 1,826행)
    rows = load_rows(latest_upload())

    # STEP 2. 스케일러 만들고 저장하기
    #   수요(1,000~2,000 GWh)와 기온(-13~30℃)은 크기 차이가 커서 둘 다 0~1로 맞춰 줍니다.
    # 🔧 [전력수요 변경] fit 은 학습 기간(2025 이전) 행으로만 — 평가 데이터 정보 누수 방지
    train_rows = [r for r in rows if r["Date"] < TEST_START]
    scaler = PowerScaler().fit(train_rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(train_rows)}행 (~{TEST_START} 이전) -> {SCALER_PATH}")

    # STEP 3. 문제(X)와 정답(y) 만들기
    #   "최근 14일 수요 + 다음날 기온·휴일·요일을 보고 → 다음날 수요를 맞혀라"를 하루씩 밀며 만듭니다.
    X, y, dates = build_sequences(rows, scaler)  # 🔧 [전력수요 변경] 정답 날짜(dates)도 함께 받음

    # STEP 4. 공부용 / 시험용 나누기 — 🔧 [전력수요 변경] 비율 대신 날짜 기준 (2021~2024 / 2025)
    X_train, y_train, X_test, y_test = train_test_split(X, y, dates)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    print(f"train {len(X_train)}개 / test {len(X_test)}개")

    # STEP 5. 공부용 정답(y_train)도 0~1로 바꾸기
    y_train_scaled = np.array([scaler.scale_demand(v) for v in y_train], dtype="float32")

    # STEP 6. 빈 LSTM 모델 만들기
    model = build_model()

    # ✅ [빈칸 1 정답] 문제=X_train, 정답=y_train_scaled (0~1로 바꾼 학습용 정답)
    #    (원래 적혀 있던 y_test 는 의도된 오답: 시험 정답이 학습에 섞이고 X_train 과 길이도 안 맞음)
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    # STEP 7. 시험 보기 — 모델 출력(0~1)을 GWh로 되돌린 뒤 실제 정답(y_test)과 비교
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = [scaler.inverse_demand(p) for p in preds_scaled]  # 실제 GWh 단위로 복원
    score = rmse(y_test, preds)
    print(f"baseline v1 RMSE = {score:.1f} GWh, MAPE = {mape(y_test, preds):.2f}%  (배포 게이트: {RMSE_GATE:.0f} GWh)")

    # 🔧 [전력수요 변경] (신규) 기준선과 비교 — LSTM 이 "어제와 같음"보다 나아야 쓸 의미가 있음
    n_test = len(y_test)
    all_y = [r["Demand"] for r in rows]
    for name, lag in [("어제와 같음", 1), ("지난주 같은 요일", 7)]:
        naive = all_y[len(all_y) - n_test - lag : len(all_y) - lag]
        print(f"  기준선 {name:9s} RMSE = {rmse(y_test, naive):.1f} GWh, MAPE = {mape(y_test, naive):.2f}%")

    # STEP 8. 모델 저장
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")
    if score > RMSE_GATE:
        print(
            f"※ 참고: 이 RMSE는 Day1 로컬 모델이며 배포 게이트({RMSE_GATE:.0f} GWh) 통과 여부는 "
            "Day2에서 MLflow로 다시 정식 검증합니다."
        )


if __name__ == "__main__":
    main()
