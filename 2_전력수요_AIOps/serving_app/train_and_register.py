"""
Day2: MLflow로 전력수요 LSTM 모델을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

실습 시나리오:
    1) 2021~2024 데이터로 base 모델 학습(100 epoch) -> 2025 데이터로 RMSE 확인
    2) 게이트(75 GWh) 통과 시 Production으로 승격
    3) (Day3) 드리프트 감지 시 Production 가중치에서 warm-start -> 최근 4주 관측 데이터로
       10 epoch만 fine-tuning (처음부터 다시 학습하지 않음 - 28일로는 스크래치 학습이 불안정)

🔧 [전력수요 변경] 요약
   - HAICScaler → PowerScaler, HAIC_Predictor → Power_Demand_Predictor
   - RMSE_GATE $4.00 → 75 GWh  (근거: 2025년이 아닌 2024년 "검증" 데이터로 정함 — scripts/calibrate_on_validation.py.
     2021~2023 학습 모델의 2024년 28일 구간별 RMSE 최댓값 71.1 GWh → 5 단위 올림. 2025년은 최종 평가에만 사용)
   - 분할: 날짜 기준 (2021~2024 학습 / 2025 평가), MAPE 지표 추가 기록
   - fine_tune(): "챔피언-챌린저" 비교 추가 — 같은 평가 구간에서 기존 Production 보다 나을 때만 승격
     (원본은 게이트만 통과하면 기존보다 나빠도 승격될 수 있었음)

실행:
    (대시보드에서 전력수요 CSV를 먼저 업로드하세요 - data/sample_power_daily.csv가 예시입니다)
    python scripts/train_baseline_v1.py     # 최초 1회 (scaler.pkl 생성)
    python serving_app/train_and_register.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import load_rows, build_sequences, train_test_split, PowerScaler, SEQ_LEN  # 🔧 [전력수요 변경]
from data.storage import latest_upload
from serving_app.lstm_model import build_model

# 시드 고정: LSTM 가중치 초기화가 랜덤이라 시드 없이는 실행마다 RMSE가 흔들립니다.
SEED = 42
keras.utils.set_random_seed(SEED)

RMSE_GATE = 75.0  # 🔧 [전력수요 변경] $4.00 → 75 GWh (2024 검증으로 정함)
MODEL_NAME = "Power_Demand_Predictor"  # 🔧 [전력수요 변경] HAIC_Predictor → Power_Demand_Predictor
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4  # base 학습(1e-3)보다 낮은 학습률로 살짝만 갱신


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:  # 🔧 [전력수요 변경] (신규)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)


def _prepare(rows: list[dict], scaler: PowerScaler):
    X, y, dates = build_sequences(rows, scaler)  # 🔧 [전력수요 변경] dates 추가
    X_train, y_train, X_test, y_test = train_test_split(X, y, dates)  # 🔧 [전력수요 변경] 날짜 기준 분할
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_demand(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _predict(model, X, scaler: PowerScaler) -> list[float]:  # 🔧 [전력수요 변경] (신규) 반복 코드 정리
    return [scaler.inverse_demand(p) for p in model.predict(X, verbose=0).flatten()]


def _register_if_gate_passed(model, run_id: str, score: float, champion_score: float | None = None) -> dict:
    result = {"run_id": run_id, "rmse": score, "promoted": False}
    # 🔧 [전력수요 변경] champion_score 가 있으면(재학습) 기존 Production 보다 좋아야 승격
    beats_champion = champion_score is None or score < champion_score
    if score <= RMSE_GATE and beats_champion:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Production", archive_existing_versions=True
        )  # 🔧 [전력수요 변경] archive_existing_versions=True: 이전 Production 은 Archived 로 내려 1개만 유지
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.1f} GWh -> {MODEL_NAME} v{v.version} promoted to Production")
    elif not beats_champion:
        print(f"[GATE FAILED] rmse={score:.1f} >= 기존 Production {champion_score:.1f} -> 배포 차단, 기존 Production 유지")
    else:
        print(f"[GATE FAILED] rmse={score:.1f} > {RMSE_GATE} -> 배포 차단, 기존 Production 유지")
    return result


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용합니다."""
    if rows is None:
        rows = load_rows(csv_path or latest_upload())
    scaler = PowerScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

        preds = _predict(model, X_test, scaler)
        score = rmse(y_test, preds)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_param("seq_len", SEQ_LEN)  # 🔧 [전력수요 변경]
        mlflow.log_param("n_train", len(X_train))  # 🔧 [전력수요 변경]
        mlflow.log_param("n_test", len(X_test))  # 🔧 [전력수요 변경]
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("mape", mape(y_test, preds))  # 🔧 [전력수요 변경]
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


def fine_tune(rows: list[dict]) -> dict:
    """
    Day3: 현재 Production 모델 가중치에서 이어서(warm start), 넘겨받은 rows(최근 관측 데이터)로
    짧게 fine-tuning합니다. rows가 적을 때(예: 최근 4주)도 스크래치 학습보다 훨씬 안정적입니다.
    """
    scaler = PowerScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    # 🔧 [전력수요 변경] 챔피언(기존 Production)의 같은 평가 구간 RMSE 를 먼저 계산
    champion = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    champion_score = rmse(y_test, _predict(champion, X_test, scaler))

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)

        preds = _predict(model, X_test, scaler)
        score = rmse(y_test, preds)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_metric("rmse", score)
        mlflow.log_metric("mape", mape(y_test, preds))  # 🔧 [전력수요 변경]
        mlflow.log_metric("champion_rmse", champion_score)  # 🔧 [전력수요 변경]
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        result = _register_if_gate_passed(model, mlflow.active_run().info.run_id, score, champion_score)
        result["champion_rmse"] = champion_score  # 🔧 [전력수요 변경]
        return result


if __name__ == "__main__":
    train_and_register()
