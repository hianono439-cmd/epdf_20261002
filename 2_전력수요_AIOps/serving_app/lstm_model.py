"""
다음날 전력수요 예측용 LSTM 아키텍처 (Day1 baseline과 Day2 MLflow 학습이 공유).

🔧 [전력수요 변경] 층 구성은 교수님 원본(LSTM 32 → 32 → 16 + Dense 16 + Dense 1)을 그대로 두고,
   입력 모양만 (20, 2) → (SEQ_LEN=14, N_FEATURES=7) 로 바꿨습니다.
   크기는 2025년(테스트)이 아니라 2024년 검증 데이터로 골랐습니다 (scripts/calibrate_on_validation.py):
   2021~2023 학습 → 2024 검증 RMSE  (32,32,16) 40.4 GWh  vs  (64,32,16) 41.7 GWh → 원본 크기 유지
   (➕ [명절·징검다리 추가] 9개 피처 기준. 7개 피처일 때도 43.5 vs 44.1 로 같은 결론)
"""
from tensorflow import keras

from data.features import SEQ_LEN, N_FEATURES  # 🔧 [전력수요 변경] N_FEATURES 를 features.py 한 곳에서 관리


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
