"""
[Day1 → Day2] 모델 불러오기  —  serving_app/model_loader.py

■ 이 파일이 하는 일 (한 줄 요약)
   서버가 예측에 쓸 모델을 "어디서, 언제" 불러올지 정하고, 예측 한 건을 수행합니다.
   다른 파일(predict.py, health.py)은 get_model() 만 부르면 되고, 모델이 어디서 왔는지 몰라도 됩니다.

■ 핵심 개념
   1) 학습 때와 똑같이 전처리해야 한다
        모델은 0~1 값으로 학습했습니다. 서빙할 때도 입력을 0~1로 바꾸고, 출력은 GWh로 되돌려야 합니다.
   2) Lazy vs Eager
        Eager : 서버가 켜질 때 모델을 바로 불러옴 → 서버 시작은 느리지만 첫 요청이 빠름
        Lazy  : 첫 /predict 요청이 올 때 불러옴  → 서버 시작은 빠르지만 첫 요청이 느림
   3) 모델은 바뀌어도 스케일러는 안 바뀐다
        MODEL_SOURCE 가 local 이든 mlflow 든, 스케일러는 항상 로컬 scaler.pkl 을 씁니다.

■ 환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local(기본, Day1) | mlflow(Day2~)

■ 빈칸 정답 위치 : [빈칸 2] [빈칸 3] predict_one()   [빈칸 4] get_model()   [빈칸 5] _load_from_mlflow()

🔧 [전력수요 변경] 요약
   - HAICScaler → PowerScaler, 모델 파일 haic_v1.keras → power_v1.keras, 레지스트리 이름 → Power_Demand_Predictor
   - predict_one(sequence) → predict_one(history, target): 예측 대상일의 기온·휴일을 함께 받음
   - 응답 버전에 실제 레지스트리 버전 번호를 표시 ("production" → "production-v2")
   - reset_cache() 추가: 재학습으로 새 버전이 승격되면 캐시를 비워 다음 요청부터 새 모델을 쓰게 함
     (원본은 승격 후에도 서버를 재시작하기 전까지 예전 모델이 캐시에 남아 있었음)
"""
import os
import time

from data.features import PowerScaler, window_to_input  # 🔧 [전력수요 변경] HAICScaler → PowerScaler

LOCAL_MODEL_PATH = "serving_app/models/power_v1.keras"  # 🔧 [전력수요 변경] haic_v1.keras → power_v1.keras
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "Power_Demand_Predictor"  # 🔧 [전력수요 변경] HAIC_Predictor → Power_Demand_Predictor
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"  # "models:/<모델 이름>/<단계>" 형식

_model_cache = None  # 한 번 불러온 모델을 담아 두는 상자 (처음엔 비어 있음 = None)


class LoadedModel:
    """
    모델 + 스케일러 + 버전을 한 묶음으로 포장한 상자.
    local 모델이든 MLflow 모델이든 이 상자에 담으면 똑같은 방법(predict_one)으로 쓸 수 있습니다.
    """

    def __init__(self, keras_model, scaler: PowerScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, history: list[dict], target: dict) -> float:
        """
        🔧 [전력수요 변경] 14일치 기록 + 예측 대상일 정보로 다음날 수요 1개를 예측합니다.
        받는 것  : history = [{"Date": "2025-01-01", "Demand": 1500.2, "Temp": -2.1, "Holiday": 1}, ... 14개]
                   target  = {"Date": "2025-01-15", "Temp": -3.0, "Holiday": 0}  (Demand 없음 = 맞힐 값)
        돌려줄 것: 다음날 예상 수요 (GWh)

        흐름:  [GWh·℃ 값 14일] → ① 0~1로 변환 → ② 입력 모양 맞추기 → ③ 예측(0~1) → ④ GWh로 복원
        """
        import numpy as np

        # ✅ [빈칸 2 정답] transform_point — 학습 때 build_sequences 가 쓴 것과 같은 변환
        # 🔧 [전력수요 변경] 하루치 변환을 window_to_input() 으로 감쌈 — 각 날의 수요 + "그다음 날"의 기온·휴일·요일.
        #    window_to_input 안에서 scaler.transform_point 를 그대로 호출하므로 빈칸 2 의 정답과 같은 도구를 씁니다.
        next_days = history[1:] + [target]
        scaled = window_to_input(history, next_days, self.scaler)

        # ② 입력 모양 맞추기 — 모델은 "문제 여러 개"를 받으므로 1개라도 [ ]로 감쌉니다. (1, 14, 7)
        x = np.array([scaled], dtype="float32")  # (1, SEQ_LEN, N_FEATURES)

        # ③ 예측 — 결과가 [[0.47]] 처럼 2겹이라 [0][0] 으로 숫자만 꺼냅니다. (아직 0~1 범위)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])

        # ✅ [빈칸 3 정답] inverse_close — 0~1 예측값을 원래 단위로 복원
        # 🔧 [전력수요 변경] inverse_close(달러) → inverse_demand(GWh)
        return self.scaler.inverse_demand(pred_scaled)


# ═══════════════════════════════ 어디서 불러올까? ═══════════════════════════════

def _load_from_local() -> LoadedModel:
    """Day1: 로컬 파일에서 모델과 스케일러를 불러와 상자에 담습니다."""
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = PowerScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _production_version() -> str:
    """🔧 [전력수요 변경] (신규) 현재 Production 단계 모델의 레지스트리 버전 번호를 조회."""
    from mlflow.tracking import MlflowClient

    try:
        versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
        return versions[0].version if versions else "?"
    except Exception:
        return "?"


def _load_from_mlflow() -> LoadedModel:
    """
    Day2: train_and_register.py 가 "Power_Demand_Predictor" 이름으로 등록하고 Production 으로 올려 둔 모델을
    MLflow Model Registry 에서 불러옵니다.
    """
    import mlflow.tensorflow

    # 모델 : MLflow 레지스트리에서 "Production" 단계 모델을 불러옵니다.
    #   버전 번호 대신 단계(Production)로 불러오므로, 재배포 때 서버 코드를 고칠 필요가 없습니다.
    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)

    # ✅ [빈칸 5 정답] 스케일러는 MLflow 가 아니라 항상 로컬 scaler.pkl (Day1에서 fit 한 그 기준)
    scaler = PowerScaler.load(SCALER_PATH)  # 🔧 [전력수요 변경] HAICScaler → PowerScaler
    # 🔧 [전력수요 변경] version="production" → "production-v{번호}" (재배포 후 새 버전이 응답하는지 확인용)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version=f"production-v{_production_version()}")


def _load_model() -> LoadedModel:
    """MODEL_SOURCE 값에 따라 로컬/MLflow 중 어디서 불러올지 고릅니다."""
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


# ═══════════════════════════ 언제 불러올까? (Eager / Lazy) ═══════════════════════════

def load_eager() -> LoadedModel:
    """Eager Loading: 서버가 켜질 때(main.py 의 startup) 바로 불러와 상자에 넣어 둡니다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """
    Lazy Loading: 첫 요청이 들어올 때만 불러오고, 이후에는 상자(_model_cache)에 있는 것을 재사용합니다.
    """
    global _model_cache

    # ✅ [빈칸 4 정답] 상자가 비어 있을 때(None)만 _load_model() 로 한 번 불러옴
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache


def reset_cache() -> None:
    """
    🔧 [전력수요 변경] (신규) 재학습으로 새 Production 이 승격되면 retrain_trigger.py 가 호출합니다.
    상자를 비워 두면 다음 요청 때 get_model() 이 새 Production 을 다시 불러옵니다 (무중단 재배포).
    """
    global _model_cache
    _model_cache = None
