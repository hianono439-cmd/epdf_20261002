"""
[Day1 → Day3] 예측 API  —  serving_app/routers/predict.py

■ 이 파일이 하는 일 (한 줄 요약)
   외부 요청을 받아 모델에게 전달하고, 결과를 돌려주는 "창구"입니다.
   계산은 직접 하지 않고, 모델(model_loader)과 감시 도구(retrain_trigger)에게 맡깁니다.

■ 엔드포인트
   [Day1] POST /predict             : 14일치 기록 + 예측일 정보 → 다음날 전력수요 1개
   [Day3] POST /predict/batch-test  : 긴 일별 기록 → 여러 번 예측 → 드리프트 검사

■ 빈칸 정답 위치 : [빈칸 6]  (batch_test 의 슬라이딩 윈도우)

🔧 [전력수요 변경] 요약
   - 요청 형식: 종가 시퀀스/종가 목록 → 일별 기록(날짜·수요·기온·공휴일) + 예측 대상일
   - batch_test 가 들어온 실제 관측값을 data/observed/observed.csv 에 저장 (재학습 데이터 원천)
   - 최근 기록 유지 길이 21 하드코딩 → drift_detector.WINDOW_SIZE(28) 상수 사용
"""
from fastapi import APIRouter

from data.features import SEQ_LEN  # = 14
from data.storage import append_observed  # 🔧 [전력수요 변경]
from serving_app import model_loader
from serving_app.monitoring.drift_detector import WINDOW_SIZE  # 🔧 [전력수요 변경]
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse

router = APIRouter()

# (Day3) 최근 예측 기록을 모아 두는 목록.  예: [{"predicted": 1612.3, "actual": 1598.0}, ...]
#        드리프트 판단은 "최근 28건"(drift_detector.py 의 WINDOW_SIZE)만 보므로 28개까지만 유지합니다.
recent_predictions: list[dict] = []


# 🔧 [전력수요 변경] (신규) API 요청(pydantic, 소문자 키) → features.py 가 쓰는 행 dict(대문자 키)
def _to_row(r) -> dict:
    row = {"Date": r.date.isoformat(), "Temp": r.temp, "Holiday": r.holiday}
    if hasattr(r, "demand"):
        row["Demand"] = r.demand
    return row


# 🔧 [전력수요 변경] SIMULATED_VOLUME(거래량 고정값) 삭제 — 시뮬레이션도 실제 기온·공휴일을 함께 보냄


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    [Day1] 다음날 전력수요 예측
    받는 것  : {"history": [{"date": "2025-01-01", "demand": 1500.2, "temp": -2.1, "holiday": 1}, ... 14개],
                "target":  {"date": "2025-01-15", "temp": -3.0, "holiday": 0}}
    돌려줄 것: {"target_date": "2025-01-15", "predicted_demand_gwh": 1712.4, "model_version": "production-v1"}
    """
    model = model_loader.get_model()
    history = [_to_row(r) for r in req.history]
    target = _to_row(req.target)
    predicted = model.predict_one(history, target)
    return PredictResponse(
        target_date=req.target.date, predicted_demand_gwh=round(predicted, 1), model_version=model.version
    )


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    [Day3] 드리프트 시뮬레이션
    받는 것  : {"records": [{"date":..., "demand":..., "temp":..., "holiday":...}, ... 42개]}
    돌려줄 것: {"predictions": [예측 28개], "actuals": [실제 28개], "drift_check": {...}}

    ■ 슬라이딩 윈도우 (14칸짜리 창문을 한 칸씩 밀기)
        i=0 : [d0  ~ d13] + d14 의 기온·휴일 → 예측   vs  실제 d14 수요
        i=1 : [d1  ~ d14] + d15 의 기온·휴일 → 예측   vs  실제 d15 수요
        ...
        → 총 42 - 14 = 28번 예측 = 드리프트 판단에 필요한 28건이 딱 채워집니다.
    """
    model = model_loader.get_model()
    predictions: list[float] = []
    actuals: list[float] = []

    rows = [_to_row(r) for r in req.records]  # 🔧 [전력수요 변경] prices → rows
    append_observed(rows)  # 🔧 [전력수요 변경] 관측 데이터 저장 (재학습 때 사용)

    for i in range(len(rows) - SEQ_LEN):
        # ✅ [빈칸 6 정답] 창문 = i ~ i+SEQ_LEN-1 (끝 인덱스는 포함 안 됨), 실제값 = 창문 바로 다음 날
        window = rows[i : i + SEQ_LEN]
        target = rows[i + SEQ_LEN]  # 🔧 [전력수요 변경] 다음날의 기온·휴일은 예측 입력으로, 수요는 정답으로
        pred = model.predict_one(window, {k: target[k] for k in ("Date", "Temp", "Holiday")})
        actual = target["Demand"]
        predictions.append(pred)
        actuals.append(actual)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 WINDOW_SIZE 건만 남기기 — 오래된 기록까지 섞이면 "지금" 상태를 판단할 수 없습니다.
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]  # 🔧 [전력수요 변경] -21 하드코딩 → WINDOW_SIZE

    # 드리프트 판단·재학습은 retrain_trigger.py 가 합니다. 여기서는 넘겨주기만!
    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(predictions=predictions, actuals=actuals, drift_check=drift_check)
