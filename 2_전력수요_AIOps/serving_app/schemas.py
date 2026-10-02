"""
Day1: FastAPI 요청/응답 Pydantic 스키마.

🔧 [전력수요 변경] 원본은 "최근 20거래일 (종가, 거래량)" 시퀀스를 받았습니다.
   전력수요는 과거 SEQ_LEN(14)일의 (날짜, 수요, 기온, 공휴일) + 예측 대상일의 (날짜, 예보 기온, 공휴일)을 받습니다.
   요일은 날짜에서 서버가 계산하므로 보내지 않아도 됩니다.
"""
from datetime import date as Date

from pydantic import BaseModel, Field

from data.features import SEQ_LEN


# 🔧 [전력수요 변경] DailyPoint(close, volume) → DayRecord / TargetDay
class TargetDay(BaseModel):
    date: Date = Field(..., description="날짜 (YYYY-MM-DD)")
    temp: float = Field(..., ge=-40, le=50, description="전국 인구가중 일평균기온(℃). 예측 대상일은 기상 예보값")
    holiday: int = Field(0, ge=0, le=1, description="공휴일·명절·대체공휴일이면 1 (주말은 날짜로 자동 판단)")


class DayRecord(TargetDay):
    demand: float = Field(..., gt=0, description="일 전력수요 (GWh = 24개 시간별 수요 MW 합 / 1000)")


class PredictRequest(BaseModel):
    history: list[DayRecord] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 날 -> 가장 최근 날 순서의 최근 {SEQ_LEN}일 기록",
    )
    target: TargetDay = Field(..., description="예측할 날(history 마지막 날의 다음 날)")


class PredictResponse(BaseModel):
    target_date: Date
    predicted_demand_gwh: float
    model_version: str


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션에서 사용 (scripts/simulate_drift.py 참고)
    # 🔧 [전력수요 변경] prices(종가 목록) → records(일별 관측 기록 목록)
    # SEQ_LEN + N 개의 연속된 날을 보내면, 서버가 슬라이딩 윈도우로 잘라 N건을 연속 예측한다.
    records: list[DayRecord] = Field(..., min_length=SEQ_LEN + 1)


class BatchTestResponse(BaseModel):
    predictions: list[float]
    actuals: list[float]  # 🔧 [전력수요 변경] 대시보드가 RMSE를 다시 계산할 수 있도록 실제값도 돌려줌
    drift_check: dict
