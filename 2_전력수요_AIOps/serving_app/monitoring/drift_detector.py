"""
[Day3] 드리프트 감지  —  serving_app/monitoring/drift_detector.py

■ 이 파일이 하는 일 (한 줄 요약)
   "최근 모델이 평균 몇 GWh씩 틀리고 있는지(RMSE)"를 계산해서,
   60 GWh보다 많이 틀리면 "데이터가 달라졌다(드리프트)"고 판단합니다.

■ 판단 기준
   최근 28건(4주)의 RMSE > 60 GWh  →  드리프트!

■ 빈칸 정답 위치 : [빈칸 7] compute_rmse   [빈칸 8] is_drift

🔧 [전력수요 변경] 요약
   - RMSE_THRESHOLD : $4.00 → 60 GWh
       근거: 2025년(테스트)은 쓰지 않고 2024년 "검증" 데이터로 정했습니다 (scripts/calibrate_on_validation.py).
             2021~2023으로 학습한 모델의 2024년 28일 구간별 RMSE: 중앙값 39.9 / 95% 52.6 / 최댓값 55.3 GWh
             (최댓값은 추석·개천절 연휴가 낀 10월 초) → 5 단위로 올려 60 을 넘으면 "평소와 다른" 상태
             ➕ [명절·징검다리 추가] 명절·징검다리 피처로 오차가 줄어 75 → 60 (이전 7개 피처: 중앙값 42.5 / 최댓값 71.1 → 75)
   - WINDOW_SIZE    : 21거래일 → 28일 (4주 — 요일이 정확히 4번씩 들어가 요일 편향이 없음)
"""
RMSE_THRESHOLD = 60.0  # 🔧 [전력수요 변경] 4.00 달러 → GWh 기준 (2024 검증으로 정함)  ➕ [명절·징검다리 추가] 75 → 60
WINDOW_SIZE = 28       # 🔧 [전력수요 변경] 21 → 28 (4주)


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    받는 것  : [{"predicted": 1600.0, "actual": 1640.0}, {"predicted": 1600.0, "actual": 1560.0}, ...]
    돌려줄 것: RMSE (숫자 1개, "평균 몇 GWh 틀렸나").  빈 목록이면 0.0
    """
    import math

    # 기록이 하나도 없으면 0.0 (빈 목록이면 0으로 나누기 에러가 나기 때문)
    if not recent_predictions:
        return 0.0

    # ✅ [빈칸 7 정답] ① 오차 = 실제 - 예측,  ③ 평균 = 합 ÷ 개수
    errors_sq = [(p["actual"] - p["predicted"]) ** 2 for p in recent_predictions]
    return math.sqrt(sum(errors_sq) / len(errors_sq))


def is_drift(recent_predictions: list[dict]) -> bool:
    """
    드리프트인지 True/False 로 판단합니다.
    흐름: (데이터 충분한가?) → 최근 28건만 골라서 → RMSE 계산 → 기준(60 GWh)보다 크면 드리프트
    """
    # ✅ [빈칸 8 정답] 최근 기록이 WINDOW_SIZE 건 미만이면 아직 판단하지 않음
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
