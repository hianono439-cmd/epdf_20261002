"""
[Day3] 드리프트 → 자동 재학습  —  serving_app/monitoring/retrain_trigger.py

■ 이 파일이 하는 일 (한 줄 요약)
   드리프트가 감지되면 최근 관측 데이터로 모델을 조금 더 학습(fine-tuning)시키고,
   시험을 통과하면 새 모델을 Production 으로 올립니다. 이 과정을 전부 로그로 남깁니다.

■ 전체 흐름
   드리프트 감지(RMSE > 60 GWh) → 경고 로그 → 최근 관측 데이터 가져오기 → fine-tuning
     → 시험 통과(RMSE ≤ 60 GWh 이고 기존 모델보다 좋음)? ─ 예   → 새 버전 Production 승격 + 성공 로그 + 모델 캐시 교체
                                                          └ 아니오 → 기존 Production 그대로 유지

■ 확인 방법 — logs/aiops.log (또는 대시보드 "재학습 로그")
     [WARN] drift detected - triggering retrain
     [INFO] retrain triggered (window=last_28_days)
     [OK] new_rmse=... - production promoted: Power_Demand_Predictor v2

■ 빈칸 정답 위치 : [빈칸 9] 데이터 범위   [빈칸 10] 학습 방식   [빈칸 11] 승격 여부

🔧 [전력수요 변경] 요약
   - 재학습 데이터: 업로드된 학습 CSV 의 마지막 행  →  서빙 중 쌓인 관측 데이터(data/observed/observed.csv)의 마지막 행
     (드리프트는 "새로 들어온 데이터"에서 생기므로, 그 데이터로 적응해야 RMSE 가 내려감)
   - 21 하드코딩 → WINDOW_SIZE(28) 상수
   - 승격 성공 시 model_loader.reset_cache() 로 서빙 모델 즉시 교체 (원본은 서버 재시작 전까지 옛 모델 유지)
   - 게이트 탈락 시에도 [FAIL] 로그를 남김 (원본은 실패 시 로그가 없어 원인 추적이 어려움)
"""
import logging

from serving_app.monitoring.drift_detector import is_drift
from serving_app.monitoring.request_log import log_drift  # 🆕 [데모 대시보드 추가] 드리프트 이력 기록

# "aiops" 이름의 기록장. main.py 가 이 기록장을 logs/aiops.log 파일에 연결해 두었습니다.
logger = logging.getLogger("aiops")


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    """
    받는 것  : 최근 예측 기록 [{"predicted": ..., "actual": ...}, ...]  (predict.py 가 넘겨줌)
    돌려줄 것:
      드리프트 없음 → {"status": "ok", "rmse": 42.1}
      재학습 함     → {"status": "retrain_triggered", "promoted": True/False, "rmse": ..., "drift_rmse": ...}
    """
    from serving_app.monitoring.drift_detector import compute_rmse, WINDOW_SIZE  # 🔧 [전력수요 변경]

    drift_rmse = compute_rmse(recent_predictions[-WINDOW_SIZE:])  # 🔧 [전력수요 변경] 응답에 현재 RMSE 표시
    if not is_drift(recent_predictions):
        if recent_predictions:
            log_drift(drift_rmse, "ok")  # 🆕 [데모 대시보드 추가] 대시보드 "드리프트 점수" 추이용
        return {"status": "ok", "rmse": drift_rmse}

    logger.warning(f"[WARN] drift detected - triggering retrain (rmse={drift_rmse:.1f} GWh)")

    # 함수 안에서 import 하는 이유: 파일끼리 서로를 import 하다 꼬이는 문제(순환 import)를 피하려고
    from data.features import load_rows, SEQ_LEN
    from data.storage import OBSERVED_PATH  # 🔧 [전력수요 변경] latest_upload → OBSERVED_PATH
    from serving_app import model_loader  # 🔧 [전력수요 변경]
    from serving_app.train_and_register import fine_tune

    logger.info(f"[INFO] retrain triggered (window=last_{WINDOW_SIZE}_days)")  # 🔧 [전력수요 변경] 21 → WINDOW_SIZE

    # ✅ [빈칸 9 정답] 정답 21일 + 창문(SEQ_LEN)일  (원본: [-(21 + SEQ_LEN):] = 41행)
    # 🔧 [전력수요 변경] 21 → WINDOW_SIZE(28), 데이터 원천 latest_upload() → OBSERVED_PATH  → 28 + 14 = 42행
    rows = load_rows(OBSERVED_PATH)[-(WINDOW_SIZE + SEQ_LEN):]

    # ✅ [빈칸 10 정답] Production 가중치를 이어받는 fine_tune (42행으로 스크래치 학습은 불안정)
    result = fine_tune(rows)

    # ✅ [빈칸 11 정답] 게이트를 통과해 실제로 Production 이 됐을 때만 성공 로그
    log_drift(drift_rmse, "retrain_triggered", result["promoted"])  # 🆕 [데모 대시보드 추가]
    if result["promoted"]:
        model_loader.reset_cache()  # 🔧 [전력수요 변경] 다음 요청부터 새 Production 으로 서빙
        logger.info(
            f"[OK] new_rmse={result['rmse']:.2f} - production promoted: {model_loader.MODEL_NAME} v{result['version']}"
        )
        return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"],
                "drift_rmse": drift_rmse, "champion_rmse": result["champion_rmse"]}
    # 🔧 [전력수요 변경] (신규) 탈락 로그
    logger.info(
        f"[FAIL] new_rmse={result['rmse']:.2f} (champion={result['champion_rmse']:.2f}) - gate not passed, keep current production"
    )
    return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"],
            "drift_rmse": drift_rmse, "champion_rmse": result["champion_rmse"]}
