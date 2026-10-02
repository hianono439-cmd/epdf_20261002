"""
🆕 [데모 대시보드 추가] (신규 파일) 요청 로그 · 드리프트 이력 기록.

교수님 데모 대시보드의 "운영 지표 요약"은 requests.log 를 실제로 읽어 시간 구간별로 집계한 값입니다.
이 파일은 그 재료를 남깁니다.
  - logs/requests.log      : API 요청 1건 = JSON 1줄  (시각, 메서드, 경로, 상태코드, 처리시간 ms)
                             → main.py 의 미들웨어가 모든 API 요청마다 log_request() 호출
  - logs/drift_history.log : 드리프트 판정 1회 = JSON 1줄 (시각, 최근 28일 RMSE, 판정 결과)
                             → retrain_trigger.check_and_trigger() 가 판정할 때마다 log_drift() 호출

JSON 한 줄씩 쌓는 형식(JSON Lines)이라 사람이 열어 봐도 읽기 쉽고, 집계 스크립트
(scripts/aggregate_metrics.py)는 줄 단위로 읽기만 하면 됩니다.
"""
import json
import os
import threading
import time

LOG_DIR = "logs"
REQUESTS_LOG = os.path.join(LOG_DIR, "requests.log")
DRIFT_LOG = os.path.join(LOG_DIR, "drift_history.log")

_lock = threading.Lock()  # 여러 요청이 동시에 같은 파일에 쓸 때 줄이 섞이지 않게


def _append(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_request(method: str, path: str, status: int, latency_ms: float) -> None:
    _append(REQUESTS_LOG, {"ts": time.time(), "method": method, "path": path,
                           "status": status, "latency_ms": round(latency_ms, 2)})


def log_drift(rmse: float, status: str, promoted: bool | None = None) -> None:
    _append(DRIFT_LOG, {"ts": time.time(), "rmse": round(rmse, 2), "status": status, "promoted": promoted})


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 쓰는 도중 끊긴 줄은 건너뜀
    return out
