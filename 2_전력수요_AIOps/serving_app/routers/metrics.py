"""
🆕 [데모 대시보드 추가] (신규 파일) 운영 대시보드용 조회 API.

교수님 데모 대시보드(4탭: Dashboard / Simulation / Datasets / System)가 보여 주던 값을
이미 존재하는 데이터(로그 파일, MLflow Registry)를 읽어서만 만듭니다. 가짜 값은 만들지 않습니다.

  GET /metrics/summary?window=5m|1h|6h|24h  운영 지표 요약 5칸 (요청 수·응답시간·성공률·모델 RMSE·드리프트 점수)
  GET /metrics/versions                     재학습 이력 (MLflow Registry 의 Power_Demand_Predictor 전체 버전)
  GET /metrics/current-model                현재 운영(Production) 모델 카드
  GET /metrics/alerts                       최근 알람 (logs/aiops.log 의 [WARN]/[INFO]/[OK]/[FAIL] 줄)
  GET /metrics/system                       System 탭 (로딩 모드, 모델 소스, 라이브러리 버전, 데이터·로그 파일 상태)
"""
import os
import platform
import re
import time
from datetime import datetime
from urllib.parse import unquote

import mlflow  # 서버 시작 때 미리 import — 대시보드가 여러 API 를 동시에 부를 때 각 스레드가 mlflow 를
from mlflow.tracking import MlflowClient  # 처음 import 하면서 꼬이는 문제(순환 import 오류)를 막기 위함
from fastapi import APIRouter, HTTPException

from data.storage import OBSERVED_PATH, UPLOAD_DIR
from scripts.aggregate_metrics import WINDOWS, aggregate
from serving_app import model_loader
from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE
from serving_app.monitoring.request_log import DRIFT_LOG, LOG_DIR, read_jsonl

router = APIRouter(prefix="/metrics")
_STARTED_AT = time.time()

# MLflow 조회는 수백 ms 걸릴 수 있어, 대시보드가 5초마다 새로고침해도 서버가 느려지지 않게 짧게 캐시
_CACHE_SEC = 3.0
_versions_cache: dict = {"t": 0.0, "data": None}


def _versions() -> list[dict]:
    if _versions_cache["data"] is not None and time.time() - _versions_cache["t"] < _CACHE_SEC:
        return _versions_cache["data"]
    client = MlflowClient()
    try:
        mvs = client.search_model_versions(f"name='{model_loader.MODEL_NAME}'")
    except Exception:
        mvs = []
    out = []
    for v in mvs:
        params, metrics = {}, {}
        if v.run_id:
            try:
                run = client.get_run(v.run_id)
                params, metrics = run.data.params, run.data.metrics
            except Exception:
                pass
        out.append({
            "version": int(v.version),
            "registered_at": v.creation_timestamp / 1000,
            "mode": params.get("mode", "?"),  # scratch(Day2 base 학습) | fine-tune(Day3 재학습)
            "rmse": metrics.get("rmse"),
            "mape": metrics.get("mape"),
            "champion_rmse": metrics.get("champion_rmse"),
            "n_rows": params.get("n_rows") or params.get("n_train"),
            "stage": v.current_stage or "None",
            "run_id": v.run_id,
        })
    out.sort(key=lambda x: x["version"], reverse=True)
    _versions_cache.update(t=time.time(), data=out)
    return out


@router.get("/summary")
def summary(window: str = "5m"):
    if window not in WINDOWS:
        raise HTTPException(400, f"window 는 {list(WINDOWS)} 중 하나")
    svc = aggregate(window)

    versions = _versions()
    prod = next((v for v in versions if v["stage"] == "Production"), None)
    # 모델 성능 미니 그래프: 등록 버전 순서대로 RMSE (오래된 → 최신, 최근 12개)
    rmse_trend = [{"version": v["version"], "rmse": v["rmse"], "mode": v["mode"]}
                  for v in sorted(versions, key=lambda x: x["version"]) if v["rmse"] is not None][-12:]

    # 드리프트 점수: 지금 메모리에 쌓인 최근 28건 예측의 RMSE (predict.py 의 recent_predictions)
    from serving_app.monitoring.drift_detector import compute_rmse
    from serving_app.routers.predict import recent_predictions

    drift_now = compute_rmse(recent_predictions[-WINDOW_SIZE:]) if recent_predictions else None
    drift_hist = read_jsonl(DRIFT_LOG)[-12:]
    return {
        "window": window,
        "service": svc,
        "model": {"production_version": prod["version"] if prod else None,
                  "rmse": prod["rmse"] if prod else None, "mode": prod["mode"] if prod else None,
                  "trend": rmse_trend},
        "drift": {"score": drift_now, "threshold": RMSE_THRESHOLD, "window_size": WINDOW_SIZE,
                  "n_recent": len(recent_predictions), "history": drift_hist},
    }


@router.get("/versions")
def versions():
    return _versions()


@router.get("/current-model")
def current_model():
    versions = _versions()
    prod = next((v for v in versions if v["stage"] == "Production"), None)
    cached = model_loader._model_cache
    return {
        "name": model_loader.MODEL_NAME,
        "production": prod,
        "serving_version": cached.version if cached else None,  # 서버 메모리에 올라와 있는 모델
        "model_loaded": cached is not None,
        "gate_rmse": RMSE_THRESHOLD,
        "healthy": prod is not None or os.getenv("MODEL_SOURCE", "local") != "mlflow",
    }


_ALERT_RE = re.compile(r"^(\S+ \S+) \[(\w+)\] (.*)$")


@router.get("/alerts")
def alerts(limit: int = 30):
    path = os.path.join(LOG_DIR, "aiops.log")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for line in reversed(lines):
        m = _ALERT_RE.match(line)
        if not m:
            continue
        ts_str, level, msg = m.groups()
        try:
            ts = datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            ts = None
        kind = "ok" if "[OK]" in msg else "fail" if "[FAIL]" in msg else "warn" if level == "WARNING" else "info"
        out.append({"ts": ts, "level": kind, "message": msg})
        if len(out) >= limit:
            break
    return out


@router.get("/system")
def system():
    import fastapi
    import numpy

    def _files(d):
        if not os.path.isdir(d):
            return []
        return [{"name": n, "size": os.path.getsize(os.path.join(d, n))}
                for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n))]

    from importlib.metadata import version as pkg_version

    try:
        tf_ver = pkg_version("tensorflow")  # import 하지 않고 설치 정보만 읽음 (tensorflow import 는 수 초 걸림)
    except Exception:
        tf_ver = "?"
    observed_rows = 0
    if os.path.exists(OBSERVED_PATH):
        with open(OBSERVED_PATH, encoding="utf-8") as f:
            observed_rows = max(0, sum(1 for _ in f) - 1)
    return {
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "mlflow_tracking_uri": unquote(mlflow.get_tracking_uri()),  # 한글 경로가 %EC.. 로 보이지 않게
        "uptime_sec": round(time.time() - _STARTED_AT),
        "python": platform.python_version(),
        "versions": {"fastapi": fastapi.__version__, "mlflow": mlflow.__version__,
                     "tensorflow": tf_ver, "numpy": numpy.__version__},
        "uploads": _files(UPLOAD_DIR),
        "observed_rows": observed_rows,
        "logs": _files(LOG_DIR),
        "drift_rule": {"threshold_gwh": RMSE_THRESHOLD, "window_days": WINDOW_SIZE},
    }
