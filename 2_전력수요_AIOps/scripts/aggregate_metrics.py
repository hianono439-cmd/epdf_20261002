"""
🆕 [데모 대시보드 추가] (신규 파일) requests.log 를 시간 구간별로 집계.

교수님 Dockerfile 주석("routers/metrics.py가 scripts.aggregate_metrics를 import함")에 남아 있던
데모 패키지 구조를 따라, 집계 로직은 scripts/ 에 두고 routers/metrics.py 가 import 해서 씁니다.
터미널에서 단독 실행도 됩니다:  python scripts/aggregate_metrics.py 1h

집계 항목 (window 안의 요청만)
  requests      : 요청 수
  avg_latency_ms: 평균 응답시간
  p95_latency_ms: 느린 쪽 5% 경계 응답시간 (평균만 보면 가끔 매우 느린 요청이 묻힘)
  success_rate  : 상태코드 2xx·3xx 비율 (%)
  series        : window 를 12칸으로 나눈 칸별 요청 수·평균 응답시간·성공률 (대시보드 미니 그래프용)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_app.monitoring.request_log import REQUESTS_LOG, read_jsonl

WINDOWS = {"5m": 5 * 60, "1h": 60 * 60, "6h": 6 * 60 * 60, "24h": 24 * 60 * 60}
N_BUCKETS = 12


def aggregate(window: str = "5m", now: float | None = None, path: str = REQUESTS_LOG) -> dict:
    if window not in WINDOWS:
        raise ValueError(f"window 는 {list(WINDOWS)} 중 하나")
    now = now or time.time()
    span = WINDOWS[window]
    start = now - span
    rows = [r for r in read_jsonl(path) if r.get("ts", 0) >= start]

    lat = sorted(r["latency_ms"] for r in rows)
    ok = [r for r in rows if 200 <= r["status"] < 400]

    width = span / N_BUCKETS
    buckets = [{"t": start + (i + 1) * width, "requests": 0, "lat_sum": 0.0, "ok": 0} for i in range(N_BUCKETS)]
    for r in rows:
        b = buckets[min(int((r["ts"] - start) // width), N_BUCKETS - 1)]
        b["requests"] += 1
        b["lat_sum"] += r["latency_ms"]
        b["ok"] += 1 if 200 <= r["status"] < 400 else 0
    series = [
        {
            "t": b["t"],
            "requests": b["requests"],
            "avg_latency_ms": round(b["lat_sum"] / b["requests"], 1) if b["requests"] else None,
            "success_rate": round(b["ok"] / b["requests"] * 100, 1) if b["requests"] else None,
        }
        for b in buckets
    ]
    return {
        "window": window,
        "requests": len(rows),
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
        "p95_latency_ms": round(lat[max(0, math.ceil(len(lat) * 0.95) - 1)], 1) if lat else 0.0,
        "success_rate": round(len(ok) / len(rows) * 100, 1) if rows else 100.0,
        "series": series,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(aggregate(sys.argv[1] if len(sys.argv) > 1 else "5m"), ensure_ascii=False, indent=2))
