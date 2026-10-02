"""
Day3 드리프트 감지 시뮬레이션.

핵심 프로세스:
    1) 기준 통계 산출   - 학습에 쓴 데이터(2021~2024)의 일 수요 평균·표준편차 계산
    2) 정상 입력 테스트 - 학습에 안 쓴 2025년 실제 42일 → RMSE 60 GWh 이내 확인 (베이스라인)
    3) 드리프트 데이터 생성 - 같은 실제 42일에 수요 점진 증가(0→+20%) 등 인위적 변화 적용
    4) 드리프트 데이터 주입 - 생성한 데이터를 서빙 서버에 요청으로 전송
    5) 결과 관찰       - RMSE 상승 -> 알림 로그 -> 재학습 트리거 -> 새 버전 승격 확인

🔧 [전력수요 변경] 요약
   - 랜덤워크 가짜 주가 → 업로드된 CSV 의 2025년 실제 일별 기록 (data/drift_scenarios.py)
   - 드리프트 = 변동성 3배 → 수요 구조 변화(ramp, 기본) / 일괄 증가(level) / 기온 센서 오류(temp_bias)
   - API 주소를 환경변수 API_URL 로 바꿀 수 있게 함, 시나리오·시드를 명령행 인자로 선택

사전 준비: uvicorn serving_app.main:app --port 8077 서버가 이미 떠 있어야 합니다.
실행: python scripts/simulate_drift.py [--scenario ramp|level|temp_bias] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.drift_scenarios import SCENARIOS, make_batch  # 🔧 [전력수요 변경]
from data.features import TEST_START, load_rows
from data.storage import latest_upload

API_URL = os.getenv("API_URL", "http://localhost:8077/predict/batch-test")  # 🔧 [전력수요 변경] 환경변수 허용


def compute_baseline_stats(csv_path: str | None = None) -> tuple[float, float]:
    """1단계: 학습에 사용한 데이터(2025 이전)의 일 수요 평균·표준편차."""
    rows = load_rows(csv_path or latest_upload())
    demands = np.array([r["Demand"] for r in rows if r["Date"] < TEST_START])  # 🔧 [전력수요 변경]
    return float(demands.mean()), float(demands.std())


def send_batch(records: list[dict], label: str) -> dict:
    # ✅ [TODO 4-2 정답] /predict/batch-test 로 배치를 일괄 전송하고 drift_check 결과를 출력
    resp = requests.post(API_URL, json={"records": records})  # 🔧 [전력수요 변경] prices → records
    resp.raise_for_status()
    result = resp.json()
    print(f"[{label}] {records[0]['date']} ~ {records[-1]['date']}  drift_check = {result['drift_check']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="ramp")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    mean, std = compute_baseline_stats()
    print(f"[1] 기준 통계(2021~2024): 일 수요 mean={mean:.1f} GWh, std={std:.1f} GWh")

    print("[2] 정상 입력 테스트 전송 (2025년 실제 42일)...")
    send_batch(make_batch("normal", seed=args.seed), label="normal")

    print(f"[3-4] 드리프트 입력 생성·주입 (scenario={args.scenario})...")
    send_batch(make_batch("drift", scenario=args.scenario, seed=args.seed), label="drift_injection")

    print("[5] 결과 확인: logs/aiops.log 또는 대시보드 '재학습 로그'에서 [WARN] → [INFO] → [OK] 순서를 확인하세요.")


if __name__ == "__main__":
    main()
