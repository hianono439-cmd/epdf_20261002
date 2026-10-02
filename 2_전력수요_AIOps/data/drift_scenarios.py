"""
🔧 [전력수요 변경] (신규 파일) 정상/드리프트 배치 생성 — scripts/simulate_drift.py 와 대시보드(/data/sample-batch)가 공용.

원본(HAIC)은 로그수익률 랜덤워크로 "가짜 주가"를 만들었습니다. 전력수요는 요일·휴일·기온 패턴이
뚜렷해서 랜덤워크로 흉내 내면 정상 배치조차 실제와 달라집니다. 그래서
  - 정상 배치  : 업로드된 CSV 중 학습에 안 쓴 2025년 실제 연속 42일을 그대로 사용 (운영 중 들어오는 새 데이터 역할)
  - 드리프트 배치: 같은 실제 42일에 인위적인 변화를 덧씌움 (apply_scenario)
       ramp      (기본) 수요가 42일 동안 0% → +RAMP_MAX 까지 점진적으로 증가
                        (예: 대규모 데이터센터·전기화로 인한 구조적 수요 증가)
       level     전 기간 수요 +LEVEL_UP 일괄 증가
       temp_bias 기온 관측값이 +TEMP_BIAS℃ 틀어짐 (관측소 센서 이상 같은 "입력 데이터" 드리프트)

변화 크기는 2025년이 아니라 2024년 검증 데이터로 정했습니다 (scripts/calibrate_on_validation.py,
결과: docs/검증기간_보정결과.json). ramp 는 검증 구간 49개 전부에서 기준값(60 GWh)을 넘는 가장 작은 크기(+20%)입니다.
(level 은 92%, temp_bias 는 61% 감지 — 기본 시나리오가 아닌 참고용)
➕ [명절·징검다리 추가] 피처 9개·기준값 60 기준으로 다시 계산 (이전: 기준값 75 에서 +25%)
"""
import random

from data.features import SEQ_LEN, TEST_START, load_rows
from data.storage import latest_upload

BATCH_N = SEQ_LEN + 28  # 14 + 28(WINDOW_SIZE) = 42일 → 배치 하나로 판정 윈도우 28건이 채워짐
SCENARIOS = ("ramp", "level", "temp_bias")
RAMP_MAX = 0.20   # 검증(2024)에서 정함 — calibrate_on_validation.py 참고  (➕ [명절·징검다리 추가] 0.25 → 0.20)
LEVEL_UP = 0.15
TEMP_BIAS = 8.0


def apply_scenario(rows: list[dict], scenario: str, ramp_max: float = RAMP_MAX) -> list[dict]:
    """rows(복사본)에 드리프트를 덧씌운다. calibrate_on_validation.py 도 이 함수를 그대로 씀."""
    for i, r in enumerate(rows):
        if scenario == "ramp":
            r["Demand"] *= 1 + ramp_max * i / (len(rows) - 1)
        elif scenario == "level":
            r["Demand"] *= 1 + LEVEL_UP
        elif scenario == "temp_bias":
            r["Temp"] += TEMP_BIAS
        else:
            raise ValueError(f"scenario 는 {SCENARIOS} 중 하나여야 합니다")
    return rows


def _pick_slice(rows: list[dict], n: int, seed: int | None) -> list[dict]:
    pool = [i for i, r in enumerate(rows) if r["Date"] >= TEST_START and i + n <= len(rows)]
    if not pool:  # 2025년 데이터가 없는 CSV 면 마지막 구간 사용
        pool = [len(rows) - n]
    start = random.Random(seed).choice(pool)
    return [dict(r) for r in rows[start : start + n]]


def make_batch(kind: str = "normal", scenario: str = "ramp", seed: int | None = None, csv_path: str | None = None) -> list[dict]:
    """kind = normal | drift.  반환: [{"date", "demand", "temp", "holiday"}, ...] (API 요청 형식)"""
    rows = _pick_slice(load_rows(csv_path or latest_upload()), BATCH_N, seed)
    if kind == "drift":
        apply_scenario(rows, scenario)
    return [
        {"date": r["Date"], "demand": round(r["Demand"], 1), "temp": round(r["Temp"], 2), "holiday": r["Holiday"],
         "myeongjeol": r["Myeongjeol"], "bridge": r["Bridge"]}  # ➕ [명절·징검다리 추가]
        for r in rows
    ]
