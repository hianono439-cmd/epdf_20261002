"""
전력수요 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

Day1 baseline 학습(scripts/train_baseline_v1.py), Day2 MLflow 학습
(serving_app/train_and_register.py), Day3 fine-tuning 재학습
(monitoring/retrain_trigger.py), 서빙(model_loader.py)이 모두 이 모듈을 재사용합니다.
시퀀스 정의를 한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는
실무 사고를 방지할 수 있습니다.

🔧 [전력수요 변경] 원본(HAIC 주가) 대비 바뀐 점
  - 입력 CSV 컬럼: Date, Close, Volume  →  Date, Demand(GWh), Temp(℃), Holiday(0/1)
  - 피처: (종가, 거래량) 2개  →  7개
        [당일 수요, 다음날 기온, 다음날 난방도일, 다음날 냉방도일, 다음날 휴일 여부, 다음날 요일 sin, cos]
        → 마지막 시점에 "예측 대상일"의 기온·휴일·요일이 들어가도록 외생변수를 하루 앞당겨 붙임
          (주가와 달리 전력수요는 요일·휴일·기온이 결정적이고, 이 값들은 하루 전에 미리 알 수 있음)
  - SEQ_LEN: 20거래일  →  14일 (2주 = 같은 요일이 2번 들어가는 길이)
  - 학습/평가 분할: 앞 80% / 뒤 20% 비율  →  날짜 기준 (2021~2024 학습, 2025 평가)
  - 스케일러: HAICScaler(close, volume)  →  PowerScaler(demand, temp)
"""
import csv
import math
import pickle
from datetime import date

SEQ_LEN = 14  # 🔧 [전력수요 변경] 20거래일 → 14일 (2주)
N_FEATURES = 7  # 🔧 [전력수요 변경] (close, volume) 2개 → 7개 (위 docstring 참고)
TEST_START = "2025-01-01"  # 🔧 [전력수요 변경] 이 날짜 이후가 정답인 샘플은 평가(test)용

# 🔧 [전력수요 변경] 냉·난방도일 기준온도 — 팀플 데이터/preprocess.py 가 2021~2024(최종 학습 기간)만으로
#    회귀 R²가 가장 높은 값을 격자 탐색해서 고른 값 (2025 정보 미사용).
#    참고: 2021~2023만으로 고르면 13.5 / 19.5 로 거의 같음 (calibrate_on_validation.py)
HDD_BASE = 14.0
CDD_BASE = 19.0


def load_rows(csv_path: str = "data/sample_power_daily.csv") -> list[dict]:
    # 🔧 [전력수요 변경] Close/Volume → Demand/Temp/Holiday
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "Date": r["Date"],
                "Demand": float(r["Demand"]),
                "Temp": float(r["Temp"]),
                "Holiday": int(float(r["Holiday"])),
            }
            for r in reader
        ]
    return rows


# 🔧 [전력수요 변경] 날짜·기온에서 파생되는 "예측 시점에 미리 아는" 변수 계산 (신규 함수)
def exog_features(date_str: str, temp: float, holiday: int) -> list[float]:
    """하루치 외생변수 → [기온, 난방도일, 냉방도일, 휴일여부, 요일 sin, 요일 cos] (스케일 전)"""
    dow = date.fromisoformat(date_str).weekday()  # 0=월 ... 6=일
    offday = 1.0 if (holiday or dow >= 5) else 0.0  # 주말 또는 공휴일
    return [
        temp,
        max(0.0, HDD_BASE - temp),
        max(0.0, temp - CDD_BASE),
        offday,
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
    ]


class PowerScaler:
    """
    🔧 [전력수요 변경] HAICScaler → PowerScaler
    수요(GWh)와 기온(℃)을 [0, 1] 범위로 정규화하는 min-max 스케일러.
    냉·난방도일은 기온 범위로 나누고, 휴일·요일(sin/cos)은 이미 작은 값이라 그대로 둡니다.

    Day1에서 학습 기간(2021~2024) 데이터로만 한 번 fit한 뒤 serving_app/models/scaler.pkl로
    저장해두고, Day2 MLflow 학습과 Day3 fine-tuning, 서빙 모두 같은 스케일러를 재사용합니다.
    """

    def __init__(self):
        self.demand_min = self.demand_max = None
        self.temp_min = self.temp_max = None

    def fit(self, rows: list[dict]) -> "PowerScaler":
        demands = [r["Demand"] for r in rows]
        temps = [r["Temp"] for r in rows]
        self.demand_min, self.demand_max = min(demands), max(demands)
        self.temp_min, self.temp_max = min(temps), max(temps)
        return self

    def _scale(self, value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _unscale(self, value: float, lo: float, hi: float) -> float:
        return value * (hi - lo) + lo

    def transform_point(self, demand: float, next_day: dict) -> list[float]:
        """
        🔧 [전력수요 변경] (close, volume) → (당일 수요, 다음날 정보 dict)
        next_day = {"Date": "2025-01-02", "Temp": -1.3, "Holiday": 0}
        """
        temp, hdd, cdd, offday, dsin, dcos = exog_features(next_day["Date"], next_day["Temp"], next_day["Holiday"])
        t_range = self.temp_max - self.temp_min
        return [
            self._scale(demand, self.demand_min, self.demand_max),
            self._scale(temp, self.temp_min, self.temp_max),
            hdd / t_range,
            cdd / t_range,
            offday,
            dsin,
            dcos,
        ]

    def scale_demand(self, demand: float) -> float:
        """타깃(다음날 수요)을 학습용으로 정규화. (원본 scale_close)"""
        return self._scale(demand, self.demand_min, self.demand_max)

    def inverse_demand(self, scaled_demand: float) -> float:
        """모델이 뱉은 정규화된 예측값을 실제 GWh 단위로 되돌린다. (원본 inverse_close)"""
        return self._unscale(scaled_demand, self.demand_min, self.demand_max)

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "PowerScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def window_to_input(window: list[dict], next_days: list[dict], scaler: PowerScaler) -> list[list[float]]:
    """
    🔧 [전력수요 변경] (신규) 길이 SEQ_LEN 의 과거 기록 + 각 날의 "다음날 정보"를 (SEQ_LEN, 7) 입력으로.
    학습(build_sequences)과 서빙(model_loader.predict_one)이 이 함수 하나를 같이 씁니다.
    """
    return [scaler.transform_point(w["Demand"], n) for w, n in zip(window, next_days)]


def build_sequences(rows: list[dict], scaler: PowerScaler, seq_len: int = SEQ_LEN):
    """
    rows(시간순 일별 데이터)에서 (SEQ_LEN, 7) 크기의 정규화된 입력 시퀀스와
    다음날 수요(정규화 전 실값, GWh) 타깃을 만든다.

        i 번째 샘플: 입력 = rows[i : i+SEQ_LEN] 의 수요 + rows[i+1 : i+SEQ_LEN+1] 의 기온·휴일·요일
                     정답 = rows[i+SEQ_LEN] 의 수요

    반환: X (n_samples, seq_len, 7), y (n_samples,), dates (각 정답의 날짜)
    """
    X, y, dates = [], [], []
    for i in range(len(rows) - seq_len):
        X.append(window_to_input(rows[i : i + seq_len], rows[i + 1 : i + seq_len + 1], scaler))
        y.append(rows[i + seq_len]["Demand"])
        dates.append(rows[i + seq_len]["Date"])
    return X, y, dates


def train_test_split(X: list, y: list, dates: list | None = None, test_ratio: float = 0.2):
    """
    🔧 [전력수요 변경] dates 가 주어지면 날짜 기준(TEST_START 이후 = test)으로 나눈다.
    dates 가 없거나(재학습용 짧은 구간) TEST_START 이후 데이터가 없으면 원본처럼 시간순 비율로 나눈다.
    어느 쪽이든 시간 순서를 유지한다 (미래 데이터 누수 방지).
    """
    if dates is not None:
        split_idx = next((k for k, d in enumerate(dates) if d >= TEST_START), len(dates))
        if 0 < split_idx < len(dates):
            return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
