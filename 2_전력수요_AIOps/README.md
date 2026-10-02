# 전력수요 예측 모델 서빙 및 AIOps

교수님의 HAIC 3일 실습 스켈레톤(`project/`)을 그대로 바탕으로, 예측 대상을
**가상 종목 종가 → 한국 일별 전력수요(GWh)** 로 바꾼 프로젝트입니다.
최근 14일의 수요와 다음날 기온·휴일·요일을 보고 다음날 전력수요를 예측하는 **LSTM** 모델을
Day1(서빙) → Day2(MLOps) → Day3(AIOps) 순서로 하나의 서빙 서버 위에 쌓아 올립니다.

> 코드에서 바뀐 곳은 모두 주석으로 표시돼 있습니다.
> - `✅ [빈칸 N 정답]` : 교수님 빈칸을 채운 곳
> - `🔧 [전력수요 변경]` : 전력수요 예측에 맞게 바꾼 곳
>
> 전체 변경 내역과 이유는 상위 폴더의 [`변경내역.md`](../변경내역.md)를 보세요.

## 데이터

`data/sample_power_daily.csv` (2021-01-01 ~ 2025-12-31, 1,826일)

| 컬럼 | 의미 | 출처 |
|---|---|---|
| Date | 날짜 | |
| Demand | 일 전력수요 (GWh) = 시간별 수요(MW) 24개의 합 ÷ 1000 | 한국전력거래소 시간별 전국 전력수요량 |
| Temp | 전국 인구가중 일평균기온 (℃) | 기상청 ASOS 97개 지점 → 17개 시도 → 연도별 인구로 가중 |
| Holiday | 공휴일·명절·대체·임시공휴일·선거일이면 1 | 직접 입력 (주말은 날짜로 자동 계산) |
| Myeongjeol | 설·추석 연휴(대체공휴일 포함)면 1 ➕ | 직접 입력 |
| Bridge | 징검다리: 평일인데 앞뒤 날이 모두 휴일이면 1 ➕ | 휴일·주말에서 계산 |

➕ 표시 열은 [명절·징검다리 추가] 작업에서 넣었습니다. 없는 CSV를 올리면 학습 스크립트가 멈추고 다시 업로드하라고 안내합니다.

만든 과정은 `팀플 데이터/preprocess.py`에 있습니다.
**2021~2024년으로 학습하고 2025년으로 성능을 확인**합니다. 모델 크기, 명절·징검다리 피처 채택 여부, 기준값(60 GWh), 드리프트 크기(+20%) 같은
설정값은 2025년을 보지 않고 **2021~2023 학습 / 2024 검증**으로 정했습니다(`scripts/calibrate_on_validation.py`).

## 모델

```
입력 (14일, 9개 피처) -> LSTM(32) -> LSTM(32) -> LSTM(16) -> Dense(16, relu) -> Dense(1)
피처 = [당일 수요, 다음날 기온, 다음날 난방도일, 다음날 냉방도일, 다음날 휴일(주말 포함), 다음날 요일 sin, cos,
        다음날 명절, 다음날 징검다리]
```
층 구성은 교수님 원본과 같고, 입력 모양만 (20, 2) → (14, 9)로 바뀌었습니다.

| 2025년 평가 | RMSE (GWh) | MAPE |
|---|---|---|
| **LSTM (명절·징검다리 포함, 9개 피처)** | **39.2** | **1.99%** |
| LSTM (이전 7개 피처) | 41.3 | 2.12% |
| 기준선: 어제와 같음 | 123.7 | 6.25% |
| 기준선: 지난주 같은 요일 | 127.6 | 5.98% |

## 실행 순서

```bash
pip install -r requirements.txt
```

**(선택) 검증 기간으로 설정값 다시 계산** — 결과는 `../docs/검증기간_보정결과.json`
```bash
python scripts/calibrate_on_validation.py
```

**Day1 — 로컬 baseline LSTM을 FastAPI로 서빙**
```bash
uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
```
대시보드(http://localhost:8077/)의 **Datasets** 탭에서 `data/sample_power_daily.csv`를 업로드한 뒤, 다른 터미널에서 실행합니다.
```bash
python scripts/train_baseline_v1.py
```

**Day2 — MLflow 학습·등록 + 게이트(RMSE ≤ 60 GWh)**
```bash
python serving_app/train_and_register.py
```
```bash
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
```

**Day3 — 드리프트 감지 → 자동 fine-tuning → 재배포**
```bash
python scripts/simulate_drift.py
```
시나리오와 시드를 고를 수 있습니다.
```bash
python scripts/simulate_drift.py --scenario temp_bias --seed 3
```

**컨테이너로 재현**
```bash
docker compose -f serving_app/docker-compose.yml up --build
```
컨테이너는 http://localhost:8000/ 에서 열립니다.

### 시뮬레이션을 처음 상태로 되돌리기
드리프트 배치로 재학습되면 Production 모델이 "드리프트가 낀 데이터"에 적응합니다. 그 상태에서
정상(실제 2025년) 배치를 다시 보내면 이번엔 그게 드리프트로 감지됩니다. 실제 운영에서도 일어나는
자연스러운 현상입니다. 데모를 처음부터 다시 하려면 아래 폴더·파일을 지우고 Day2 학습을 다시 실행하세요.
```bash
rm -rf mlflow.db mlruns data/observed logs
```
```bash
python serving_app/train_and_register.py
```

## 운영 대시보드 (교수님 데모 구성)

http://localhost:8077/ — 탭 4개 (✏️ 이전 1장짜리 화면은 http://localhost:8077/basic.html 에 그대로 있음)

| 탭 | 내용 | 데이터 출처 |
|---|---|---|
| **Dashboard** | 운영 지표 요약 5칸(요청 수·평균 응답시간·성공률·모델 RMSE·드리프트 점수, 5분/1시간/6시간/24시간), 재학습 파이프라인 7단계, 재학습 이력 표, 현재 운영 모델, 최근 알람 | `logs/requests.log`, `logs/drift_history.log`, MLflow Registry, `logs/aiops.log` |
| **Simulation** | 정상/드리프트 배치 전송 + 실제 vs 예측 그래프, 28일 RMSE·MAPE·판정 | `/data/sample-batch` → `/predict/batch-test` |
| **Datasets** | CSV 업로드, 현재 데이터 요약, 업로드·관측 파일 목록 | `/data/upload`, `/data/status`, `/metrics/system` |
| **System** | 서버 상태(로딩 모드·모델 소스·라이브러리 버전·가동 시간), 로그 파일 열람, API 목록 | `/health`, `/metrics/system`, `/logs` |

요청 로그는 터미널에서도 집계해 볼 수 있습니다.
```bash
python scripts/aggregate_metrics.py 1h
```

## /predict 요청 예시

```json
{
  "history": [
    {"date": "2025-08-11", "demand": 1731.5, "temp": 24.74, "holiday": 0},
    {"...": "14일치"}
  ],
  "target": {"date": "2025-08-25", "temp": 28.76, "holiday": 0}
}
```
응답 예시 (v1 모델, 실제 수요는 1,920.9 GWh):
```json
{"target_date": "2025-08-25", "predicted_demand_gwh": 1900.9, "model_version": "production-v1"}
```
`target.temp`는 실제 운영에서는 기상청 **예보** 기온을 넣습니다.

## 완료 기준 (원본 체크리스트를 전력수요 기준으로 바꿈)

- [x] `/data/upload`로 CSV를 올리면 업로드 완료로 표시되는가 (`/data/status`)
- [x] 정상 데이터(2025년 실제)는 RMSE 60 GWh 이내, 드리프트 데이터는 60 GWh 초과가 재현되는가
- [x] `logs/aiops.log`에 `[WARN] drift detected` → `[INFO] retrain triggered` → `[OK] new_rmse=...` 순서로 기록되는가
- [x] 재배포 후 `/predict` 응답의 `model_version`이 새 Production 버전(`production-v2`)으로 바뀌는가
