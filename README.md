# 전력수요 예측 AIOps (팀 공유용)

한국 일별 전력수요를 LSTM으로 예측하고, FastAPI 서빙 → MLflow 학습·등록 → 드리프트 감지 → 자동 재학습·재배포까지
이어지는 AIOps 파이프라인입니다. 수업(모델 서빙 및 AIOps 구성)의 HAIC 실습 스켈레톤 구조를 바탕으로 만들었습니다.

- 코드와 상세 실행법: [`2_전력수요_AIOps/README.md`](2_전력수요_AIOps/README.md)
- 무엇을 왜 바꿨는지: [`변경내역.md`](변경내역.md) (코드 주석 `🔧` `🆕` `✏️` 표시와 1:1 대응)

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 데이터 | 2021~2025 일별 전국 전력수요(GWh, 전력거래소) + 인구가중 전국 기온(기상청 ASOS) + 공휴일·명절·징검다리 |
| 분할 | 2021~2023 학습 / 2024 검증(설정값 결정) / 최종 모델 2021~2024 학습 → **2025 평가** |
| 모델 | LSTM 3층, 입력 = 최근 14일 수요 + 예측일 기온·냉난방도일·휴일·요일·명절·징검다리 (9개 피처) |
| 2025 성능 | **RMSE 39.2 GWh, MAPE 1.99%** (기준선 "어제와 같음" 6.25%, "지난주 같은 요일" 5.98%) |
| 드리프트 | 최근 28일 RMSE > 60 GWh → fine-tuning → 게이트(≤60 GWh, 기존 모델보다 좋을 때) 통과 시 Production 승격 |
| 대시보드 | 4탭 (Dashboard / Simulation / Datasets / System) — 운영 지표, 재학습 이력, 알람 |

## 빠른 시작

```bash
git clone https://github.com/hianono439-cmd/epdf_20261002.git
```
```bash
cd epdf_20261002/2_전력수요_AIOps
```
```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
```bash
mkdir -p data/uploads && cp data/sample_power_daily.csv data/uploads/
```
```bash
python serving_app/train_and_register.py
```
```bash
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
```
브라우저에서 http://localhost:8077/ → **Simulation** 탭에서 "정상 배치 전송" → "드리프트 배치 전송"

Docker로 한 번에 띄우기 (http://localhost:8000/):
```bash
docker compose -f serving_app/docker-compose.yml up --build
```

## 같이 수정할 때

- 바꾼 곳에는 기존 규칙처럼 주석을 달아 주세요 (예: `# 🔧 [전력수요 변경] ...`), 그리고 `변경내역.md`에 한 줄 추가
- `main`에 바로 올리기보다 각자 브랜치에서 작업 → Pull Request로 리뷰
- 실행하면 생기는 `mlruns/`, `mlflow.db`, `logs/`, `data/uploads/`, `data/observed/`는 `.gitignore`로 제외돼 있어요

## 알아둘 점

- 2025 성능은 **실제 관측 기온**을 넣은 결과입니다. 운영에서는 예보 기온을 쓰므로 오차가 더 커질 수 있습니다.
- 명절·징검다리 변수로 추석 오차는 크게 줄었지만(−134.7 → −10.2 GWh), 징검다리는 오히려 과하게 낮게 예측하는 날이 있어요 → 변경내역.md 6장의 개선 후보 참고
- 학습용 CSV에 `Myeongjeol`·`Bridge` 열이 없으면 학습이 멈춥니다. `data/sample_power_daily.csv`를 업로드하세요
- 원본 데이터 수집·전처리 스크립트(시도별 기온 가중, 공휴일 표)는 이 저장소에 포함돼 있지 않습니다 (`data/sample_power_daily.csv`가 그 결과물)
