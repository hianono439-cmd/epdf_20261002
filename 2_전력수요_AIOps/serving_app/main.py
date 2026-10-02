"""
FastAPI 앱 진입점.

Day1: app 생성, 라우터(predict, health) 등록, startup 이벤트에서 로딩 모드에 따라 모델 준비
Day2: data 라우터 등록 (전력수요 데이터 업로드)
Day3: "aiops" 로거를 logs/aiops.log 파일로 연결(로깅 설정) + logs 라우터(로그 파일 조회) 등록

✏️ [데모 대시보드 수정] 교수님 데모처럼
  - 모든 API 요청을 logs/requests.log 에 기록하는 미들웨어 추가 (운영 지표 요약의 재료)
  - metrics 라우터(운영 지표·재학습 이력·현재 모델·알람·시스템 정보) 등록
  - 대시보드: static/index.html 을 4탭 운영 대시보드로 새로 작성, 이전 화면은 static/basic.html 로 보존

정적 대시보드: serving_app/static/index.html 이 /health · /predict · /predict/batch-test ·
/data/upload · /logs · /metrics 를 호출하는 확인용 화면입니다. API 라우터를 먼저 등록한 뒤
StaticFiles를 "/"에 마지막으로 mount해야, /predict 같은 API 경로가 정적 파일보다
먼저 매칭됩니다(Starlette는 등록 순서대로 라우트를 검사합니다).
"""
import logging
import os
import time  # ✏️ [데모 대시보드 수정] 요청 처리시간 측정용

from fastapi import FastAPI, Request  # ✏️ [데모 대시보드 수정] Request 추가 (미들웨어)
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.monitoring.request_log import log_request  # 🆕 [데모 대시보드 추가]
from serving_app.routers import data, health, logs, metrics, predict  # ✏️ [데모 대시보드 수정] metrics 추가

# monitoring/retrain_trigger.py가 쓰는 "aiops" 로거를 logs/aiops.log 파일에 연결한다.
# (routers/logs.py가 같은 디렉토리를 읽기 전용으로 노출한다.) 여기서 이 로거 하나만
# 직접 설정하므로, uvicorn 자체 로깅 설정과 충돌하지 않는다.
_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(os.path.join(_LOG_DIR, "aiops.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())  # 터미널에서도 동일하게 확인 가능

app = FastAPI(title="전력수요 예측 Serving & AIOps")  # 🔧 [전력수요 변경] 제목

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(data.router)  # 전력수요 데이터 업로드
app.include_router(logs.router)  # 대시보드: 재학습 로그 파일 조회
app.include_router(metrics.router)  # 🆕 [데모 대시보드 추가] 운영 지표·재학습 이력·알람·시스템 정보

# 🆕 [데모 대시보드 추가] 요청 로깅 미들웨어 — 모든 요청이 이 함수를 거쳐 갑니다.
#   처리 전후 시각 차이로 응답시간을 재고, 결과(상태코드)와 함께 logs/requests.log 에 한 줄 남깁니다.
#   대시보드 자체가 5초마다 부르는 /metrics, /logs 와 정적 파일(HTML·아이콘)은 "서비스 요청"이 아니라서 제외합니다.
_NOT_SERVICE_PREFIXES = ("/metrics", "/logs")
_SERVICE_PREFIXES = ("/predict", "/health", "/data")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    status = 500  # 처리 중 예외가 나면 실패(500)로 기록
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if path.startswith(_SERVICE_PREFIXES) and not path.startswith(_NOT_SERVICE_PREFIXES):
            log_request(request.method, path, status, (time.perf_counter() - start) * 1000)


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")  # 대시보드 UI


@app.on_event("startup")
def startup():
    # Day1 실습 포인트: LOADING_MODE=eager 로 켜고 서버 시작 시간을 lazy와 비교해보세요.
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
