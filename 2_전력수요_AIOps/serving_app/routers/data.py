"""
전력수요 데이터 업로드 - data/generate_haic_data.py로 자동 생성하던 방식을 대체합니다.

🔧 [전력수요 변경] 필수 컬럼 Date/Close/Volume → Date/Demand/Temp/Holiday, 파일명 haic_ → power_,
   UTF-8-SIG(BOM) 허용, /data/status 를 수요·기온 요약으로, 대시보드용 GET /data/sample-batch 추가

/data 폴더는 이 라우터로 업로드된 CSV만 쌓이는 곳입니다(data/uploads/). 여러 번
업로드하면 계속 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상 가장
최근 파일 하나를 사용합니다(data/storage.py의 latest_upload()).

대시보드(static/index.html)에서 파일을 올리면 이 엔드포인트가 호출됩니다.
"""
import csv
import io
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.drift_scenarios import SCENARIOS, make_batch  # 🔧 [전력수요 변경]
from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Date", "Demand", "Temp", "Holiday"}  # 🔧 [전력수요 변경]
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # 🔧 [전력수요 변경] 엑셀에서 저장한 BOM 포함 CSV 도 허용
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"power_{int(time.time())}.csv")  # 🔧 [전력수요 변경] haic_ → power_
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    demands = [r["Demand"] for r in rows]  # 🔧 [전력수요 변경] 종가 → 수요·기온 요약
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_demand_gwh": min(demands),
        "max_demand_gwh": max(demands),
        "min_temp": min(r["Temp"] for r in rows),
        "max_temp": max(r["Temp"] for r in rows),
    }


# 🔧 [전력수요 변경] (신규) 대시보드의 "정상/드리프트 배치 전송" 버튼이 쓰는 배치 생성 API.
#    원본 대시보드는 브라우저에서 랜덤워크 가격을 만들었지만, 전력수요는 실제 2025년 기록이 필요해서
#    서버가 업로드된 CSV 에서 잘라 준다. (scripts/simulate_drift.py 와 같은 data/drift_scenarios.py 사용)
@router.get("/sample-batch")
def sample_batch(kind: str = "normal", scenario: str = "ramp"):
    if kind not in ("normal", "drift") or scenario not in SCENARIOS:
        raise HTTPException(400, f"kind=normal|drift, scenario={'|'.join(SCENARIOS)}")
    try:
        return {"records": make_batch(kind=kind, scenario=scenario)}
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
