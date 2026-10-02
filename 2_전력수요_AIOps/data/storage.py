"""
업로드된 전력수요 데이터 파일 관리.

data/generate_haic_data.py로 자동 생성하던 방식 대신, 대시보드에서 CSV 파일을
직접 업로드하는 방식으로 바뀌었습니다 (serving_app/routers/data.py 참고).
업로드된 파일은 이 디렉터리(data/uploads/)에 타임스탬프가 붙은 이름으로 계속
쌓이고(과거 파일을 덮어쓰지 않습니다), 학습(train_and_register.py 등)은 항상
가장 최근에 올라온 파일 하나를 사용합니다.

🔧 [전력수요 변경] 관측 데이터 저장소(data/observed/observed.csv) 추가
   원본은 드리프트 재학습 때도 "업로드된 학습 CSV의 마지막 41행"을 썼습니다. 그런데 드리프트는
   서빙 중에 새로 들어온 데이터에서 일어나므로, 재학습은 그 "새로 관측된 데이터"로 해야 의미가
   있습니다. 그래서 /predict/batch-test 로 들어온 실제값(관측치)을 이 파일에 계속 쌓고,
   retrain_trigger.py 는 여기서 최근 구간을 잘라 fine-tuning 합니다.
"""
import csv
import glob
import os

UPLOAD_DIR = "data/uploads"
OBSERVED_PATH = "data/observed/observed.csv"  # 🔧 [전력수요 변경] 신규
OBSERVED_COLUMNS = ["Date", "Demand", "Temp", "Holiday"]  # 🔧 [전력수요 변경] 신규


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    """data/uploads/ 에 쌓인 CSV 중 가장 최근에 업로드된 파일의 경로를 반환한다."""
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 전력수요 데이터가 없습니다. 대시보드에서 CSV 파일을 먼저 업로드하세요 "
            f"(data/sample_power_daily.csv를 예시로 업로드해볼 수 있습니다 -> {upload_dir}/)."
        )
    return files[-1]


# 🔧 [전력수요 변경] 신규 함수 — 서빙 중 관측된 실제 데이터를 누적 저장
def append_observed(records: list[dict], path: str = OBSERVED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OBSERVED_COLUMNS)
        if is_new:
            writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in OBSERVED_COLUMNS})
