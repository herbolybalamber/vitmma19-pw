from pathlib import Path
import tempfile
import requests
import zipfile
import shutil
import json
import csv
import re
from typing import List, Dict, Any, Optional

from util.config_manager import settings
from util.logger import Logger

logger = Logger("aggregate_jsons")

# =========================
# CONFIG
# =========================

EXCLUDED_FOLDERS = {f.lower() for f in settings.get("preprocess.folders_to_exclude")}
INPUT_DIR = Path(settings.get("preprocess.user_input_dir"))
OUTPUT_DIR = Path(settings.get("preprocess.aggregated_dir"))
DATA_URL = settings.get("preprocess.data_url")


# =========================
# DOWNLOAD & EXTRACT
# =========================

def download_and_extract_data(url: str, target_dir: Path) -> bool:
    """
    Download a zip file from URL and extract its contents to target directory.
    """
    logger.info("Downloading dataset...")

    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            zip_path = Path(tmp.name)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)

        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            contents = list(extract_path.iterdir())
            source_dir = contents[0] if len(contents) == 1 and contents[0].is_dir() else extract_path

            if target_dir.exists():
                shutil.rmtree(target_dir)

            shutil.copytree(source_dir, target_dir)

        zip_path.unlink(missing_ok=True)
        logger.info("Download and extraction successful.")
        return True

    except Exception as e:
        logger.error("Failed to download or extract dataset.", exc_info=e)
        return False


# =========================
# LABEL HANDLING
# =========================

LABEL_REGEX = re.compile(r"^(\d+)")

def extract_label_number(label_text: str) -> Optional[int]:
    if not isinstance(label_text, str):
        return None
    match = LABEL_REGEX.match(label_text.strip())
    return int(match.group(1)) if match else None


# =========================
# JSON PROCESSING
# =========================

def process_json_file(json_path: Path, student_code: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        for task in data:
            annotations = task.get("annotations") or []
            text = task.get("data", {}).get("text", "")

            if not text or not annotations:
                continue

            for ann in annotations:
                result = ann.get("result") or []
                if not result:
                    continue

                choices = result[0].get("value", {}).get("choices") or []
                if not choices:
                    continue

                label_text = choices[0]
                label_numeric = extract_label_number(label_text)

                records.append({
                    "student_code": student_code,
                    "source_file": task.get("file_upload", ""),
                    "json_filename": json_path.name,
                    "task_id": task.get("id", ""),
                    "task_inner_id": task.get("inner_id", ""),
                    "annotation_id": ann.get("id", ""),
                    "text": text,
                    "label_text": label_text,
                    "label_numeric": label_numeric,
                    "completed_by": ann.get("completed_by", ""),
                    "annotation_created_at": ann.get("created_at", ""),
                    "annotation_updated_at": ann.get("updated_at", ""),
                    "task_created_at": task.get("created_at", ""),
                    "task_updated_at": task.get("updated_at", ""),
                    "lead_time_seconds": ann.get("lead_time", None),
                })

    except Exception as e:
        logger.error(f"Failed to process {json_path}", exc_info=e)

    return records


# =========================
# AGGREGATION
# =========================

CSV_FIELDS = [
    "student_code",
    "source_file",
    "json_filename",
    "task_id",
    "task_inner_id",
    "annotation_id",
    "text",
    "label_text",
    "label_numeric",
    "completed_by",
    "annotation_created_at",
    "annotation_updated_at",
    "task_created_at",
    "task_updated_at",
    "lead_time_seconds",
]


def aggregate_labeled_data(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []

    for student_dir in sorted(input_dir.iterdir()):
        if not student_dir.is_dir():
            continue

        if student_dir.name.lower() in EXCLUDED_FOLDERS:
            logger.info(f"Skipping excluded folder: {student_dir.name}")
            continue

        logger.info(f"Processing student: {student_dir.name}")
        json_files = list(student_dir.glob("*.json"))

        for json_file in json_files:
            records = process_json_file(json_file, student_dir.name)
            all_records.extend(records)
            logger.info(f"  {json_file.name}: {len(records)} records")

    output_file = output_dir / "labeled_data.csv"

    if not all_records:
        logger.warning("No records found. Nothing to write.")
        return

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    # ---------- STATS ----------
    label_dist: Dict[int, int] = {}
    for r in all_records:
        if r["label_numeric"] is not None:
            label_dist[r["label_numeric"]] = label_dist.get(r["label_numeric"], 0) + 1

    logger.info("Aggregation complete.")
    logger.info(f"Total records: {len(all_records)}")
    logger.info(f"Unique students: {len(set(r['student_code'] for r in all_records))}")
    logger.info(f"Unique texts: {len(set(r['text'] for r in all_records))}")
    logger.info("Label distribution:")
    for k in sorted(label_dist):
        logger.info(f"  Label {k}: {label_dist[k]}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    logger.info("Starting aggregation pipeline")
    logger.info(f"Input dir: {INPUT_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info("-" * 60)

    if not download_and_extract_data(DATA_URL, INPUT_DIR):
        logger.error("Download failed. Aborting.")
        raise SystemExit(1)

    logger.info("-" * 60)
    aggregate_labeled_data(INPUT_DIR, OUTPUT_DIR)