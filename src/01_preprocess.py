from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from util.config_manager import settings
from util.logger import Logger

logger = Logger("refinement")

MIN_CONTENT_LENGTH: int = settings.get("preprocess.min_text_length")
MIN_AVG_LEAD_TIME: float = settings.get("preprocess.min_mean_lead_time")
source_csv_path: str = settings.get("preprocess.aggregated_file")
destination_dir: str = settings.get("preprocess.processed_dir")


# =========================
# DATA SUMMARY
# =========================

@dataclass
class DataSummary:
    entry_count: int = 0
    unique_content_count: int = 0
    unique_annotator_count: int = 0
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    avg_content_length: float = 0.0
    median_content_length: float = 0.0
    avg_lead_time: float = 0.0
    median_lead_time: float = 0.0

    def display(self, name: str = "Dataset Summary") -> None:
        logger.info(f"{name}:")
        logger.info(f"  Total Entries: {self.entry_count}")
        logger.info(f"  Unique Content Snippets: {self.unique_content_count}")
        logger.info(f"  Unique Annotators: {self.unique_annotator_count}")
        logger.info(f"  Avg Content Length: {self.avg_content_length:.0f} chars")
        logger.info(f"  Median Content Length: {self.median_content_length:.0f} chars")
        logger.info(f"  Avg Annotation Time: {self.avg_lead_time:.2f}s")
        logger.info(f"  Median Annotation Time: {self.median_lead_time:.2f}s")

        if self.rating_distribution:
            total = sum(self.rating_distribution.values())
            logger.info("  Rating Distribution:")
            for rating in sorted(self.rating_distribution):
                count = self.rating_distribution[rating]
                logger.info(f"    Rating {rating}: {count} ({count / total * 100:.2f}%)")


# =========================
# PREPROCESSING LOG
# =========================

@dataclass
class PreprocessingLog:
    initial_entry_count: int = 0
    test_set_entry_count: int = 0
    duplicate_entries_removed: int = 0
    short_content_entries_removed: int = 0
    low_quality_annotator_entries_removed: int = 0
    annotators_excluded: List[str] = field(default_factory=list)
    final_train_entry_count: int = 0
    final_validation_entry_count: int = 0
    final_test_entry_count: int = 0

    def display(self) -> None:
        logger.info("DATA PREPROCESSING REPORT")
        logger.info(f"Initial entry count: {self.initial_entry_count}")
        logger.info(f"Duplicate entries removed: {self.duplicate_entries_removed}")
        logger.info(f"Short content entries removed: {self.short_content_entries_removed}")
        logger.info(f"Entries removed due to rushed annotators: {self.low_quality_annotator_entries_removed}")

        if self.annotators_excluded:
            logger.info(f"Excluded annotators: {', '.join(self.annotators_excluded)}")

        logger.info(f"Final training entries: {self.final_train_entry_count}")
        logger.info(f"Final validation entries: {self.final_validation_entry_count}")
        logger.info(f"Final test entries: {self.final_test_entry_count}")


# =========================
# DATASET PROCESSOR
# =========================

class DatasetProcessor:
    """
    Manages the data refinement process for readability ratings.
    """

    RELEVANT_COLUMNS: List[str] = [
        "student_code",
        "json_filename",
        "text",
        "label_text",
        "label_numeric",
        "annotation_created_at",
        "lead_time_seconds",
    ]

    # 🔒 PROTECTED ANNOTATORS (NEVER EXCLUDED)
    PROTECTED_ANNOTATORS: Set[str] = {"hubert"}

    def __init__(self, source_path: Path, dest_dir: Path):
        self.source_path = source_path
        self.destination_dir = dest_dir
        self.dataframe: Optional[pd.DataFrame] = None
        self.validation_dataframe: Optional[pd.DataFrame] = None
        self.test_dataframe: Optional[pd.DataFrame] = None
        self.log = PreprocessingLog()

    # ---------- IO ----------

    def read_source_data(self) -> pd.DataFrame:
        logger.info(f"Reading source data from {self.source_path}")
        self.dataframe = pd.read_csv(self.source_path)
        self.log.initial_entry_count = len(self.dataframe)
        return self.dataframe

    # ---------- SPLITS ----------

    def create_stratified_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_val, self.test_dataframe = train_test_split(
            self.dataframe,
            test_size=0.2,
            stratify=self.dataframe["label_numeric"],
            random_state=42,
        )

        self.dataframe, self.validation_dataframe = train_test_split(
            train_val,
            test_size=0.1,
            stratify=train_val["label_numeric"],
            random_state=42,
        )

        self.log.final_train_entry_count = len(self.dataframe)
        self.log.final_validation_entry_count = len(self.validation_dataframe)
        self.log.final_test_entry_count = len(self.test_dataframe)

        return self.dataframe, self.validation_dataframe, self.test_dataframe

    # ---------- CLEANING ----------

    def filter_columns(self) -> pd.DataFrame:
        self.dataframe = self.dataframe[self.RELEVANT_COLUMNS].copy()
        self.dataframe.rename(columns={"annotation_created_at": "labeled_at"}, inplace=True)
        return self.dataframe

    def deduplicate_entries(self) -> pd.DataFrame:
        initial = len(self.dataframe)
        self.dataframe.sort_values(
            by=["text", "lead_time_seconds"],
            ascending=[True, False],
            inplace=True,
        )
        self.dataframe.drop_duplicates(subset=["text"], keep="first", inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.log.duplicate_entries_removed = initial - len(self.dataframe)
        return self.dataframe

    def calculate_text_length(self) -> pd.DataFrame:
        self.dataframe["content_length"] = self.dataframe["text"].str.len()
        return self.dataframe

    def remove_short_rows(self) -> pd.DataFrame:
        initial = len(self.dataframe)
        self.dataframe = self.dataframe[
            self.dataframe["content_length"] >= MIN_CONTENT_LENGTH
        ].copy()
        self.log.short_content_entries_removed = initial - len(self.dataframe)
        return self.dataframe

    # ---------- ANNOTATOR FILTERING ----------

    def exclude_rushed_annotators(self) -> pd.DataFrame:
        logger.info(
            f"Excluding annotators with avg lead time < {MIN_AVG_LEAD_TIME}s "
            f"(protected: {', '.join(self.PROTECTED_ANNOTATORS)})"
        )

        initial = len(self.dataframe)

        annotator_stats = (
            self.dataframe
            .groupby("student_code")["lead_time_seconds"]
            .mean()
            .reset_index()
        )

        annotators_to_exclude = annotator_stats[
            (annotator_stats["lead_time_seconds"] < MIN_AVG_LEAD_TIME) &
            (~annotator_stats["student_code"]
              .str.lower()
              .isin(self.PROTECTED_ANNOTATORS))
        ]["student_code"].tolist()

        if annotators_to_exclude:
            logger.info("Excluded annotators:")
            for a in annotators_to_exclude:
                logger.info(f"  - {a}")

            self.dataframe = self.dataframe[
                ~self.dataframe["student_code"].isin(annotators_to_exclude)
            ].copy()

            self.log.annotators_excluded = annotators_to_exclude

        self.log.low_quality_annotator_entries_removed = initial - len(self.dataframe)
        return self.dataframe

    # ---------- STATS ----------

    def calculate_statistics(self, df: pd.DataFrame) -> DataSummary:
        if "content_length" not in df.columns:
            content_lengths = df["text"].str.len()
        else:
            content_lengths = df["content_length"]

        summary = DataSummary(
            entry_count=len(df),
            unique_content_count=df["text"].nunique(),
            unique_annotator_count=df["student_code"].nunique(),
            rating_distribution=df["label_numeric"].value_counts().to_dict(),
            avg_content_length=float(content_lengths.mean()),
            median_content_length=float(content_lengths.median()),
            avg_lead_time=float(df["lead_time_seconds"].mean()),
            median_lead_time=float(df["lead_time_seconds"].median()),
        )
        return summary

    # ---------- OUTPUT ----------

    def write_output_files(self) -> None:
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        self.dataframe.to_csv(self.destination_dir / "train.csv", index=False)
        self.validation_dataframe.to_csv(self.destination_dir / "validation.csv", index=False)
        self.test_dataframe.to_csv(self.destination_dir / "test.csv", index=False)

    # ---------- PIPELINE ----------

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.read_source_data()
        self.create_stratified_splits()
        self.filter_columns()
        self.deduplicate_entries()
        self.calculate_text_length()
        self.remove_short_rows()
        self.exclude_rushed_annotators()

        logger.info("FINAL DATASET SUMMARIES")
        self.calculate_statistics(self.dataframe).display("Training Set")
        self.calculate_statistics(self.validation_dataframe).display("Validation Set")
        self.calculate_statistics(self.test_dataframe).display("Test Set")

        self.write_output_files()
        self.log.display()

        logger.info("DATA REFINEMENT COMPLETE")
        return self.dataframe, self.validation_dataframe, self.test_dataframe


# =========================
# ENTRY POINT
# =========================

def main() -> None:
    processor = DatasetProcessor(
        Path(source_csv_path),
        Path(destination_dir),
    )
    processor.process_dataset()


if __name__ == "__main__":
    main()