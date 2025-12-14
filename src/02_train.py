from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from tqdm import tqdm
import warnings

from util.config_manager import settings
from util.logger import Logger

warnings.filterwarnings("ignore")

log = Logger("hubert_training")


# ============================================================================
# RUNTIME PARAMETERS
# ============================================================================

@dataclass
class HubertRunConfig:
    hubert_name: str = settings.get("train.model_name")
    n_labels: int = settings.get("train.num_classes")
    seq_len: int = settings.get("train.max_token_length")

    batch_sz: int = settings.get("train.batch_size")
    lr: float = float(settings.get("train.learning_rate"))
    epochs: int = settings.get("train.num_epochs")
    warmup_ratio: float = settings.get("train.warmup_ratio")

    _train_src: str = settings.get("train.train_path")
    _val_src: str = settings.get("train.val_path")
    _test_src: str = settings.get("train.test_path")

    _model_out: str = settings.get("train.model_dir")
    _logs_out: str = settings.get("train.log_dir")

    seed: int = settings.get("train.random_seed")

    @property
    def train_csv(self) -> Path:
        return Path(self._train_src)

    @property
    def val_csv(self) -> Path:
        return Path(self._val_src)

    @property
    def test_csv(self) -> Path:
        return Path(self._test_src)

    @property
    def model_dir(self) -> Path:
        return Path(self._model_out)

    @property
    def log_dir(self) -> Path:
        return Path(self._logs_out)


# ============================================================================
# DATASET WRAPPER
# ============================================================================

class HubertTextDataset(Dataset):


    def __init__(self, encoded: Dict[str, List[Any]]):
        self.data = encoded

    def __len__(self) -> int:
        return len(self.data["labels"])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.data["input_ids"][index]),
            "attention_mask": torch.tensor(self.data["attention_mask"][index]),
            "labels": torch.tensor(self.data["labels"][index] - 1),
        }


# ============================================================================
# DATA PREPARATION
# ============================================================================

class HubertDataPipeline:

    def __init__(self, cfg: HubertRunConfig):
        self.cfg = cfg
        self.tokenizer: Optional[AutoTokenizer] = None
        self.weights: Optional[Dict[int, float]] = None

    def _load_csvs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        log.info("Reading CSV inputs")

        tr = pd.read_csv(self.cfg.train_csv)
        va = pd.read_csv(self.cfg.val_csv)
        te = pd.read_csv(self.cfg.test_csv)

        log.info(f"Train: {tr.shape}, Val: {va.shape}, Test: {te.shape}")
        return tr, va, te

    def _init_tokenizer(self) -> AutoTokenizer:
        log.info(f"Initializing HuBERT tokenizer: {self.cfg.hubert_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.hubert_name)
        return self.tokenizer

    def _encode(self, texts: np.ndarray, targets: np.ndarray) -> Dict[str, List[Any]]:
        tokens = self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.cfg.seq_len,
        )
        tokens["labels"] = targets.tolist()
        return tokens

    def _balance_weights(self, labels: pd.Series) -> Dict[int, float]:
        classes = np.unique(labels)
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        return dict(zip(classes, weights))

    def build(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, float]]:
        train_df, val_df, test_df = self._load_csvs()
        self._init_tokenizer()

        full_labels = pd.concat([train_df["label_numeric"], val_df["label_numeric"]])
        self.weights = self._balance_weights(full_labels)

        enc_train = self._encode(train_df["text"].values, train_df["label_numeric"].values)
        enc_val = self._encode(val_df["text"].values, val_df["label_numeric"].values)
        enc_test = self._encode(test_df["text"].values, test_df["label_numeric"].values)

        train_ds = HubertTextDataset(enc_train)
        val_ds = HubertTextDataset(enc_val)
        test_ds = HubertTextDataset(enc_test)

        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        loader_args = dict(
            batch_size=self.cfg.batch_sz,
            num_workers=2,
            pin_memory=device.type != "cpu",
        )

        return (
            DataLoader(train_ds, shuffle=True, **loader_args),
            DataLoader(val_ds, shuffle=False, **loader_args),
            DataLoader(test_ds, shuffle=False, **loader_args),
            self.weights,
        )


# ============================================================================
# TRAINING ENGINE
# ============================================================================

@dataclass
class EpochLog:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = 0.0


class HubertTrainer:


    def __init__(
        self,
        cfg: HubertRunConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Dict[int, float],
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weights = class_weights

        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model: Optional[AutoModelForSequenceClassification] = None
        self.optim: Optional[AdamW] = None
        self.scheduler = None
        self.loss_fn: Optional[nn.CrossEntropyLoss] = None

        self.history = EpochLog()
        self._seed()

    def _seed(self) -> None:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def initialize(self) -> None:
        log.info("Bootstrapping HuBERT model")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.hubert_name,
            num_labels=self.cfg.n_labels,
        ).to(self.device)

        w = torch.tensor(
            [self.weights[i] for i in range(1, 6)],
            dtype=torch.float,
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(weight=w)
        self.optim = AdamW(self.model.parameters(), lr=self.cfg.lr)

        total_steps = len(self.train_loader) * self.cfg.epochs
        warmup = int(total_steps * self.cfg.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optim,
            warmup,
            total_steps,
        )

        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        self.model.train()

        loss_sum, hit, count = 0.0, 0, 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch_idx}"):
            x = batch["input_ids"].to(self.device)
            m = batch["attention_mask"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optim.zero_grad()
            logits = self.model(x, attention_mask=m).logits
            loss = self.loss_fn(logits, y)

            loss.backward()
            self.optim.step()
            self.scheduler.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            hit += (preds == y).sum().item()
            count += y.size(0)

        return loss_sum / len(self.train_loader), hit / count

    def _validate(self) -> Tuple[float, float]:
        self.model.eval()

        loss_sum, hit, count = 0.0, 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["input_ids"].to(self.device)
                m = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)

                logits = self.model(x, attention_mask=m).logits
                loss = self.loss_fn(logits, y)

                loss_sum += loss.item()
                hit += (logits.argmax(dim=1) == y).sum().item()
                count += y.size(0)

        return loss_sum / len(self.val_loader), hit / count

    def train(self) -> EpochLog:
        log.info("Starting HuBERT fine-tuning")

        for ep in range(1, self.cfg.epochs + 1):
            tr_l, tr_a = self._run_epoch(ep)
            va_l, va_a = self._validate()

            self.history.train_loss.append(tr_l)
            self.history.train_acc.append(tr_a)
            self.history.val_loss.append(va_l)
            self.history.val_acc.append(va_a)

            log.info(f"Epoch {ep} | train={tr_a:.3f} | val={va_a:.3f}")

            if va_a > self.history.best_score:
                self.history.best_score = va_a
                self.history.best_epoch = ep
                self._save()

        return self.history

    def _save(self) -> None:
        path = self.cfg.model_dir / "hubert_best.pt"
        checkpoint = {
            'epoch': self.history.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        log.info(f"✓ Best HuBERT model saved → {path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    log.info("HuBERT text classification training")

    cfg = HubertRunConfig()

    pipeline = HubertDataPipeline(cfg)
    train_dl, val_dl, test_dl, weights = pipeline.build()

    trainer = HubertTrainer(cfg, train_dl, val_dl, weights)
    trainer.initialize()
    trainer.train()

    log.info("Training finished")


if __name__ == "__main__":
    main()