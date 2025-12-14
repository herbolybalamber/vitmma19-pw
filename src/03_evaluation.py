from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from util.config_manager import settings
from util.logger import Logger

warnings.filterwarnings('ignore')

logger = Logger("evaluation")

# ============================================================================ 
# CONFIG
# ============================================================================ 

@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Model settings
    model_name: str = settings.get("train.model_name")
    num_classes: int = settings.get("train.num_classes")  # expected: 5
    max_length: int = settings.get("train.max_token_length")

    # Data settings
    batch_size: int = settings.get("train.batch_size")
    val_split: float = settings.get("train.val_split")
    random_seed: int = settings.get("train.random_seed")

    # Paths
    _train_path: str = settings.get("train.train_path")
    _test_path: str = settings.get("train.test_path")
    _model_dir: str = settings.get("train.model_dir")
    _model_path: str = settings.get("train.model_path")

    @property
    def train_path(self) -> Path:
        return Path(self._train_path)

    @property
    def test_path(self) -> Path:
        return Path(self._test_path)

    @property
    def model_dir(self) -> Path:
        return Path(self._model_dir)

    @property
    def model_path(self) -> Path:
        return Path(self._model_path)


# ============================================================================ 
# DATASET
# ============================================================================ 

class ASZFDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[Any]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings['labels'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.encodings['labels'][idx], dtype=torch.long),
        }


# ============================================================================ 
# DATA LOADER
# ============================================================================ 

class EvalDataLoader:
    """Handles data loading for evaluation."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def load_tokenizer(self) -> AutoTokenizer:
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        logger.info("✓ Tokenizer loaded")
        return self.tokenizer

    def tokenize_texts(self, texts: np.ndarray, labels: np.ndarray) -> Dict[str, List[Any]]:
        enc = self.tokenizer(
            texts.tolist(),
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
        )

        # convert labels 1–5 → 0–4
        labels = labels.astype(int) - 1

        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': labels.tolist(),
        }

    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
        logger.info("PREPARING DATA FOR EVALUATION")

        df_train = pd.read_csv(self.config.train_path)
        df_test = pd.read_csv(self.config.test_path)

        logger.info(f"Training data: {df_train.shape}")
        logger.info(f"Test data:     {df_test.shape}")

        self.load_tokenizer()

        # ------------------------------------------------------------------ 
        # Class weights (labels already 1–5 in CSV)
        # ------------------------------------------------------------------ 
        raw_labels = df_train['label_numeric'].astype(int).values
        zero_based_labels = raw_labels - 1

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.config.num_classes),
            y=zero_based_labels,
        )

        class_weights_tensor = torch.tensor(weights, dtype=torch.float)

        # ------------------------------------------------------------------ 
        # Tokenization
        # ------------------------------------------------------------------ 
        train_enc = self.tokenize_texts(
            df_train['text'].values,
            df_train['label_numeric'].values,
        )

        test_enc = self.tokenize_texts(
            df_test['text'].values,
            df_test['label_numeric'].values,
        )

        # ------------------------------------------------------------------ 
        # Validation split
        # ------------------------------------------------------------------ 
        _, val_idx = train_test_split(
            np.arange(len(train_enc['labels'])),
            test_size=self.config.val_split,
            random_state=self.config.random_seed,
            stratify=train_enc['labels'],
        )

        val_enc = {
            k: [train_enc[k][i] for i in val_idx]
            for k in train_enc
        }

        logger.info(f"Validation samples: {len(val_enc['labels'])}")
        logger.info(f"Test samples:       {len(test_enc['labels'])}")

        # ------------------------------------------------------------------ 
        # DataLoaders
        # ------------------------------------------------------------------ 
        device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )

        pin_memory = device.type in {"cuda", "mps"}

        val_loader = DataLoader(
            ASZFDataset(val_enc),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            ASZFDataset(test_enc),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin_memory,
        )

        return val_loader, test_loader, class_weights_tensor


# ============================================================================ 
# EVALUATOR
# ============================================================================ 

class ModelEvaluator:

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

    def load_model(self, class_weights: torch.Tensor) -> Optional[Dict[str, Any]]:
        logger.info(f"Device: {self.device}")
        logger.info(f"Loading model from: {self.config.model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
        )

        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        logger.info(f"✓ Model loaded (epoch {checkpoint.get('epoch')})")
        return checkpoint.get('history')

    def evaluate(self, loader: DataLoader, name: str):
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        preds, labels_all = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {name}"):
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                out = self.model(input_ids=ids, attention_mask=mask)
                logits = out.logits

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                p = torch.argmax(logits, dim=1)
                correct += (p == labels).sum().item()
                total += labels.size(0)

                preds.extend(p.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        return (
            total_loss / len(loader),
            correct / total,
            np.array(preds),
            np.array(labels_all),
        )

    def within_k_accuracy(self, true: np.ndarray, pred: np.ndarray, k: int = 1) -> float:
        """
        Calculate accuracy where prediction within k classes of true is correct.

        Args:
            true: True labels
            pred: Predicted labels
            k: Tolerance (default 1)

        Returns:
            Within-k accuracy
        """
        return float((np.abs(true - pred) <= k).mean())

    def print_classification_report(self, labels: np.ndarray, predictions: np.ndarray, split_name: str):
        """Print classification report for a dataset."""
        logger.info(f"\nClassification Report ({split_name}):")
        logger.info(classification_report(
            labels,
            predictions,
            target_names=[f'Class {i}' for i in range(1, 6)],
            digits=4,
        ))

    def print_ordinal_metrics(
        self, 
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
    ) -> None:
        """Print ordinal evaluation metrics."""
        logger.info(f"\n{split_name}:")
        logger.info(f"  Exact Accuracy:     {(predictions == labels).mean():.4f}")
        logger.info(f"  Within-1 Accuracy:  {self.within_k_accuracy(labels, predictions, 1):.4f}")
        logger.info(f"  Within-2 Accuracy:  {self.within_k_accuracy(labels, predictions, 2):.4f}")

    def print_per_class_recall(
        self,
        val_labels: np.ndarray,
        val_preds: np.ndarray,
        test_labels: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """Print per-class recall comparison."""

        logger.info("\nRecall by class:")
        logger.info("-" * 40)
        logger.info(f"{ 'Class':<10} {'Validation':<15} {'Test':<15}")
        logger.info("-" * 40)

        for c in range(5):
            val_mask: np.ndarray = val_labels == c
            test_mask: np.ndarray = test_labels == c

            val_recall: float = (val_preds[val_mask] == c).sum() / val_mask.sum() if val_mask.sum() > 0 else 0
            test_recall: float = (test_preds[test_mask] == c).sum() / test_mask.sum() if test_mask.sum() > 0 else 0

            logger.info(f"Class {c+1:<4} {val_recall:<15.4f} {test_recall:<15.4f}")

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
        cmap: str = 'Blues',
        normalize: str | None = None,
    ) -> None:
        """Plot and save confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        cm: np.ndarray = confusion_matrix(labels, predictions)

        fmt = 'd'
        if normalize:
            cm_plot = cm.astype(float)
            if normalize == "true":
                row_sums = cm_plot.sum(axis=1, keepdims=True)
                cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)
                fmt = ".1%"
            elif normalize == "pred":
                col_sums = cm_plot.sum(axis=0, keepdims=True)
                cm_plot = np.divide(cm_plot, col_sums, out=np.zeros_like(cm_plot), where=col_sums != 0)
                fmt = ".1%"
            elif normalize == "all":
                total = cm_plot.sum()
                cm_plot = (cm_plot / total) if total else cm_plot
                fmt = ".1%"
        else:
            cm_plot = cm

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=[f'Pred {i}' for i in range(1, 6)],
            yticklabels=[f'True {i}' for i in range(1, 6)],
            vmin=0.0 if normalize else None,
            vmax=1.0 if normalize else None,
        )

        title_extra = f" (normalized: {normalize})" if normalize else ""
        plt.title(f'Confusion Matrix - {split_name}{title_extra}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path: Path = self.config.model_dir / f'confusion_matrix_{split_name.lower().replace(" ", "_")}{"_norm_" + normalize if normalize else ""}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f"✓ Confusion matrix saved to {save_path}")

    def plot_training_history(self, history: Any, name: str):
        """Plots training and validation loss and accuracy."""
        if not history or not hasattr(history, 'train_loss'):
             logger.warning("History object not found or invalid. Skipping training history plot.")
             return
        
        history_dict = asdict(history)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history_dict['train_loss'], label='Train Loss')
        plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title(f'Loss over epochs - {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history_dict['train_acc'], label='Train Accuracy')
        plt.plot(history_dict['val_acc'], label='Validation Accuracy')
        plt.title(f'Model Accuracy over Epochs - {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        out_path = self.config.model_dir / f"training_history_{name.lower()}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"✓ Training history plot saved: {out_path}")


# ============================================================================ 
# MAIN
# ============================================================================ 

def main():
    logger.info("MODEL EVALUATION")

    cfg = EvalConfig()

    data = EvalDataLoader(cfg)
    val_loader, test_loader, class_weights = data.prepare_dataloaders()

    evaluator = ModelEvaluator(cfg)
    history = evaluator.load_model(class_weights)

    if history:
        evaluator.plot_training_history(history, "Training")

    logger.info("EVALUATING ON VALIDATION AND TEST SETS")
    _, _, v_pred, v_true = evaluator.evaluate(val_loader, "Validation")
    _, _, t_pred, t_true = evaluator.evaluate(test_loader, "Test")

    evaluator.print_classification_report(v_true, v_pred, "Validation")
    evaluator.print_classification_report(t_true, t_pred, "Test")
    
    evaluator.print_ordinal_metrics(v_true, v_pred, "Validation")
    evaluator.print_ordinal_metrics(t_true, t_pred, "Test")

    evaluator.print_per_class_recall(v_true, v_pred, t_true, t_pred)
    
    # Generate and save confusion matrices
    evaluator.plot_confusion_matrix(v_true, v_pred, "Validation")
    evaluator.plot_confusion_matrix(v_true, v_pred, "Validation", normalize="true")
    evaluator.plot_confusion_matrix(t_true, t_pred, "Test")
    evaluator.plot_confusion_matrix(t_true, t_pred, "Test", normalize="true")


if __name__ == '__main__':
    main()