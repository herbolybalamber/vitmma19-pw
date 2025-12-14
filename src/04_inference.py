from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
from collections import Counter, defaultdict
from typing import Optional

from util.config_manager import settings
from util.logger import Logger

try:
    # TensorBoard is optional; inference should not hard-fail without it.
    from torch.utils.tensorboard import summary  # type: ignore
except ModuleNotFoundError:
    summary = None  # TensorBoard not available in this environment

logger = Logger(
    "inference",
    log_dir=Path(settings.get("train.log_dir")),
    log_to_file=True,
)


# ============================
# CONFIG
# ============================

@dataclass
class InferenceConfig:
    model_name: str = settings.get("train.model_name")
    num_classes: int = settings.get("train.num_classes")  # MUST be 5
    max_length: int = settings.get("train.max_token_length")
    model_path: Path = Path(settings.get("train.model_path"))

    # internal labels are 0–4, user-facing labels are 1–5
    label_descriptions: Dict[int, str] = field(default_factory=lambda: {
        1: "Nagyon nehezen érthető",
        2: "Nehezen érthető",
        3: "Többnyire érthető",
        4: "Könnyen érthető",
        5: "Nagyon könnyen érthető",
    })

    # User-provided mapping: predicted label -> difficulty tag
    label_to_difficulty_tag: Dict[int, str] = field(default_factory=lambda: {
        1: "very_easy",
        2: "easy",
        3: "medium",
        4: "hard",
        5: "very_hard",
    })

    # Expected difficulty tag -> expected label (for evaluation)
    expected_difficulty_to_label: Dict[str, int] = field(default_factory=lambda: {
        "very_easy": 1,
        "easy": 2,
        "medium": 3,
        "hard": 4,
        "very_hard": 5,
    })

# ============================
# SAMPLE TEXTS
# ============================

SAMPLE_TEXTS = [
    # EASY
    {
        "id": 1,
        "text": "A szolgáltatás aktiválása a regisztrációt követően automatikusan megtörténik.",
        "description": "Simple service activation statement",
        "expected_difficulty": "easy",
    },
    {
        "id": 2,
        "text": "Az előfizetési díj minden hónap első napján esedékes.",
        "description": "Basic payment timing rule",
        "expected_difficulty": "easy",
    },
    {
        "id": 3,
        "text": "A felhasználó köteles érvényes e-mail címet megadni.",
        "description": "Basic user obligation",
        "expected_difficulty": "easy",
    },
    {
        "id": 4,
        "text": "A szerződés elektronikus úton jön létre.",
        "description": "Simple contract formation clause",
        "expected_difficulty": "easy",
    },

    # MEDIUM
    {
        "id": 5,
        "text": "A Szolgáltató jogosult a szolgáltatás nyújtását ideiglenesen szüneteltetni technikai karbantartás idejére, amelyről előzetesen tájékoztatja az Előfizetőt.",
        "description": "Service suspension due to maintenance",
        "expected_difficulty": "medium",
    },
    {
        "id": 6,
        "text": "A Felhasználó a fiókjához tartozó belépési adatokat köteles bizalmasan kezelni, és felel minden olyan tevékenységért, amely a fiókján keresztül történik.",
        "description": "Account security responsibility",
        "expected_difficulty": "medium",
    },
    {
        "id": 7,
        "text": "A szerződés megszűnése nem érinti a felek azon jogait és kötelezettségeit, amelyek jellegüknél fogva a megszűnést követően is fennmaradnak.",
        "description": "Survival clause",
        "expected_difficulty": "medium",
    },
    {
        "id": 8,
        "text": "A Szolgáltató jogosult alvállalkozót igénybe venni a szolgáltatás teljesítéséhez, amelyért úgy felel, mintha a teljesítést maga végezte volna.",
        "description": "Use of subcontractors",
        "expected_difficulty": "medium",
    },

    # HARD
    {
        "id": 9,
        "text": "A Szolgáltató nem vállal felelősséget az olyan károkért, amelyek a szolgáltatás igénybevételéhez használt harmadik fél által biztosított rendszerek hibájából erednek.",
        "description": "Third-party system liability exclusion",
        "expected_difficulty": "hard",
    },
    {
        "id": 10,
        "text": "A felek rögzítik, hogy a szerződés teljesítése során tudomásukra jutott üzleti titkokat kötelesek bizalmasan kezelni, és azokat harmadik személy részére nem hozhatják nyilvánosságra.",
        "description": "Confidentiality obligation",
        "expected_difficulty": "hard",
    },
    {
        "id": 11,
        "text": "A Felhasználó tudomásul veszi, hogy a szolgáltatás rendeltetésellenes használata esetén a Szolgáltató jogosult a szerződést azonnali hatállyal felmondani.",
        "description": "Immediate termination for misuse",
        "expected_difficulty": "hard",
    },
    {
        "id": 12,
        "text": "A szerződésből eredő jogok harmadik személyre történő átruházása kizárólag a másik fél előzetes írásbeli hozzájárulásával érvényes.",
        "description": "Assignment restriction clause",
        "expected_difficulty": "hard",
    },

    # VERY HARD
    {
        "id": 13,
        "text": "A felek megállapodnak abban, hogy a jelen szerződés értelmezésére és teljesítésére a magyar jog az irányadó, különös tekintettel a Polgári Törvénykönyv kötelmi jogi rendelkezéseire.",
        "description": "Governing law clause with legal reference",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 14,
        "text": "A Szolgáltató adatfeldolgozóként jár el a GDPR 28. cikke alapján, és kizárólag az Adatkezelő dokumentált utasításai szerint jogosult személyes adatokat kezelni.",
        "description": "GDPR data processor clause",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 15,
        "text": "Amennyiben a jelen szerződés bármely rendelkezése részben vagy egészben érvénytelennek bizonyul, az nem érinti a szerződés további rendelkezéseinek érvényességét.",
        "description": "Severability clause",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 16,
        "text": "A Szolgáltató kizárja felelősségét az elmaradt haszonért és a közvetett károkért, kivéve, ha azok szándékos vagy súlyosan gondatlan magatartásból erednek.",
        "description": "Advanced liability exclusion",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 17,
        "text": "A Felhasználó elfogadja, hogy a Szolgáltató a jogszabályi kötelezettségeinek teljesítése érdekében hatósági megkeresés esetén adatokat adhat át az arra jogosult szerveknek.",
        "description": "Data disclosure for legal compliance",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 18,
        "text": "A szerződés elektronikus archiválása megfelel az eIDAS rendelet és a vonatkozó magyar jogszabályok követelményeinek.",
        "description": "eIDAS and electronic archiving reference",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 19,
        "text": "A jelen megállapodás teljes egészében kifejezi a felek akaratát, és minden korábbi szóbeli vagy írásbeli megállapodást hatályon kívül helyez.",
        "description": "Entire agreement clause",
        "expected_difficulty": "very_hard",
    },
    {
        "id": 20,
        "text": "A felek megállapodnak abban, hogy a szerződésből eredő jogviták rendezése során a közvetítés igénybevételét a peres eljárást megelőzően kötelezően megkísérlik.",
        "description": "Mandatory mediation clause",
        "expected_difficulty": "very_hard",
    },
]


# ============================
# PREDICTOR
# ============================

class ReadabilityPredictor:
    def __init__(self, cfg: Optional[InferenceConfig] = None):
        self.cfg = cfg or InferenceConfig()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_classes,
            problem_type="single_label_classification",
        )

        self._load_weights()

    def _load_weights(self) -> None:
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.cfg.model_path}")

        checkpoint = torch.load(self.cfg.model_path, map_location="cpu")

        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint must contain 'model_state_dict'")

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model weights loaded successfully")

    # ----------------------------
    # CORE INFERENCE
    # ----------------------------

    def _predict_logits(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            return outputs.logits

    # ----------------------------
    # PUBLIC API
    # ----------------------------

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        logits = self._predict_logits(texts)
        probs = torch.softmax(logits, dim=1)

        results: List[Dict[str, Any]] = []

        for i, (text, prob_vec) in enumerate(zip(texts, probs)):
            internal_label = int(torch.argmax(prob_vec).item())  # 0–4
            user_label = internal_label + 1  # 1–5

            results.append({
                "index": i,
                "text": text[:120] + ("..." if len(text) > 120 else ""),
                "predicted_label": user_label,
                "label_description": self.cfg.label_descriptions[user_label],
                "confidence": float(prob_vec.max().item()),
                "probabilities": {
                    j + 1: float(p) for j, p in enumerate(prob_vec.cpu().tolist())
                },
            })

        return results


# ============================
# CLI / DEMO
# ============================

def run_demo() -> None:
    logger.info("Starting inference demo")

    predictor = ReadabilityPredictor()
    texts = [s["text"] for s in SAMPLE_TEXTS]

    results = predictor.predict(texts)

    # Local summary aggregator (no TensorBoard dependency)
    summary = defaultdict(
        lambda: {
            "count": 0,
            "conf_sum": 0.0,
            "min_conf": float("inf"),
            "max_conf": float("-inf"),
            "pred_dist": Counter(),
            "correct": 0,
            "wrong": 0,
            "conf_sum_correct": 0.0,
            "conf_sum_wrong": 0.0,
        }
    )

    # Overall (global) summary across all samples
    overall = {
        "count": 0,
        "conf_sum": 0.0,
        "min_conf": float("inf"),
        "max_conf": float("-inf"),
        "pred_dist": Counter(),
        "correct": 0,
        "wrong": 0,
        "conf_sum_correct": 0.0,
        "conf_sum_wrong": 0.0,
    }

    for r, s in zip(results, SAMPLE_TEXTS):
        logger.info("-")

        expected = (
            s.get("expected")
            if isinstance(s, dict)
            else None
        )
        if expected is None and isinstance(s, dict):
            expected = s.get("expected_difficulty", "N/A")

        predicted_label = int(r["predicted_label"])
        conf = float(r["confidence"])

        # expected_label számolása a megadott mapping alapján (ha lehet)
        expected_label: Optional[int] = None
        if isinstance(expected, str):
            expected_label = predictor.cfg.expected_difficulty_to_label.get(expected)

        is_correct: Optional[bool] = None
        if expected_label is not None:
            is_correct = (predicted_label == expected_label)

        logger.info(
            f"Text {r['index'] + 1} | expected: {expected} | expected_label: {expected_label}"
        )
        logger.info(
            f"Predicted: {predicted_label} – {r['label_description']} "
            f"({predictor.cfg.label_to_difficulty_tag.get(predicted_label, 'N/A')})"
        )
        logger.info(f"Confidence: {conf:.2%}")
        logger.info(", ".join(f"L{k}:{v:.1%}" for k, v in r["probabilities"].items()))

        # summary update (category + overall)
        key = str(expected)
        row = summary[key]

        for target in (row, overall):
            target["count"] += 1
            target["conf_sum"] += conf
            target["min_conf"] = min(target["min_conf"], conf)
            target["max_conf"] = max(target["max_conf"], conf)
            target["pred_dist"][predicted_label] += 1

            if is_correct is True:
                target["correct"] += 1
                target["conf_sum_correct"] += conf
            elif is_correct is False:
                target["wrong"] += 1
                target["conf_sum_wrong"] += conf

    # ===== SUMMARY a végén (confidence alapú) =====
    logger.info("-")
    logger.info("SUMMARY (confidence-based, by expected category)")

    def _fmt_dist(c: Counter) -> str:
        return ", ".join(f"L{lab}:{c.get(lab, 0)}" for lab in range(1, 6))

    for category in sorted(summary.keys()):
        row = summary[category]
        n = row["count"]
        avg_conf = row["conf_sum"] / n if n else 0.0

        # Accuracy csak akkor értelmezhető, ha van expected_label mapping (correct+wrong > 0)
        judged = row["correct"] + row["wrong"]
        acc = (row["correct"] / judged) if judged else None

        avg_conf_correct = (row["conf_sum_correct"] / row["correct"]) if row["correct"] else None
        avg_conf_wrong = (row["conf_sum_wrong"] / row["wrong"]) if row["wrong"] else None

        min_conf = row["min_conf"] if row["min_conf"] != float("inf") else None
        max_conf = row["max_conf"] if row["max_conf"] != float("-inf") else None

        logger.info(
            f"  {category}: n={n} | "
            f"acc={(f'{acc:.2%}' if acc is not None else 'N/A')} | "
            f"avg_conf={avg_conf:.2%} | "
            f"avg_conf_correct={(f'{avg_conf_correct:.2%}' if avg_conf_correct is not None else 'N/A')} | "
            f"avg_conf_wrong={(f'{avg_conf_wrong:.2%}' if avg_conf_wrong is not None else 'N/A')} | "
            f"min_conf={(f'{min_conf:.2%}' if min_conf is not None else 'N/A')} | "
            f"max_conf={(f'{max_conf:.2%}' if max_conf is not None else 'N/A')} | "
            f"{_fmt_dist(row['pred_dist'])}"
        )

    # ===== OVERALL SUMMARY =====
    logger.info("-")
    logger.info("SUMMARY (confidence-based, OVERALL)")

    n = overall["count"]
    avg_conf = overall["conf_sum"] / n if n else 0.0

    judged = overall["correct"] + overall["wrong"]
    acc = (overall["correct"] / judged) if judged else None

    avg_conf_correct = (overall["conf_sum_correct"] / overall["correct"]) if overall["correct"] else None
    avg_conf_wrong = (overall["conf_sum_wrong"] / overall["wrong"]) if overall["wrong"] else None

    min_conf = overall["min_conf"] if overall["min_conf"] != float("inf") else None
    max_conf = overall["max_conf"] if overall["max_conf"] != float("-inf") else None

    logger.info(
        f"  ALL: n={n} | "
        f"acc={(f'{acc:.2%}' if acc is not None else 'N/A')} | "
        f"avg_conf={avg_conf:.2%} | "
        f"avg_conf_correct={(f'{avg_conf_correct:.2%}' if avg_conf_correct is not None else 'N/A')} | "
        f"avg_conf_wrong={(f'{avg_conf_wrong:.2%}' if avg_conf_wrong is not None else 'N/A')} | "
        f"min_conf={(f'{min_conf:.2%}' if min_conf is not None else 'N/A')} | "
        f"max_conf={(f'{max_conf:.2%}' if max_conf is not None else 'N/A')} | "
        f"{_fmt_dist(overall['pred_dist'])}"
    )
if __name__ == "__main__":
    run_demo()
