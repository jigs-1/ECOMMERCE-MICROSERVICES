import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from cameo.core.preprocess.text import clean_text
from cameo.core.preprocess.image import preprocess_image


@dataclass
class Sample:
    text: str
    image_path: str
    emotion: Optional[int] = None
    intensity: Optional[float] = None
    intent: Optional[int] = None


class MultimodalDataset(Dataset):
    """
    CSV manifest columns: text,image_path,emotion,intensity,intent
    emotion/intensity/intent are optional (for inference data).
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        max_length: int = 128,
        image_size: int = 224,
        root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.root = Path(root) if root else None
        self.samples: List[Sample] = self._load_manifest(manifest_path)

    def _load_manifest(self, path: str) -> List[Sample]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    Sample(
                        text=row["text"],
                        image_path=row["image_path"],
                        emotion=int(row["emotion"]) if row.get("emotion") not in (None, "", "None") else None,
                        intensity=float(row["intensity"]) if row.get("intensity") not in (None, "", "None") else None,
                        intent=int(row["intent"]) if row.get("intent") not in (None, "", "None") else None,
                    )
                )
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_clean = clean_text(sample.text)
        toks = self.tokenizer(
            text_clean,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}

        img_path = Path(sample.image_path)
        if self.root:
            img_path = self.root / img_path
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess_image(img, image_size=self.image_size)

        item: Dict[str, Any] = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "image": img_tensor,
        }
        if sample.emotion is not None:
            item["emotion"] = torch.tensor(sample.emotion, dtype=torch.long)
        if sample.intensity is not None:
            item["intensity"] = torch.tensor(sample.intensity, dtype=torch.float)
        if sample.intent is not None:
            item["intent"] = torch.tensor(sample.intent, dtype=torch.long)
        return item
