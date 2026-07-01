import re
from dataclasses import dataclass
from typing import List, Tuple

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s.,!?]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Basic text cleaning tuned for social captions.
    - lowercasing
    - remove urls, mentions, hashtags
    - strip non-alnum clutter
    - collapse whitespace
    """
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


@dataclass
class TextBatch:
    input_ids: List[int]
    attention_mask: List[int]
    tokens: List[str]


def tokenize(tokenizer, text: str, max_length: int = 128) -> TextBatch:
    """
    Tokenize with a Hugging Face tokenizer. Returns a lightweight dataclass to
    avoid pulling torch inside preprocessing.
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    return TextBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokens=tokens,
    )


def preprocess_and_tokenize(tokenizer, raw_text: str, max_length: int = 128) -> Tuple[str, TextBatch]:
    cleaned = clean_text(raw_text)
    batch = tokenize(tokenizer, cleaned, max_length=max_length)
    return cleaned, batch
