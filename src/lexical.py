from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any


class StanzaTagger:
    """Load at most one Stanza language pipeline at a time to limit RAM use."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._pipe = None
        self._language: str | None = None
        self._unavailable: set[str] = set()
        self._stanza = None
        self._model_dir = Path(cfg.get("stanza_model_dir", "outputs/stanza_resources"))

    def _build_pipeline(self, language: str):
        if self._stanza is None:
            try:
                import stanza
            except ImportError as exc:
                raise RuntimeError("The 'stanza' package is not installed. Install requirements.txt.") from exc
            self._stanza = stanza

        kwargs = {
            "lang": language,
            "processors": "tokenize,pos",
            "dir": str(self._model_dir),
            "use_gpu": False,
            "verbose": False,
            "download_method": None,
        }
        try:
            return self._stanza.Pipeline(**kwargs)
        except Exception as first_error:
            if not self.cfg.get("download_missing_models", False):
                raise first_error
            self._model_dir.mkdir(parents=True, exist_ok=True)
            # Explicit processor package selection avoids downloading unrelated
            # dependency, NER and sentiment models.
            self._stanza.download(
                lang=language,
                model_dir=str(self._model_dir),
                processors={"tokenize": "default", "pos": "default"},
                verbose=False,
            )
            return self._stanza.Pipeline(**kwargs)

    def _ensure_pipeline(self, language: str):
        if self._language == language and self._pipe is not None:
            return self._pipe
        self.close()
        if language in self._unavailable:
            return None
        try:
            self._pipe = self._build_pipeline(language)
            self._language = language
            return self._pipe
        except Exception as exc:
            message = f"Stanza POS pipeline unavailable for language {language}: {exc}"
            if self.cfg.get("strict", False):
                raise RuntimeError(message) from exc
            warnings.warn(message + "; POS labels will be marked POS_UNAVAILABLE.", RuntimeWarning)
            self._unavailable.add(language)
            return None

    def tag_text(self, text: str, language: str) -> tuple[list[dict[str, Any]], str]:
        pipe = self._ensure_pipeline(language)
        if pipe is None:
            return [], "unavailable"
        try:
            doc = pipe(text)
        except Exception as exc:
            if self.cfg.get("strict", False):
                raise RuntimeError(f"Stanza failed for language {language}: {exc}") from exc
            return [], "error"

        words: list[dict[str, Any]] = []
        search_cursor = 0
        for sent_idx, sentence in enumerate(doc.sentences):
            for word_idx, word in enumerate(sentence.words):
                word_text = str(getattr(word, "text", ""))
                start = getattr(word, "start_char", None)
                end = getattr(word, "end_char", None)
                if start is None or end is None:
                    start, end = _find_span(text, word_text, search_cursor)
                if start is not None and end is not None and end > start:
                    search_cursor = int(end)
                words.append({
                    "sentence_id": sent_idx,
                    "word_id": word_idx,
                    "word_text": word_text,
                    "char_start": int(start) if start is not None else -1,
                    "char_end": int(end) if end is not None else -1,
                    "upos": str(getattr(word, "upos", None) or "X"),
                    "xpos": str(getattr(word, "xpos", None) or ""),
                    "feats": str(getattr(word, "feats", None) or ""),
                })
        return words, "ok"

    def close(self) -> None:
        self._pipe = None
        self._language = None
        gc.collect()


def _find_span(text: str, word: str, start_at: int) -> tuple[int, int]:
    if not word:
        return -1, -1
    start = text.find(word, max(0, start_at))
    if start < 0:
        start = text.find(word)
    return (start, start + len(word)) if start >= 0 else (-1, -1)


def align_pos_to_token(
    text: str,
    start: int,
    end: int,
    words: list[dict[str, Any]],
    tag_status: str,
) -> dict[str, Any]:
    """Assign the Stanza word with the largest character-span overlap to a model token."""
    if tag_status != "ok":
        return {
            "word_text": "",
            "upos": "POS_UNAVAILABLE",
            "xpos": "",
            "feats": "",
            "sentence_id": -1,
            "word_id": -1,
            "pos_status": tag_status,
        }
    if end <= start:
        return {
            "word_text": "",
            "upos": "SPECIAL_OR_EMPTY_OFFSET",
            "xpos": "",
            "feats": "",
            "sentence_id": -1,
            "word_id": -1,
            "pos_status": "no_offset",
        }

    best = None
    best_overlap = 0
    for word in words:
        word_start, word_end = int(word["char_start"]), int(word["char_end"])
        if word_start < 0 or word_end <= word_start:
            continue
        overlap = max(0, min(end, word_end) - max(start, word_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = word
    if best is None:
        return {
            "word_text": text[start:end],
            "upos": "NO_WORD_ALIGNMENT",
            "xpos": "",
            "feats": "",
            "sentence_id": -1,
            "word_id": -1,
            "pos_status": "unaligned",
        }
    return {
        "word_text": best["word_text"],
        "upos": best["upos"],
        "xpos": best["xpos"],
        "feats": best["feats"],
        "sentence_id": best["sentence_id"],
        "word_id": best["word_id"],
        "pos_status": "ok",
    }


def pos_posterior_summary(token_rows: list[dict[str, Any]], languages: list[str]) -> list[dict[str, Any]]:
    """Aggregate token posterior by layer and POS; retain one row per posterior language."""
    if not token_rows:
        return []
    posterior_columns = [f"P_{language}" for language in languages]
    buckets: dict[tuple[int, str, str], dict[str, float]] = {}
    for row in token_rows:
        pos = str(row.get("upos", "X"))
        layer = int(row["layer"])
        key = (layer, pos, str(row.get("pos_status", "unknown")))
        bucket = buckets.setdefault(key, {"n": 0.0, **{col: 0.0 for col in posterior_columns}})
        bucket["n"] += 1.0
        for col in posterior_columns:
            bucket[col] += float(row.get(col, 0.0))

    summary: list[dict[str, Any]] = []
    for (layer, pos, status), values in sorted(buckets.items()):
        n = int(values["n"])
        for language in languages:
            col = f"P_{language}"
            summary.append({
                "layer": layer,
                "upos": pos,
                "pos_status": status,
                "posterior_language": language,
                "num_tokens": n,
                "mean_posterior": values[col] / max(n, 1),
            })
    return summary
