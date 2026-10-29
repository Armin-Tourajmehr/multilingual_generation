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
                raise RuntimeError(
                    "The 'stanza' package is not installed. "
                    "Install requirements.txt."
                ) from exc
            self._stanza = stanza

        self._model_dir.mkdir(parents=True, exist_ok=True)

        allow_download = bool(
            self.cfg.get("download_missing_models", False)
        )

        kwargs = {
            "lang": language,
            "processors": {
                "tokenize": "default",
                "pos": "default",
            },
            "package": None,
            "dir": str(self._model_dir),
            "use_gpu": False,
            "verbose": False,
            "download_method": (
                "reuse_resources" if allow_download else None
            ),
        }

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
            word_idx = 0

            # Keep the original surface-token offsets. A Stanza Word created
            # by MWT expansion may not exist literally in the original text.
            for token in sentence.tokens:
                token_start = getattr(token, "start_char", None)
                token_end = getattr(token, "end_char", None)

                try:
                    token_start = (
                        int(token_start)
                        if token_start is not None
                        else -1
                    )
                    token_end = (
                        int(token_end)
                        if token_end is not None
                        else -1
                    )
                except (TypeError, ValueError):
                    token_start, token_end = -1, -1

                token_has_span = (
                    token_start >= 0 and token_end > token_start
                )
                token_cursor = (
                    token_start if token_has_span else search_cursor
                )

                token_words = list(getattr(token, "words", []) or [])

                for local_idx, word in enumerate(token_words):
                    word_text = str(getattr(word, "text", ""))

                    start = getattr(word, "start_char", None)
                    end = getattr(word, "end_char", None)

                    try:
                        start = int(start) if start is not None else -1
                        end = int(end) if end is not None else -1
                    except (TypeError, ValueError):
                        start, end = -1, -1

                    valid_span = start >= 0 and end > start

                    # First try Stanza's original word offsets.
                    if not valid_span and token_has_span and word_text:
                        search_from = max(token_start, token_cursor)
                        found = text.find(
                            word_text,
                            search_from,
                            token_end,
                        )

                        if (
                            found >= 0
                            and found + len(word_text) <= token_end
                        ):
                            start = found
                            end = found + len(word_text)
                            valid_span = True

                    # MWT-expanded text may not occur literally in the
                    # source. Use the remaining parent-token span, stopping
                    # at the next valid child-word boundary when available.
                    if not valid_span and token_has_span:
                        search_from = max(token_start, token_cursor)
                        next_start = None

                        for future in token_words[local_idx + 1:]:
                            future_start = getattr(
                                future, "start_char", None
                            )
                            future_end = getattr(
                                future, "end_char", None
                            )

                            try:
                                future_start = (
                                    int(future_start)
                                    if future_start is not None
                                    else -1
                                )
                                future_end = (
                                    int(future_end)
                                    if future_end is not None
                                    else -1
                                )
                            except (TypeError, ValueError):
                                continue

                            if (
                                future_start >= search_from
                                and future_end > future_start
                                and future_start < token_end
                            ):
                                next_start = future_start
                                break

                        approximate_end = (
                            next_start
                            if next_start is not None
                            else token_end
                        )

                        if approximate_end > search_from:
                            start = search_from
                            end = approximate_end
                            valid_span = True

                    # Last fallback for words without a usable parent span.
                    if not valid_span:
                        start, end = _find_span(
                            text, word_text, search_cursor
                        )

                    if start >= 0 and end > start:
                        token_cursor = max(token_cursor, end)
                        search_cursor = max(search_cursor, end)

                    words.append({
                        "sentence_id": sent_idx,
                        "word_id": word_idx,
                        "word_text": word_text,
                        "char_start": start if start >= 0 else -1,
                        "char_end": end if end > start else -1,
                        "upos": str(
                            getattr(word, "upos", None) or "X"
                        ),
                        "xpos": str(
                            getattr(word, "xpos", None) or ""
                        ),
                        "feats": str(
                            getattr(word, "feats", None) or ""
                        ),
                    })

                    word_idx += 1

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
