"""Character offsets for the original generated IDs, separate from model replay."""
from __future__ import annotations

from operator import index


NO_OFFSET = (0, 0)
# Prefix decoding is quadratic in the number of checked IDs. Cap that work
# for long continuations while keeping an offset entry for every original ID.
_MAX_PREFIX_TOKENS = 256


def tokenizer_special_ids(tokenizer) -> set[int]:
    """Include special added tokens that may be absent from all_special_ids."""
    special_ids = {int(token_id) for token_id in (getattr(tokenizer, "all_special_ids", None) or ())}
    added_tokens = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special_ids.update(int(token_id) for token_id, token in added_tokens.items() if getattr(token, "special", False))
    return special_ids


def _matching_offsets(encoded, row: int, ids: list[int], text: str) -> list[tuple[int, int]] | None:
    if encoded is None:
        return None
    try:
        if list(encoded["input_ids"][row]) != ids:
            return None
        offsets = [(index(start), index(end)) for start, end in encoded["offset_mapping"][row]]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if len(offsets) != len(ids):
        return None
    return [(start, end) if 0 <= start < end <= len(text) else NO_OFFSET for start, end in offsets]


def _decode(tokenizer, ids: list[int]) -> str | None:
    try:
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError, OverflowError):
        return None


def _prefix_offsets(tokenizer, ids: list[int], positions: list[int], text: str) -> dict[int, tuple[int, int]]:
    if _decode(tokenizer, ids) != text:
        return {}

    spans = {}
    previous_end: int | None = 0
    furthest_end = 0
    for position in positions:
        if position >= _MAX_PREFIX_TOKENS:
            break
        prefix = text if position == len(ids) - 1 else _decode(tokenizer, ids[:position + 1])
        # Incomplete UTF-8 often decodes to U+FFFD. A decoder may also omit the
        # incomplete bytes entirely, leaving the previous boundary unchanged.
        if prefix is None or prefix.endswith("\ufffd") or not text.startswith(prefix):
            previous_end = None
            continue
        end = len(prefix)
        if end < furthest_end:
            return {}
        if end == furthest_end:
            previous_end = None
            continue
        if previous_end is not None and "\ufffd" not in text[previous_end:end]:
            spans[position] = (previous_end, end)
        furthest_end = end
        previous_end = end
    return spans


def align_generated_token_offsets(
    tokenizer,
    generated_ids: list[list[int]],
    generated_texts: list[str],
) -> list[list[tuple[int, int]]]:
    """Return one character span per original ID in each decoded continuation.

    Fast-tokenizer offsets are used only when re-encoding yields exactly the
    original IDs after omitting tokenizer specials. These offsets may share a
    span when several byte tokens encode one character. Otherwise, the text
    must match decoding the original IDs, and a token needs two adjacent
    verified character boundaries to receive a span. Special IDs, ambiguous
    fallback spans, and tokens beyond the fallback budget receive (0, 0).
    """
    if len(generated_ids) != len(generated_texts):
        raise ValueError("Generated IDs and texts must have equal batch lengths.")

    special_ids = tokenizer_special_ids(tokenizer)
    spans = [[NO_OFFSET] * len(ids) for ids in generated_ids]
    pending = []
    for row, (ids, text) in enumerate(zip(generated_ids, generated_texts)):
        if text:
            positions = [position for position, token_id in enumerate(ids) if token_id not in special_ids]
            if positions:
                pending.append((row, positions))
    if not pending:
        return spans

    encoded = None
    if getattr(tokenizer, "is_fast", True):
        try:
            encoded = tokenizer(
                [generated_texts[row] for row, _ in pending],
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
        except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
            pass

    for encoded_row, (row, positions) in enumerate(pending):
        ids, text = generated_ids[row], generated_texts[row]
        offsets = _matching_offsets(encoded, encoded_row, [ids[position] for position in positions], text)
        if offsets is not None:
            for position, span in zip(positions, offsets):
                spans[row][position] = span
        else:
            for position, span in _prefix_offsets(tokenizer, ids, positions, text).items():
                spans[row][position] = span
    return spans
