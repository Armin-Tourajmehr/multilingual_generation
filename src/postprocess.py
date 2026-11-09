"""CPU-only POS annotation of previously saved token posterior files."""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import stat
import tempfile
from threading import Thread
from time import monotonic
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .lexical import StanzaTagger, align_pos_to_token


ANNOTATION_COLUMNS = (
    "word_text", "upos", "xpos", "feats", "sentence_id", "word_id", "pos_status",
)
SUMMARY_COLUMNS = (
    "model", "input_language", "layer", "upos", "pos_status",
    "posterior_language", "num_tokens", "mean_posterior",
)


@dataclass(frozen=True)
class AnnotationResult:
    model: str
    language: str
    num_rows: int
    num_samples: int
    token_path: Path
    summary_path: Path | None
    status: str = "annotated"


def _temporary_sibling(path: Path, suffix: str = ".tmp") -> Path:
    handle, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(handle)
    return Path(filename)


def _backup_sibling(path: Path) -> Path:
    """Keep the original inode when possible, copying only if links are unavailable."""
    backup = _temporary_sibling(path, suffix=".backup")
    try:
        backup.unlink()
        try:
            os.link(path, backup)
        except OSError:
            shutil.copy2(path, backup)
        return backup
    except BaseException:
        with suppress(FileNotFoundError):
            backup.unlink()
        raise


def _commit_annotations(token_temp: Path, token_path: Path, summary_temp: Path | None, summary_path: Path | None) -> None:
    """Commit both products, restoring the original token file if the CSV commit fails."""
    os.chmod(token_temp, stat.S_IMODE(token_path.stat().st_mode))
    backup = _backup_sibling(token_path) if summary_temp is not None else None
    preserve_backup = False
    try:
        os.replace(token_temp, token_path)
        if summary_temp is not None:
            try:
                os.replace(summary_temp, summary_path)
            except BaseException as commit_error:
                try:
                    os.replace(backup, token_path)
                except BaseException as rollback_error:
                    preserve_backup = True
                    raise RuntimeError(
                        f"Replacing summary {summary_path} failed ({commit_error}); restoring "
                        f"{token_path} also failed ({rollback_error}). Original token file "
                        f"preserved at {backup}; restore it before retrying."
                    ) from rollback_error
                raise
    finally:
        if backup is not None and not preserve_backup:
            with suppress(FileNotFoundError):
                backup.unlink()


def _generated_texts(path: Path, batch_size: int) -> dict[Any, str]:
    if not path.exists():
        raise FileNotFoundError(f"Generated output file is required to annotate token rows: {path}")
    source = pq.ParquetFile(path)
    try:
        missing = {"sample_id", "generated_text"}.difference(source.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        texts: dict[Any, str] = {}
        for batch in source.iter_batches(batch_size=batch_size, columns=["sample_id", "generated_text"]):
            for sample_id, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if sample_id is None:
                    raise ValueError(f"{path} contains a null sample_id")
                if sample_id in texts:
                    raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
                texts[sample_id] = "" if text is None else str(text)
        return texts
    finally:
        source.close()


def _write_summary(
    path: Path,
    model: str,
    language: str,
    languages: list[str],
    buckets: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for (layer, upos, status), bucket in sorted(buckets.items()):
            for index, candidate in enumerate(languages):
                writer.writerow({
                    "model": model,
                    "input_language": language,
                    "layer": layer,
                    "upos": upos,
                    "pos_status": status,
                    "posterior_language": candidate,
                    "num_tokens": bucket["n"],
                    "mean_posterior": bucket["sums"][index] / bucket["n"],
                })


def annotate_token_file(
    token_path: str | Path,
    generated_path: str | Path,
    *,
    model: str,
    language: str,
    languages: list[str],
    lexical_config: dict[str, Any],
    summary_path: str | Path | None = None,
    batch_size: int = 65536,
    tagger_factory: Callable[[dict[str, Any]], Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnnotationResult:
    """Replace only POS fields after a complete, successful streamed annotation.

    The tagger and span caches are local to one model/language file. Numeric
    posterior arrays and every other column are passed unchanged to the writer.
    A factory can supply a lightweight tagger for offline checks.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    token_path = Path(token_path)
    generated_path = Path(generated_path)
    summary_path = Path(summary_path) if summary_path is not None else None
    if not token_path.exists():
        return AnnotationResult(model, language, 0, 0, token_path, None, "no_token_file")

    source = pq.ParquetFile(token_path)
    writer = None
    tagger = None
    token_temp = None
    summary_temp = None
    num_rows = 0
    tagged_samples: dict[Any, tuple[list[dict[str, Any]], str]] = {}
    aligned_spans: dict[tuple[Any, int, int], dict[str, Any]] = {}
    buckets: dict[tuple[int, str, str], dict[str, Any]] = {}
    posterior_columns = [f"P_{candidate}" for candidate in languages]
    try:
        schema = source.schema_arrow
        required = {"sample_id", "char_start", "char_end", *ANNOTATION_COLUMNS}
        if summary_path is not None:
            required.update(["layer", *posterior_columns])
        missing = required.difference(schema.names)
        if missing:
            raise ValueError(f"{token_path} is missing required columns: {sorted(missing)}")
        total_rows = source.metadata.num_rows
        if progress is not None:
            progress(f"[{model}/{language}] Starting POS annotation of {total_rows:,} token rows.")
            next_progress = monotonic() + 10
        texts = _generated_texts(generated_path, batch_size) if total_rows else {}
        token_temp = _temporary_sibling(token_path)
        writer = pq.ParquetWriter(token_temp, schema, compression="zstd")
        factory = tagger_factory if tagger_factory is not None else StanzaTagger

        for batch in source.iter_batches(batch_size=batch_size):
            sample_ids = batch.column(schema.get_field_index("sample_id")).to_pylist()
            starts = batch.column(schema.get_field_index("char_start")).to_pylist()
            ends = batch.column(schema.get_field_index("char_end")).to_pylist()
            replacements: dict[str, list[Any]] = {name: [] for name in ANNOTATION_COLUMNS}
            if summary_path is not None:
                layers = batch.column(schema.get_field_index("layer")).to_pylist()
                posterior_values = [batch.column(schema.get_field_index(name)).to_pylist() for name in posterior_columns]

            for row_index, (sample_id, start, end) in enumerate(zip(sample_ids, starts, ends)):
                if progress is not None and row_index % 1024 == 0:
                    now = monotonic()
                    if now >= next_progress:
                        processed = num_rows + row_index
                        progress(f"[{model}/{language}] Processed {processed:,}/{total_rows:,} token rows.")
                        next_progress = now + 10
                if sample_id not in texts:
                    raise ValueError(
                        f"{generated_path} has no generated text for sample_id {sample_id!r} "
                        f"referenced by {token_path}"
                    )
                if sample_id not in tagged_samples:
                    text = texts[sample_id]
                    if text.strip():
                        if tagger is None:
                            tagger = factory(dict(lexical_config))
                        tagged_samples[sample_id] = tagger.tag_text(text, language)
                    else:
                        tagged_samples[sample_id] = ([], "empty")
                key = (sample_id, int(start), int(end))
                if key not in aligned_spans:
                    words, status = tagged_samples[sample_id]
                    aligned_spans[key] = align_pos_to_token(texts[sample_id], key[1], key[2], words, status)
                annotation = aligned_spans[key]
                for name in ANNOTATION_COLUMNS:
                    replacements[name].append(annotation[name])

                if summary_path is not None:
                    bucket_key = (int(layers[row_index]), str(annotation["upos"]), str(annotation["pos_status"]))
                    bucket = buckets.setdefault(bucket_key, {"n": 0, "sums": [0.0] * len(languages)})
                    bucket["n"] += 1
                    for index, values in enumerate(posterior_values):
                        bucket["sums"][index] += float(values[row_index])

            updated = batch
            for name in ANNOTATION_COLUMNS:
                index = schema.get_field_index(name)
                updated = updated.set_column(index, schema.field(index), pa.array(replacements[name], type=schema.field(index).type))
            writer.write_batch(updated)
            num_rows += batch.num_rows

        # Finish both products and release all readers/taggers before committing.
        # Any tagging, conversion or writer failure leaves the source untouched.
        completed_writer, writer = writer, None
        completed_writer.close()
        if tagger is not None:
            completed_tagger, tagger = tagger, None
            completed_tagger.close()
        source.close()
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_temp = _temporary_sibling(summary_path)
            _write_summary(summary_temp, model, language, languages, buckets)
        _commit_annotations(token_temp, token_path, summary_temp, summary_path)
        token_temp = None
        summary_temp = None
        return AnnotationResult(model, language, num_rows, len(tagged_samples), token_path, summary_path)
    finally:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        if tagger is not None:
            with suppress(Exception):
                tagger.close()
        with suppress(Exception):
            source.close()
        for temporary in (token_temp, summary_temp):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()


def _annotate_partition(
    task: tuple[Path, str, str],
    *,
    languages: list[str],
    lexical: dict[str, Any],
    batch_size: int,
    tagger_factory: Callable[[dict[str, Any]], Any] | None,
    progress: Callable[[str], None] | None,
) -> AnnotationResult:
    model_root, model, language = task
    summary = None
    if lexical.get("save_pos_posterior_summary", True):
        summary = model_root / "lexical_analysis" / "pos_posterior_summary" / f"{language}.csv"
    result = annotate_token_file(
        model_root / "token_posteriors" / f"{language}.parquet",
        model_root / "generated_outputs" / f"{language}.parquet",
        model=model, language=language, languages=languages, lexical_config=lexical,
        summary_path=summary, batch_size=batch_size, tagger_factory=tagger_factory, progress=progress,
    )
    if progress is not None:
        if result.status == "no_token_file":
            progress(f"[{model}/{language}] No token file, skipped.")
        else:
            progress(f"[{model}/{language}] Annotated {result.num_rows:,} token rows from {result.num_samples:,} samples.")
    return result


_worker_messages = None
_stanza_resource_lock = None


def _initialize_worker(messages, resource_lock) -> None:
    global _worker_messages, _stanza_resource_lock
    _worker_messages = messages
    _stanza_resource_lock = resource_lock
    # Stanza imports PyTorch lazily. Keep its math threads and Arrow's pools
    # small so each partition worker can make progress on its own core.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _worker_tagger(cfg: dict[str, Any]) -> StanzaTagger:
    return StanzaTagger(cfg, resource_lock=_stanza_resource_lock)


def _run_worker(
    annotate_partition: Callable[..., AnnotationResult], task: tuple[Path, str, str],
) -> AnnotationResult:
    return annotate_partition(task, progress=_worker_messages.put if _worker_messages is not None else None)


def _forward_progress(messages, progress: Callable[[str], None], errors: list[BaseException]) -> None:
    for message in iter(messages.get, None):
        if not errors:
            try:
                progress(message)
            except BaseException as exc:
                # Keep draining the pipe while the workers finish their current files.
                errors.append(exc)


def _annotate_parallel(
    tasks: list[tuple[Path, str, str]],
    annotate_partition: Callable[..., AnnotationResult],
    workers: int,
    progress: Callable[[str], None] | None,
    allow_downloads: bool,
) -> list[AnnotationResult]:
    # Spawn fresh processes rather than inheriting Arrow or PyTorch threads.
    context = get_context("spawn")
    # Pipeline initialization can download into the shared Stanza cache.
    resource_lock = context.Lock() if allow_downloads else None
    messages = context.SimpleQueue() if progress is not None else None
    progress_errors: list[BaseException] = []
    forwarder = None
    results: dict[int, AnnotationResult] = {}
    if progress is not None:
        forwarder = Thread(target=_forward_progress, args=(messages, progress, progress_errors))
        forwarder.start()
    try:
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
            initializer=_initialize_worker, initargs=(messages, resource_lock),
        )
        try:
            pending = {executor.submit(_run_worker, annotate_partition, task): i for i, task in enumerate(tasks)}
            for future in as_completed(pending):
                if progress_errors:
                    raise progress_errors[0]
                results[pending[future]] = future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        if forwarder is not None:
            messages.put(None)
            forwarder.join()
            messages.close()
    if progress_errors:
        raise progress_errors[0]
    return [results[index] for index in range(len(tasks))]


def annotate_outputs(
    cfg: dict[str, Any],
    *,
    models: list[str] | None = None,
    batch_size: int = 65536,
    tagger_factory: Callable[[dict[str, Any]], Any] | None = None,
    progress: Callable[[str], None] | None = None,
    workers: int = 1,
) -> list[AnnotationResult]:
    """Annotate existing outputs, optionally in parallel by model and language."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    configured = cfg["models"]
    selected = list(models) if models is not None else [name for name, spec in configured.items() if spec.get("enabled", False)]
    unknown = set(selected).difference(configured)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Model keys must not be duplicated")
    repo_root = Path(cfg.get("_repo_root", Path.cwd()))
    output_root = Path(cfg["outputs"]["root_dir"])
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    result_root = output_root / cfg["outputs"]["result_dir"]
    lexical = dict(cfg.get("lexical_analysis", {}))
    stanza_root = Path(lexical.get("stanza_model_dir", "outputs/stanza_resources"))
    if not stanza_root.is_absolute():
        stanza_root = repo_root / stanza_root
    lexical["stanza_model_dir"] = str(stanza_root)
    languages = list(cfg["languages"])
    tasks = [(result_root / model, model, language) for model in selected for language in languages]
    workers = min(workers, len(tasks))
    if workers > 1 and len(languages) != len(set(languages)):
        raise ValueError("Languages must be unique for parallel annotation.")
    annotate_partition = partial(
        _annotate_partition, languages=languages, lexical=lexical, batch_size=batch_size,
        tagger_factory=_worker_tagger if workers > 1 and tagger_factory is None else tagger_factory,
    )
    if workers <= 1:
        return [annotate_partition(task, progress=progress) for task in tasks]
    return _annotate_parallel(tasks, annotate_partition, workers, progress, lexical.get("download_missing_models", False))
