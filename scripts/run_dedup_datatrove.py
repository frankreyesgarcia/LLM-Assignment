#!/usr/bin/env python3
"""Cross-source MinHash dedup for pt-fineweb2 / es-fineweb2 / hi-fineweb2 /
hi-sangraha, using `datatrove` (HuggingFace's own pipeline library --
literally what built fineweb-2 itself) instead of a from-scratch dedup
implementation.

Why datatrove: our previous from-scratch pipeline (custom SourceAdapter +
ExactDeduper/SqliteNearDeduper) processed one source with exactly one
single-threaded worker, no matter how many cores/nodes were available --
measured ~30 docs/sec for fineweb2 and ~4.25 docs/sec for hi-sangraha's
OCR'd PDF text, putting a full run at ~46+ hours even with filtering
skipped. datatrove's `SlurmPipelineExecutor` shards each stage across a
real SLURM job array (`--tasks`), which is genuine horizontal scaling
across nodes, not just cores on one machine.

No filtering (language/quality/cleaning) is applied here at all -- Task 1's
custom filter pipeline was removed along with the other 15 sources this
pipeline used to support. These 4 sources are used as-is; dedup is the
only cleaning step.

Runs the standard 4-stage MinHash flow (datatrove's own
examples/minhash_deduplication.py, verified against the installed
package's actual signatures, not assumed) once per language:

  1. MinhashDedupSignature -- compute MinHash signatures per document.
  2. MinhashDedupBuckets   -- find matching signatures within each LSH bucket.
  3. MinhashDedupCluster   -- build duplicate clusters across all buckets
                              (single task -- needs the whole bucket output).
  4. MinhashDedupFilter    -- re-read the original input, keep one
                              representative per cluster, write final output.

hi's stage 1 and stage 4 chain the hi-fineweb2 and hi-sangraha readers
back-to-back in the same pipeline list (datatrove's BaseDiskReader.run()
does `if data: yield from data` before yielding its own documents, so two
readers back-to-back concatenate their output into one stream) -- this is
what makes hi-fineweb2 and hi-sangraha dedup *against each other*, not
just internally.

Usage:
  # Smoke-test locally on a tiny sample first (see tests/test_dedup_pipeline.py
  # for an even smaller synthetic-data version of this same wiring):
  uv run scripts/run_dedup_datatrove.py --executor local --limit 500 --tasks 2

  # Real run, sharded across a SLURM job array (datatrove submits its own
  # sbatch jobs when a SlurmPipelineExecutor's .run() is called -- this
  # script itself only needs to run once, e.g. from the login node or a
  # lightweight driver job):
  uv run scripts/run_dedup_datatrove.py --executor slurm --tasks 200 \\
      --account berzelius-2026-167 --partition berzelius-cpu --time 24:00:00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from datatrove.utils.word_tokenizers import WordTokenizer

from src.sources import ROWS_BY_LANGUAGE, SOURCE_SPECS, VALID_LANGUAGES

REPO_ROOT = Path(__file__).resolve().parent.parent


class WhitespaceWordTokenizer(WordTokenizer):
    """Plain `str.split()` word tokenizer for MinHash shingling, passed
    directly as `MinhashDedupSignature(language=...)` instead of a language
    code string.

    Why not datatrove's own language routing: pt/es route to
    `SpaCyTokenizer` (needs `spacy` + a downloaded model) and hi routes to
    `StanzaTokenizer("kmr")` (needs `stanza`, which pulls in a full PyTorch
    + CUDA stack just to split words) -- confirmed by actually hitting both
    import errors. Even datatrove's own `WhitespaceTokenizer` "fallback"
    secretly delegates to spaCy's multilingual "xx" model for
    `word_tokenize`, so it doesn't avoid the dependency either. Our old
    from-scratch dedup (`src/dedup/minhash.py::_shingles`, since removed)
    only ever did `\\S+` splitting for shingling regardless of language --
    this preserves that exact behavior with zero extra dependencies.
    `sent_tokenize`/`span_tokenize` are required by the `WordTokenizer`
    ABC but are never called by `MinhashDedupSignature` (only
    `word_tokenize` is), so they're trivial/unused here.
    """

    def word_tokenize(self, text: str) -> list[str]:
        return text.split()

    def sent_tokenize(self, text: str) -> list[str]:
        return [text]

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(0, len(text))]


def load_minhash_config(path: Path):
    from datatrove.pipeline.dedup.minhash import MinhashConfig

    with open(path) as f:
        raw = yaml.safe_load(f)
    return MinhashConfig(**raw)


def build_readers(lang: str, raw_dir: Path, limit: int):
    from datatrove.pipeline.readers import ParquetReader

    rows = ROWS_BY_LANGUAGE[lang]
    # Each source has different extra columns (fineweb-2: dump/url/date/...,
    # sangraha: type) -- with read_metadata's default (True), those become a
    # per-row `metadata` struct whose shape differs by source, which breaks
    # ParquetWriter when a shared output file gets batches from more than one
    # source (mismatched struct schemas). Only hi actually combines two
    # sources into one pipeline, so only hi needs metadata disabled -- pt/es
    # are single-source and keep their real provenance columns.
    read_metadata = len(rows) == 1

    readers = []
    for row in rows:
        spec = SOURCE_SPECS[row]
        readers.append(
            ParquetReader(
                data_folder=str(raw_dir / row),
                glob_pattern=spec["pattern"],
                text_key=spec["text_key"],
                id_key=spec["id_key"],
                read_metadata=read_metadata,
                limit=limit,
            )
        )
    return readers


def build_language_stages(
    lang: str,
    raw_dir: Path,
    work_dir: Path,
    out_dir: Path,
    minhash_config,
    limit: int,
    tasks: int,
    executor_kind: str,
    slurm_kwargs: dict,
):
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.executor.slurm import SlurmPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
    from datatrove.pipeline.writers.parquet import ParquetWriter

    lang_dir = work_dir / lang
    sig_dir, buckets_dir, clusters_dir, removed_dir = (
        lang_dir / "signatures",
        lang_dir / "buckets",
        lang_dir / "remove_ids",
        lang_dir / "removed",
    )
    final_dir = out_dir / lang

    def executor(pipeline, stage_tasks, stage_name, depends=None, **stage_slurm_kwargs):
        logging_dir = str(lang_dir / "logs" / stage_name)
        if executor_kind == "local":
            return LocalPipelineExecutor(pipeline=pipeline, tasks=stage_tasks, logging_dir=logging_dir, depends=depends)
        return SlurmPipelineExecutor(
            pipeline=pipeline,
            tasks=stage_tasks,
            logging_dir=logging_dir,
            slurm_logs_folder=str(lang_dir / "slurm_logs" / stage_name),
            depends=depends,
            **{**slurm_kwargs, **stage_slurm_kwargs},
        )

    # Built once, reused by both stage 1 and stage 4 -- must read the exact
    # same input in the exact same order/task split both times (datatrove's
    # own requirement, see MinhashDedupFilter's docstring).
    readers = build_readers(lang, raw_dir, limit)

    stage1 = executor(
        [*readers, MinhashDedupSignature(output_folder=str(sig_dir), config=minhash_config, language=WhitespaceWordTokenizer())],
        tasks,
        "signatures",
    )
    stage2 = executor(
        [MinhashDedupBuckets(input_folder=str(sig_dir), output_folder=str(buckets_dir), config=minhash_config)],
        minhash_config.num_buckets,
        "buckets",
        depends=stage1,
    )
    stage3 = executor(
        [MinhashDedupCluster(input_folder=str(buckets_dir), output_folder=str(clusters_dir), config=minhash_config)],
        1,
        "clusters",
        depends=stage2,
        # Single unparallelized task that loads every bucket's duplicate-match
        # pairs into memory at once -- datatrove's own default (2GB, shared
        # with the other stages' many-tasks-of-1-CPU-each shape) OOM-killed
        # partway through es-fineweb2 (441M docs, by far the largest source)
        # at real scale, confirmed via the SLURM job's own oom_kill log.
        # pt/hi (smaller) finished fine on the default -- this override only
        # affects stage 3, not the other three (which don't need it).
        mem_per_cpu_gb=200,
    )
    stage4 = executor(
        [
            *readers,
            MinhashDedupFilter(
                input_folder=str(clusters_dir), exclusion_writer=ParquetWriter(output_folder=str(removed_dir))
            ),
            ParquetWriter(output_folder=str(final_dir)),
        ],
        tasks,
        "filter",
        depends=stage3,
    )
    return stage1, stage2, stage3, stage4


def run(args: argparse.Namespace) -> None:
    minhash_config = load_minhash_config(args.config)
    languages = args.languages or sorted(VALID_LANGUAGES)

    slurm_kwargs = {}
    if args.executor == "slurm":
        slurm_kwargs = {
            "time": args.time,
            "partition": args.partition,
            "sbatch_args": {"account": args.account},
            # datatrove's launch script does `source {venv_path}` verbatim
            # (no /bin/activate appended) -- confirmed against the installed
            # package's source (executor/slurm.py's get_launch_file_contents).
            "venv_path": str(REPO_ROOT / ".venv" / "bin" / "activate"),
        }

    for lang in languages:
        print(f"--- {lang}: building {len(ROWS_BY_LANGUAGE[lang])} row(s) -> {ROWS_BY_LANGUAGE[lang]} ---")
        _, _, _, stage4 = build_language_stages(
            lang,
            args.raw_dir,
            args.work_dir,
            args.out_dir,
            minhash_config,
            args.limit,
            args.tasks,
            args.executor,
            slurm_kwargs,
        )
        stage4.run()
        print(f"--- {lang}: deduped output -> {args.out_dir / lang} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw", help="scripts/download_sources.py's --out-dir")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed", help="Final deduped output (same shape scripts/build_final_dataset.py expects)")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "data" / "dedup_work", help="Intermediate signatures/buckets/clusters")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dedup.yaml")
    parser.add_argument("--languages", nargs="*", choices=sorted(VALID_LANGUAGES), default=None, help="Default: all 3")
    parser.add_argument("--limit", type=int, default=-1, help="Cap docs read per source row (-1 = no cap); passed to each ParquetReader")
    parser.add_argument("--tasks", type=int, default=4, help="Shards for stage 1/4 (stage 2 always uses num_buckets tasks, stage 3 always 1). A real full-scale slurm run should use far more, e.g. 200.")
    parser.add_argument("--executor", choices=["local", "slurm"], required=True)
    parser.add_argument("--account", default=None, help="SLURM account (sbatch -A) -- required if --executor slurm")
    parser.add_argument("--partition", default="berzelius-cpu", help="SLURM partition -- only used if --executor slurm")
    parser.add_argument("--time", default="24:00:00", help="Per-stage SLURM time limit -- only used if --executor slurm")
    args = parser.parse_args()

    if args.executor == "slurm" and not args.account:
        parser.error("--account is required with --executor slurm")

    run(args)
