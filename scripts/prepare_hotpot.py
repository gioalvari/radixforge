#!/usr/bin/env python3
"""Create the deterministic, compact HotpotQA development spike dataset.

The primary source is the CC BY-SA 4.0 HotpotQA distractor development JSON.
It is deliberately downloaded with the standard library so the script has no
Python package dependency.  The Hugging Face fallback is retained for source
provenance: it downloads the requested parquet revision with the installed
``hf`` CLI, then explains that a parquet decoder is required to convert it.
The normal Hotpot endpoint is used in the recorded spike run.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SEED = 20260925
COUNT = 100
SOURCE_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


def trim_words(text: str, limit: int = 120) -> str:
    """Normalize whitespace and retain at most ``limit`` whitespace words."""
    return " ".join(text.split()[:limit])


def download_primary(destination: Path) -> list[dict[str, Any]]:
    """Download and parse the official distractor development JSON."""
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(str(error)) from error
    destination.write_bytes(payload)
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("HotpotQA source did not contain a JSON array")
    return parsed


def download_hf_fallback(cache_dir: Path) -> list[dict[str, Any]]:
    """Download the requested HF parquet fallback using the installed CLI.

    The script remains stdlib-only: DuckDB is invoked as an external executable
    to turn the parquet rows into JSON.  This is preferable to importing a
    parquet package into the preparation script.
    """
    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("official source failed and the `hf` CLI is unavailable")
    command = [
        hf,
        "download",
        "hotpotqa/hotpot_qa",
        "--repo-type",
        "dataset",
        "--include",
        "distractor/validation-*",
        "--local-dir",
        str(cache_dir),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"HF download failed: {error}") from error
    if not shutil.which("duckdb"):
        raise RuntimeError("HF parquet downloaded but `duckdb` is required to read it")
    parquet_files = sorted(cache_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError("HF download completed without a validation parquet file")
    quoted = str(parquet_files[0]).replace("'", "''")
    command = [
        "duckdb",
        "--no-stdin",
        "-json",
        "-c",
        "SELECT id AS _id, question, answer, supporting_facts, "
        "list_transform(range(1, array_length(context.title) + 1), "
        "lambda i: struct_pack(title := context.title[i], "
        "sentences := context.sentences[i])) AS context "
        f"FROM read_parquet('{quoted}')",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"DuckDB parquet conversion failed: {error.stderr}"
        ) from error
    json_start = completed.stdout.find('[{"')
    if json_start < 0:
        raise RuntimeError("DuckDB parquet conversion emitted no JSON")
    parsed = json.loads(completed.stdout[json_start:])
    if not isinstance(parsed, list):
        raise ValueError("HF fallback did not contain a JSON array")
    return parsed


def bm25_scores(question: str, chunks: list[str]) -> list[float]:
    """Score each chunk against a question with per-item BM25 statistics."""
    documents = [
        [
            token
            for token in re.findall(r"[a-z0-9]+", chunk.lower())
            if token not in STOPWORDS
        ]
        for chunk in chunks
    ]
    query = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in STOPWORDS
    }
    document_frequency = collections.Counter(
        token for document in documents for token in set(document)
    )
    average_length = sum(map(len, documents)) / len(documents)
    scores: list[float] = []
    for document in documents:
        frequencies = collections.Counter(document)
        length_norm = 1.2 * (1.0 - 0.75 + 0.75 * len(document) / average_length)
        score = 0.0
        for token in query:
            frequency = frequencies[token]
            if frequency == 0:
                continue
            inverse_frequency = math.log(
                1.0
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += inverse_frequency * frequency * 2.2 / (frequency + length_norm)
        scores.append(score)
    return scores


def supporting_titles(record: dict[str, Any]) -> set[str]:
    """Extract the supporting paragraph titles from either source schema."""
    supporting_facts = record.get("supporting_facts")
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
    elif isinstance(supporting_facts, list) and supporting_facts:
        titles = [
            fact[0] for fact in supporting_facts if isinstance(fact, list) and fact
        ]
    else:
        raise ValueError(
            f"missing supporting facts for {record.get('_id', '<unknown>')}"
        )
    if not isinstance(titles, list):
        raise ValueError("supporting fact titles must be a list")
    return {str(title) for title in titles}


def to_item(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an official HotpotQA record to the spike's JSONL schema."""
    context = record["context"]
    if not isinstance(context, list) or len(context) != 10:
        raise ValueError(f"invalid context for {record.get('_id', '<unknown>')}")
    titles: list[str] = []
    texts: list[str] = []
    for paragraph in context:
        if isinstance(paragraph, dict):
            title = paragraph["title"]
            sentences = paragraph["sentences"]
        elif isinstance(paragraph, list) and len(paragraph) == 2:
            title, sentences = paragraph
        else:
            raise ValueError("context paragraph must contain title and sentences")
        if not isinstance(sentences, list):
            raise ValueError("paragraph sentences must be a list")
        titles.append(str(title))
        texts.append(trim_words(f"{title}: {' '.join(sentences)}"))
    scores = bm25_scores(str(record["question"]), texts)
    gold_titles = supporting_titles(record)
    return {
        "id": record["_id"],
        "question": record["question"],
        "answer": record["answer"],
        # Keep the established chunk-string schema and expose retrieval labels in
        # parallel arrays so existing consumers do not need to unwrap objects.
        "chunks": texts,
        "gold": [title in gold_titles for title in titles],
        "bm25": scores,
    }


def main() -> int:
    """Download, deterministically sample, and write the JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/hotpot_dev_100.jsonl"))
    parser.add_argument(
        "--cache", type=Path, default=Path("data/hotpot_dev_distractor_v1.json")
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hf_cache = args.out.parent / "hf-hotpotqa"
    if list(hf_cache.rglob("*.parquet")):
        records = download_hf_fallback(args.out.parent / "hf-hotpotqa")
        source = "HF hotpotqa/hotpot_qa distractor validation parquet"
    else:
        try:
            records = download_primary(args.cache)
            source = SOURCE_URL
        except RuntimeError as error:
            print(f"official HotpotQA download failed: {error}", file=sys.stderr)
            records = download_hf_fallback(hf_cache)
            source = "HF hotpotqa/hotpot_qa distractor validation parquet"

    ordered = sorted(records, key=lambda record: record["_id"])
    selected = random.Random(SEED).sample(ordered, COUNT)
    selected.sort(key=lambda record: record["_id"])
    top_hits = {2: 0, 3: 0}
    eligible = 0
    with args.out.open("w", encoding="utf-8") as output:
        for record in selected:
            item = to_item(record)
            gold = {index for index, is_gold in enumerate(item["gold"]) if is_gold}
            if len(gold) == 2:
                eligible += 1
                ranked = sorted(
                    range(len(item["chunks"])),
                    key=lambda index: (-item["bm25"][index], index),
                )
                for limit in top_hits:
                    if gold.issubset(ranked[:limit]):
                        top_hits[limit] += 1
            output.write(json.dumps(item, ensure_ascii=False))
            output.write("\n")
    print(f"wrote {len(selected)} items to {args.out}")
    print(f"source used: {source}")
    for limit, hits in top_hits.items():
        rate = hits / eligible if eligible else 0.0
        print(
            f"BM25 top-{limit} contains both gold paragraphs: {hits}/{eligible} ({rate:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
