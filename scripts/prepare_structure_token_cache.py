from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Any

from src.datasets.structure_io import (
    build_structural_token_sequence,
    save_structural_token_cache,
)


CACHE_COLUMNS = ("dyn_cache_path", "peptide_dyn_cache_path")


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else output_dir / f"{manifest_path.stem}_cached.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Downstream manifest does not exist: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        input_fieldnames = list(reader.fieldnames or [])

    if args.task:
        rows = [
            row
            for row in rows
            if (row.get("task_name") or "").strip().lower() == args.task.strip().lower()
        ]
    if not rows:
        raise ValueError(f"No rows selected from {manifest_path}")

    fieldnames = list(input_fieldnames)
    for column in CACHE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    cache_index: dict[str, str] = {}
    built_count = 0
    reused_count = 0
    dyn_rows = 0

    for row_index, row in enumerate(rows, start=1):
        row.setdefault("dyn_cache_path", "")
        row.setdefault("peptide_dyn_cache_path", "")

        if _parse_bool(row.get("has_dyn")):
            dyn_rows += 1
            cache_path, was_built = _cache_one_side(
                row=row,
                side="protein",
                output_dir=output_dir,
                max_residues=args.max_residues,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
                absolute_cache_paths=args.absolute_cache_paths,
                cache_index=cache_index,
            )
            row["dyn_cache_path"] = cache_path
            built_count += int(was_built)
            reused_count += int(not was_built)

        if _parse_bool(row.get("peptide_has_dyn")):
            dyn_rows += 1
            cache_path, was_built = _cache_one_side(
                row=row,
                side="peptide",
                output_dir=output_dir,
                max_residues=args.max_residues,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
                absolute_cache_paths=args.absolute_cache_paths,
                cache_index=cache_index,
            )
            row["peptide_dyn_cache_path"] = cache_path
            built_count += int(was_built)
            reused_count += int(not was_built)

        if args.log_interval > 0 and row_index % args.log_interval == 0:
            print(
                f"processed_rows={row_index} built={built_count} reused={reused_count}",
                flush=True,
            )

    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"selected_rows={len(rows)}")
    print(f"dyn_sides={dyn_rows}")
    print(f"built_caches={built_count}")
    print(f"reused_caches={reused_count}")
    print(f"cache_dir={output_dir}")
    print(f"output_manifest={output_manifest}")


def _cache_one_side(
    *,
    row: dict[str, str],
    side: str,
    output_dir: Path,
    max_residues: int,
    max_frames: int,
    overwrite: bool,
    absolute_cache_paths: bool,
    cache_index: dict[str, str],
) -> tuple[str, bool]:
    nature_path_column = "nature_path" if side == "protein" else "peptide_nature_path"
    md_path_column = "md_path" if side == "protein" else "peptide_md_path"
    hash_column = "sequence_hash" if side == "protein" else "peptide_sequence_hash"

    nature_path = (row.get(nature_path_column) or "").strip()
    md_path = (row.get(md_path_column) or "").strip()
    sequence_hash = (row.get(hash_column) or "").strip()
    if not nature_path or not md_path:
        raise ValueError(
            f"Row {row.get('sample_id')!r} has {side} dynamics but lacks "
            f"{nature_path_column} or {md_path_column}."
        )

    cache_key = _cache_key(
        nature_path=nature_path,
        md_path=md_path,
        max_residues=max_residues,
        max_frames=max_frames,
    )
    if cache_key in cache_index:
        return cache_index[cache_key], False

    prefix = sequence_hash or _short_hash(row.get("sample_id") or cache_key)
    cache_filename = f"{side}_{prefix}_{cache_key[:16]}.pkl"
    cache_path = output_dir / cache_filename
    was_built = False

    if overwrite or not cache_path.exists():
        structural_tokens = build_structural_token_sequence(
            nature_path=nature_path,
            md_path=md_path,
            max_residues=max_residues,
            max_frames=max_frames,
        )
        save_structural_token_cache(
            cache_path,
            structural_tokens,
            metadata={
                "sample_id": row.get("sample_id") or "",
                "side": side,
                "sequence_hash": sequence_hash,
                "nature_path": nature_path,
                "md_path": md_path,
                "max_residues": max_residues,
                "max_frames": max_frames,
                "num_tokens": len(structural_tokens.token_features),
            },
        )
        was_built = True

    stored_path = _stored_cache_path(cache_path, absolute=absolute_cache_paths)
    cache_index[cache_key] = stored_path
    return stored_path, was_built


def _stored_cache_path(path: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    return os.path.relpath(path, Path.cwd())


def _cache_key(
    *,
    nature_path: str,
    md_path: str,
    max_residues: int,
    max_frames: int,
) -> str:
    value = "\n".join(
        [
            Path(nature_path).as_posix(),
            Path(md_path).as_posix(),
            str(max_residues),
            str(max_frames),
        ]
    )
    return _short_hash(value, length=32)


def _short_hash(value: str, *, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-tokenize downstream structural-dynamics raw files into pkl caches "
            "and write a manifest that points training to those caches."
        )
    )
    parser.add_argument("--manifest-path", required=True, help="Input downstream manifest CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where pkl structural token caches will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Output cached manifest CSV. Defaults to <output-dir>/<input_stem>_cached.csv.",
    )
    parser.add_argument("--task", default=None, help="Optional task_name filter.")
    parser.add_argument("--max-residues", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cache files even when an output pkl already exists.",
    )
    parser.add_argument(
        "--absolute-cache-paths",
        action="store_true",
        help="Write absolute cache paths into the output manifest instead of repo-relative paths.",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
