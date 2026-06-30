from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
import random
from typing import Any


SUPPORTED_STRUCTURE_SUFFIXES = frozenset({".pdb", ".ent"})


@dataclass(frozen=True)
class StructuralTokenSequence:
    token_features: list[list[float]]
    token_mask: list[bool]


@dataclass(frozen=True)
class CAFrame:
    frame_index: int
    residue_keys: tuple[tuple[str, str, str], ...]
    coordinates: tuple[tuple[float, float, float], ...]


def build_structural_token_sequence(
    *,
    nature_path: str | Path | None = None,
    md_path: str | Path | None = None,
    max_residues: int = 100,
    max_frames: int = 160,
) -> StructuralTokenSequence:
    token_features: list[list[float]] = []

    if nature_path is not None:
        token_features.extend(
            load_structure_tokens(nature_path, max_residues=max_residues)
        )
    if md_path is not None:
        token_features.extend(
            load_md_clip_tokens(
                md_path,
                max_residues=max_residues,
                max_frames=max_frames,
            )
        )

    if not token_features:
        raise ValueError("At least one of nature_path or md_path must provide tokens.")

    return StructuralTokenSequence(
        token_features=token_features,
        token_mask=[True] * len(token_features),
    )


def save_structural_token_cache(
    path: str | Path,
    structural_tokens: StructuralTokenSequence,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_features": structural_tokens.token_features,
        "token_mask": structural_tokens.token_mask,
        "metadata": dict(metadata or {}),
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_structural_token_cache(path: str | Path) -> StructuralTokenSequence:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Structural token cache does not exist: {cache_path}")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, StructuralTokenSequence):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported structural token cache format: {cache_path}")

    token_features = payload.get("token_features")
    token_mask = payload.get("token_mask")
    if not isinstance(token_features, list) or not isinstance(token_mask, list):
        raise ValueError(
            f"Structural token cache must contain list token_features and token_mask: {cache_path}"
        )
    if len(token_features) != len(token_mask):
        raise ValueError(
            f"Structural token cache has mismatched feature/mask lengths: {cache_path}"
        )
    return StructuralTokenSequence(
        token_features=token_features,
        token_mask=[bool(value) for value in token_mask],
    )


def load_ca_frame(
    path: str | Path,
    *,
    max_residues: int | None = None,
) -> CAFrame:
    frames = _extract_ca_frames_from_structure(
        path,
        max_residues=max_residues,
    )
    if not frames:
        raise ValueError(
            f"No CA atom coordinates could be parsed from {Path(path)}"
        )
    return frames[0]


def load_structure_tokens(
    path: str | Path,
    *,
    max_residues: int = 100,
) -> list[list[float]]:
    coordinates = _extract_residue_coordinates(path, max_residues=max_residues)
    total_residues = len(coordinates)

    return [
        _build_feature_vector(
            x=x_coord,
            y=y_coord,
            z=z_coord,
            residue_index=residue_index,
            total_residues=total_residues,
            frame_index=0,
            total_frames=1,
            source_flag=0.0,
        )
        for residue_index, (x_coord, y_coord, z_coord) in enumerate(coordinates)
    ]


def load_md_clip_tokens(
    path: str | Path,
    *,
    max_residues: int = 100,
    max_frames: int = 160,
) -> list[list[float]]:
    frames = load_md_ca_frames(
        path,
        max_residues=max_residues,
        max_frames=max_frames,
    )
    total_frames = len(frames)
    token_features: list[list[float]] = []

    for frame_index, frame in enumerate(frames):
        total_residues = len(frame.coordinates)
        token_features.extend(
            _build_feature_vector(
                x=x_coord,
                y=y_coord,
                z=z_coord,
                residue_index=residue_index,
                total_residues=total_residues,
                frame_index=frame_index,
                total_frames=total_frames,
                source_flag=1.0,
            )
            for residue_index, (x_coord, y_coord, z_coord) in enumerate(frame.coordinates)
        )

    return token_features


def load_md_ca_frames(
    path: str | Path,
    *,
    max_residues: int = 100,
    max_frames: int = 160,
) -> list[CAFrame]:
    md_path = Path(path)
    frames = _extract_md_ca_frames(md_path, max_residues=max_residues)
    frames = _sample_ca_frames(frames, seed_path=md_path, max_frames=max_frames)
    return [
        CAFrame(
            frame_index=frame_index,
            residue_keys=frame.residue_keys,
            coordinates=frame.coordinates,
        )
        for frame_index, frame in enumerate(frames)
    ]


def resolve_md_frame_paths(path: str | Path, *, max_frames: int = 160) -> list[Path]:
    md_path = Path(path)
    if not md_path.exists():
        raise FileNotFoundError(f"MD path does not exist: {md_path}")

    if md_path.is_file():
        return [md_path]

    frame_paths = sorted(
        (
            file_path
            for file_path in md_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES
        ),
        key=lambda file_path: file_path.as_posix(),
    )
    if not frame_paths:
        raise FileNotFoundError(f"No supported MD frame files were found in {md_path}")
    if len(frame_paths) <= max_frames:
        return frame_paths

    selection_rng = random.Random(_stable_frame_sampling_seed(md_path))
    sampled_indices = sorted(selection_rng.sample(range(len(frame_paths)), k=max_frames))
    return [frame_paths[index] for index in sampled_indices]


def _extract_residue_coordinates(
    path: str | Path,
    *,
    max_residues: int | None,
) -> list[tuple[float, float, float]]:
    frames = _extract_ca_frames_from_structure(path, max_residues=max_residues)
    if not frames:
        raise ValueError(
            f"No CA atom coordinates could be parsed from {Path(path)}"
        )
    return list(frames[0].coordinates)


def _extract_md_frame_coordinates(
    path: str | Path,
    *,
    max_residues: int,
) -> list[list[tuple[float, float, float]]]:
    return [list(frame.coordinates) for frame in _extract_md_ca_frames(path, max_residues=max_residues)]


def _extract_md_ca_frames(
    path: str | Path,
    *,
    max_residues: int,
) -> list[CAFrame]:
    md_path = Path(path)
    structure_files = _resolve_md_structure_files(md_path)
    frames: list[CAFrame] = []
    for structure_file in structure_files:
        frames.extend(
            _extract_ca_frames_from_structure(
                structure_file,
                max_residues=max_residues,
            )
        )
    if not frames:
        raise ValueError(
            f"No CA atom coordinate frames could be parsed from MD path {md_path}"
        )
    return [
        CAFrame(
            frame_index=frame_index,
            residue_keys=frame.residue_keys,
            coordinates=frame.coordinates,
        )
        for frame_index, frame in enumerate(frames)
    ]


def _resolve_md_structure_files(path: str | Path) -> list[Path]:
    md_path = Path(path)
    if not md_path.exists():
        raise FileNotFoundError(f"MD path does not exist: {md_path}")
    if md_path.is_file():
        return [md_path]

    structure_files = sorted(
        (
            file_path
            for file_path in md_path.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES
            and file_path.stat().st_size > 0
        ),
        key=lambda file_path: file_path.as_posix(),
    )
    if not structure_files:
        raise FileNotFoundError(
            f"No non-empty supported MD structure files were found in {md_path}"
        )
    return structure_files


def _extract_ca_coordinate_frames(
    path: str | Path,
    *,
    max_residues: int | None,
) -> list[list[tuple[float, float, float]]]:
    return [list(frame.coordinates) for frame in _extract_ca_frames_from_structure(path, max_residues=max_residues)]


def _extract_ca_frames_from_structure(
    path: str | Path,
    *,
    max_residues: int | None,
) -> list[CAFrame]:
    structure_path = Path(path)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure path does not exist: {structure_path}")
    if not structure_path.is_file():
        raise ValueError(f"Structure path is not a file: {structure_path}")

    frames: list[CAFrame] = []
    current_residue_order: list[tuple[str, str, str]] = []
    current_residue_coordinates: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    saw_model = False
    in_model = False

    def append_current_frame() -> None:
        if not current_residue_order:
            return
        ordered_residue_keys = tuple(current_residue_order if max_residues is None else current_residue_order[:max_residues])
        ordered_coordinates = tuple(
            current_residue_coordinates[residue_key]
            for residue_key in ordered_residue_keys
        )
        if ordered_coordinates:
            frames.append(
                CAFrame(
                    frame_index=len(frames),
                    residue_keys=ordered_residue_keys,
                    coordinates=ordered_coordinates,
                )
            )

    def reset_current_frame() -> None:
        current_residue_order.clear()
        current_residue_coordinates.clear()

    with structure_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_name = line[:6].strip()
            if record_name == "MODEL":
                append_current_frame()
                reset_current_frame()
                saw_model = True
                in_model = True
                continue
            if record_name == "ENDMDL" and saw_model:
                append_current_frame()
                reset_current_frame()
                in_model = False
                continue
            if record_name != "ATOM":
                continue
            if saw_model and not in_model:
                continue

            atom_name = line[12:16].strip()
            # Use protein backbone alpha-carbon coordinates only. HETATM records
            # can contain calcium ions named "CA", which must not become residue tokens.
            if atom_name != "CA":
                continue
            chain_id = line[21].strip() or "_"
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_number, insertion_code)

            try:
                coordinates = (
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip()),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate line in structure file {structure_path}: {line.rstrip()}"
                ) from exc

            if residue_key not in current_residue_coordinates:
                current_residue_order.append(residue_key)
            current_residue_coordinates[residue_key] = coordinates

    append_current_frame()
    return frames


def _sample_frame_coordinates(
    frames: list[list[tuple[float, float, float]]],
    *,
    seed_path: Path,
    max_frames: int,
) -> list[list[tuple[float, float, float]]]:
    return [
        list(frame.coordinates)
        for frame in _sample_ca_frames(
            [
                CAFrame(
                    frame_index=frame_index,
                    residue_keys=tuple(("_", str(residue_index), "") for residue_index in range(len(coordinates))),
                    coordinates=tuple(coordinates),
                )
                for frame_index, coordinates in enumerate(frames)
            ],
            seed_path=seed_path,
            max_frames=max_frames,
        )
    ]


def _sample_ca_frames(
    frames: list[CAFrame],
    *,
    seed_path: Path,
    max_frames: int,
) -> list[CAFrame]:
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    if len(frames) <= max_frames:
        return frames
    selection_rng = random.Random(_stable_frame_sampling_seed(seed_path))
    sampled_indices = sorted(selection_rng.sample(range(len(frames)), k=max_frames))
    return [frames[index] for index in sampled_indices]


def _build_feature_vector(
    *,
    x: float,
    y: float,
    z: float,
    residue_index: int,
    total_residues: int,
    frame_index: int,
    total_frames: int,
    source_flag: float,
) -> list[float]:
    residue_position = (
        0.0 if total_residues <= 1 else residue_index / float(total_residues - 1)
    )
    frame_position = 0.0 if total_frames <= 1 else frame_index / float(total_frames - 1)

    # Minimal structural token features: coordinates, normalized residue position,
    # normalized frame position, and a source flag (0=nature, 1=MD).
    # Extension point: replace this with richer geometry features when the downstream model
    # contracts for structural-dynamics tokens are stable.
    return [x, y, z, residue_position, frame_position, source_flag]


def _stable_frame_sampling_seed(md_path: Path) -> int:
    digest = hashlib.sha256(md_path.as_posix().encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
