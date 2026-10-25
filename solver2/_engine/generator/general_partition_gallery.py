"""通用 mixed 分割图库、联系表和多样性统计。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .general_partition_generator import descriptor_key
from .random_generator import random_pieces
from .validator import check_partition

COLORS = [
    (102, 194, 165),
    (252, 141, 98),
    (141, 160, 203),
    (231, 138, 195),
]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _render_partition(
    path: Path,
    pieces,
    width: float,
    height: float,
    title: str,
    size: tuple[int, int] = (720, 520),
) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    margin_x, margin_y = 55, 75
    usable_w = size[0] - 2 * margin_x
    usable_h = size[1] - margin_y - 55
    scale = min(usable_w / width, usable_h / height)
    origin_x = (size[0] - width * scale) / 2.0
    origin_y = margin_y + (usable_h - height * scale) / 2.0

    def screen(point):
        return (
            int(round(origin_x + point[0] * scale)),
            int(round(origin_y + point[1] * scale)),
        )

    for index, piece in enumerate(pieces):
        points = [screen(point) for point in piece.vertices]
        draw.polygon(points, fill=COLORS[index % len(COLORS)], outline="black")
        draw.line(points + [points[0]], fill="black", width=3, joint="curve")
        center = np.asarray(piece.vertices, dtype=float).mean(axis=0)
        label = f"P{index}"
        draw.text(screen(center), label, fill="black", anchor="mm")
    draw.rectangle(
        [
            screen((0.0, 0.0)),
            screen((width, height)),
        ],
        outline="black",
        width=4,
    )
    draw.text((size[0] // 2, 25), title, fill="black", anchor="mm")
    draw.text(
        (size[0] // 2, size[1] - 20),
        f"{width:.1f} x {height:.1f} mm",
        fill="black",
        anchor="mm",
    )
    image.save(path, format="PNG", optimize=True)


def _distribution(values: list[float], bins: list[float]) -> dict[str, int]:
    counts = Counter()
    for value in values:
        placed = False
        for low, high in zip(bins[:-1], bins[1:]):
            if low <= value < high:
                counts[f"[{low:g},{high:g})"] += 1
                placed = True
                break
        if not placed and value >= bins[-1]:
            counts[f"[{bins[-1]:g},inf)"] += 1
    return dict(counts)


def _write_contact_sheet(
    root: Path,
    cases: list[dict],
    rng: np.random.Generator,
) -> None:
    sample_count = min(100, len(cases))
    selected_indices = sorted(
        int(index)
        for index in rng.choice(len(cases), size=sample_count, replace=False)
    )
    thumb_size = (200, 145)
    columns = 10
    rows = (sample_count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")
    selected = []
    for slot, index in enumerate(selected_indices):
        case = cases[index]
        with Image.open(root / case["image"]) as source:
            thumb = source.copy()
            thumb.thumbnail(thumb_size)
        x = (slot % columns) * thumb_size[0]
        y = (slot // columns) * thumb_size[1]
        sheet.paste(thumb, (x, y))
        selected.append(
            {
                "index": case["index"],
                "seed": case["seed"],
                "image": case["image"],
            }
        )
    sheet.save(root / "contact_sheet_100.png", optimize=True)
    (root / "contact_sheet_selection.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_general_partition_gallery(
    count: int = 1000,
    seed: int = 10000,
    output: str | Path = "outputs/general_partition_gallery",
) -> dict:
    if count <= 0:
        raise ValueError("count 必须大于 0")
    root = Path(output)
    images_dir = root / "images"
    examples_dir = root / "signature_examples"
    images_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    descriptor_counts: Counter[str] = Counter()
    rejected_near_duplicates = 0
    candidate_index = 0
    signature_examples: dict[str, list[Path]] = defaultdict(list)
    maximum_candidates = max(10000, count * 30)
    while len(cases) < count and candidate_index < maximum_candidates:
        case_seed = seed + candidate_index
        candidate_index += 1
        rng = np.random.default_rng(case_seed)
        pieces, width, height, metadata = random_pieces(
            rng,
            seed=case_seed,
            generation_mode="mixed",
            return_metadata=True,
        )
        validation = check_partition(pieces, width, height)
        if not validation.valid:
            raise AssertionError(
                f"seed={case_seed} 统一验证失败: {validation.errors}"
            )
        key = descriptor_key(metadata)
        if descriptor_counts[key] >= 3:
            rejected_near_duplicates += 1
            continue
        descriptor_counts[key] += 1
        index = len(cases) + 1
        signature = metadata["adjacency_canonical_signature"]
        filename = (
            f"{index:04d}_{metadata['generator_mode']}_"
            f"{_safe_name(signature)}_seed_{case_seed:06d}.png"
        )
        image_path = images_dir / filename
        _render_partition(
            image_path,
            pieces,
            width,
            height,
            (
                f"{metadata['generator_mode']} | {signature} | "
                f"seed={case_seed}"
            ),
        )
        case = {
            "index": index,
            "seed": case_seed,
            "generator_mode": metadata["generator_mode"],
            "template_topology": metadata.get("template_topology"),
            "piece_count": len(pieces),
            "width_mm": width,
            "height_mm": height,
            "image": f"images/{filename}",
            "valid": True,
            **{
                key_name: metadata[key_name]
                for key_name in (
                    "adjacency_canonical_signature",
                    "interior_node_count",
                    "interior_node_degrees",
                    "common_edge_count",
                    "piece_edge_counts",
                    "minimum_interior_angle_deg",
                    "near_parallel_common_edge_pair_count",
                    "near_equal_common_edge_pair_count",
                    "has_t_junction",
                    "has_multiple_internal_nodes",
                    "geometric_symmetry_or_ambiguity_risk",
                    "common_edge_direction_deg",
                    "normalized_geometry_descriptor",
                )
            },
        }
        cases.append(case)
        if len(signature_examples[signature]) < 5:
            signature_examples[signature].append(image_path)

    if len(cases) != count:
        raise RuntimeError(
            f"只生成 {len(cases)}/{count} 个非重复合法案例，"
            f"候选数={candidate_index}"
        )

    for signature, paths in signature_examples.items():
        signature_dir = examples_dir / _safe_name(signature)
        signature_dir.mkdir(parents=True, exist_ok=True)
        for source in paths:
            shutil.copyfile(source, signature_dir / source.name)

    piece_count_distribution = Counter(case["piece_count"] for case in cases)
    signature_distribution = Counter(
        case["adjacency_canonical_signature"] for case in cases
    )
    internal_node_distribution = Counter(
        case["interior_node_count"] for case in cases
    )
    edge_count_distribution = Counter(
        edge_count
        for case in cases
        for edge_count in case["piece_edge_counts"]
    )
    generator_mode_distribution = Counter(
        case["generator_mode"] for case in cases
    )
    minimum_angles = [
        float(case["minimum_interior_angle_deg"]) for case in cases
    ]
    common_directions = [
        float(direction)
        for case in cases
        for direction in case["common_edge_direction_deg"]
    ]
    duplicate_ratio = rejected_near_duplicates / max(candidate_index, 1)
    summary = {
        "count": count,
        "seed": seed,
        "all_valid": True,
        "candidate_count": candidate_index,
        "different_adjacency_signature_count": len(signature_distribution),
        "piece_count_distribution": {
            str(key): value
            for key, value in sorted(piece_count_distribution.items())
        },
        "adjacency_signature_distribution": dict(
            sorted(signature_distribution.items())
        ),
        "internal_node_count_distribution": {
            str(key): value
            for key, value in sorted(internal_node_distribution.items())
        },
        "t_junction_count": sum(case["has_t_junction"] for case in cases),
        "t_junction_ratio": sum(
            case["has_t_junction"] for case in cases
        ) / count,
        "multiple_internal_node_count": sum(
            case["has_multiple_internal_nodes"] for case in cases
        ),
        "multiple_internal_node_ratio": sum(
            case["has_multiple_internal_nodes"] for case in cases
        ) / count,
        "piece_edge_count_distribution": {
            str(key): value
            for key, value in sorted(edge_count_distribution.items())
        },
        "minimum_interior_angle_distribution_deg": _distribution(
            minimum_angles, [0, 20, 30, 45, 60, 90, 120, 180]
        ),
        "common_edge_direction_distribution_deg": _distribution(
            common_directions,
            list(range(0, 181, 15)),
        ),
        "near_equal_edge_difficult_case_count": sum(
            case["near_equal_common_edge_pair_count"] > 0 for case in cases
        ),
        "near_equal_edge_difficult_case_ratio": sum(
            case["near_equal_common_edge_pair_count"] > 0 for case in cases
        ) / count,
        "generator_mode_distribution": dict(
            sorted(generator_mode_distribution.items())
        ),
        "template_sample_ratio": generator_mode_distribution["template"] / count,
        "rejected_near_duplicate_count": rejected_near_duplicates,
        "near_duplicate_candidate_ratio": duplicate_ratio,
        "accepted_descriptor_duplicate_ratio": sum(
            amount - 1 for amount in descriptor_counts.values() if amount > 1
        ) / count,
        "cases": cases,
    }
    (root / "gallery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (root / "adjacency_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["adjacency_canonical_signature", "count", "ratio"],
        )
        writer.writeheader()
        for signature, amount in sorted(signature_distribution.items()):
            writer.writerow(
                {
                    "adjacency_canonical_signature": signature,
                    "count": amount,
                    "ratio": amount / count,
                }
            )

    geometry_fields = [
        "index",
        "seed",
        "generator_mode",
        "piece_count",
        "adjacency_canonical_signature",
        "interior_node_count",
        "interior_node_degrees",
        "common_edge_count",
        "piece_edge_counts",
        "minimum_interior_angle_deg",
        "near_parallel_common_edge_pair_count",
        "near_equal_common_edge_pair_count",
        "has_t_junction",
        "has_multiple_internal_nodes",
        "geometric_symmetry_or_ambiguity_risk",
        "common_edge_direction_deg",
        "image",
    ]
    with (root / "geometry_distribution.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=geometry_fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    field: (
                        json.dumps(case[field], ensure_ascii=False)
                        if isinstance(case[field], (list, dict))
                        else case[field]
                    )
                    for field in geometry_fields
                }
            )

    _write_contact_sheet(
        root, cases, np.random.default_rng(seed + 9_999_991)
    )
    return summary
