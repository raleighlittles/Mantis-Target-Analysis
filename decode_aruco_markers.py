#!/usr/bin/env python3
"""Scan images and decode ArUco marker IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import cv2
except ImportError:  # pragma: no cover - depends on environment setup
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def _flatten_ids(ids) -> List[int]:
    if ids is None:
        return []
    return [int(marker_id) for marker_id in ids.flatten().tolist()]


def detect_markers(image, dictionary_name: str) -> List[int]:
    if cv2 is None or not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available in this environment.")

    dictionary_attr = f"DICT_{dictionary_name}"
    if not hasattr(cv2.aruco, dictionary_attr):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_attr))
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=cv2.aruco.DetectorParameters_create(),
        )
    _ = corners
    return _flatten_ids(ids)


def decode_images(image_paths: Iterable[Path], dictionary_name: str) -> Dict[str, List[int]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")

    results: Dict[str, List[int]] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            results[str(image_path)] = []
            continue
        results[str(image_path)] = detect_markers(image, dictionary_name)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode ArUco markers from images.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an image file or directory containing images.",
    )
    parser.add_argument(
        "--dictionary",
        default="4X4_50",
        help="ArUco dictionary name suffix (default: 4X4_50).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input path does not exist: {args.input_path}")

    image_paths = gather_images(args.input_path)
    results = decode_images(image_paths, args.dictionary)
    json_kwargs = {"indent": 2} if args.pretty else {}
    print(json.dumps(results, **json_kwargs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
