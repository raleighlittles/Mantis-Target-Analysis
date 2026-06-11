#!/usr/bin/env python3
"""Scan images and decode ArUco marker IDs, including dictionary metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# OpenCV's built-in ArUco dictionaries.
# Note: standard predefined ArUco dictionaries are 4x4, 5x5, 6x6, 7x7,
# plus DICT_ARUCO_ORIGINAL, which is 5x5.
ARUCO_DICTIONARY_NAMES = [
    "4X4_50",
    "4X4_100",
    "4X4_250",
    "4X4_1000",
    "5X5_50",
    "5X5_100",
    "5X5_250",
    "5X5_1000",
    "6X6_50",
    "6X6_100",
    "6X6_250",
    "6X6_1000",
    "7X7_50",
    "7X7_100",
    "7X7_250",
    "7X7_1000",
    "ARUCO_ORIGINAL",
]


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _flatten_ids(ids) -> List[int]:
    if ids is None:
        return []
    return [int(marker_id) for marker_id in ids.flatten().tolist()]


def normalize_dictionary_name(dictionary_name: str) -> str:
    """Accept either '4X4_50' or 'DICT_4X4_50'."""
    dictionary_name = dictionary_name.strip()
    if dictionary_name.startswith("DICT_"):
        dictionary_name = dictionary_name[len("DICT_") :]
    return dictionary_name


def dictionary_metadata(dictionary_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Return:
      marker_size: e.g. '4x4'
      dictionary_capacity: e.g. 50

    OpenCV naming example:
      4X4_50 -> marker_size='4x4', dictionary_capacity=50
    """
    dictionary_name = normalize_dictionary_name(dictionary_name)

    match = re.fullmatch(r"(\d+)X(\d+)_(\d+)", dictionary_name)
    if match:
        rows, cols, count = match.groups()
        return f"{rows}x{cols}", int(count)

    if dictionary_name == "ARUCO_ORIGINAL":
        return "5x5", 1024

    return None, None


def get_dictionary(dictionary_name: str):
    if cv2 is None or not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available in this environment.")

    dictionary_name = normalize_dictionary_name(dictionary_name)
    dictionary_constant_name = f"DICT_{dictionary_name}"

    if not hasattr(cv2.aruco, dictionary_constant_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_constant_name)
    )


def detect_marker_ids(image, dictionary_name: str) -> List[int]:
    dictionary = get_dictionary(dictionary_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            cv2.aruco.DetectorParameters(),
        )
        _, ids, _ = detector.detectMarkers(image)
    else:
        _, ids, _ = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=cv2.aruco.DetectorParameters_create(),
        )

    return _flatten_ids(ids)


def detect_markers_for_dictionary(image, dictionary_name: str) -> List[Dict[str, Any]]:
    dictionary_name = normalize_dictionary_name(dictionary_name)
    marker_size, dictionary_capacity = dictionary_metadata(dictionary_name)

    detections: List[Dict[str, Any]] = []
    for marker_id in detect_marker_ids(image, dictionary_name):
        detections.append(
            {
                "id": marker_id,
                "dictionary": f"DICT_{dictionary_name}",
                "marker_size": marker_size,
                "dictionary_capacity": dictionary_capacity,
            }
        )

    return detections


def available_aruco_dictionary_names() -> List[str]:
    if cv2 is None or not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available in this environment.")

    return [
        name
        for name in ARUCO_DICTIONARY_NAMES
        if hasattr(cv2.aruco, f"DICT_{name}")
    ]


def detect_markers_auto(image) -> List[Dict[str, Any]]:
    """
    Try every built-in OpenCV ArUco dictionary and report all successful matches.

    Important:
    The same physical marker may sometimes be detectable under more than one
    related dictionary. If that happens, this intentionally reports all matches
    rather than pretending the marker ID alone proves one unique dictionary.
    """
    detections: List[Dict[str, Any]] = []

    for dictionary_name in available_aruco_dictionary_names():
        detections.extend(detect_markers_for_dictionary(image, dictionary_name))

    return detections


def decode_images(
    image_paths: Iterable[Path],
    dictionary_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")

    results: Dict[str, List[Dict[str, Any]]] = {}
    dictionary_name = normalize_dictionary_name(dictionary_name)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            results[str(image_path)] = []
            continue

        if dictionary_name.lower() == "auto":
            results[str(image_path)] = detect_markers_auto(image)
        else:
            results[str(image_path)] = detect_markers_for_dictionary(
                image,
                dictionary_name,
            )

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
        default="auto",
        help=(
            "ArUco dictionary name suffix, such as 4X4_50 or 5X5_100. "
            "Use 'auto' to try all built-in ArUco dictionaries. "
            "Default: auto."
        ),
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