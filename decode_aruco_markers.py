#!/usr/bin/env python3
"""Scan images and decode ArUco marker IDs, including dictionary metadata."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Smaller dictionaries are listed before larger dictionaries, so if a marker
# matches DICT_4X4_50 and also DICT_4X4_100, the 4X4_50 result wins, since Aruco dictionaries are supersets
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

    Example:
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


def available_aruco_dictionary_names() -> List[str]:
    if cv2 is None or not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available in this environment.")

    return [
        name
        for name in ARUCO_DICTIONARY_NAMES
        if hasattr(cv2.aruco, f"DICT_{name}")
    ]


def detect_marker_raw(image, dictionary_name: str) -> List[Tuple[int, List[List[float]]]]:
    """
    Return raw detections as:
      [
        (marker_id, [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]),
        ...
      ]
    """
    dictionary = get_dictionary(dictionary_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            cv2.aruco.DetectorParameters(),
        )
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=cv2.aruco.DetectorParameters_create(),
        )

    if ids is None:
        return []

    detections: List[Tuple[int, List[List[float]]]] = []

    for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
        # OpenCV usually returns marker_corners with shape (1, 4, 2).
        pts = marker_corners.reshape(4, 2).tolist()
        detections.append((int(marker_id), pts))

    return detections


def marker_center(corners: List[List[float]]) -> Tuple[float, float]:
    x = sum(point[0] for point in corners) / 4.0
    y = sum(point[1] for point in corners) / 4.0
    return x, y


def marker_average_side_length(corners: List[List[float]]) -> float:
    total = 0.0

    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        total += math.hypot(x2 - x1, y2 - y1)

    return total / 4.0


def same_physical_marker(
    corners_a: List[List[float]],
    corners_b: List[List[float]],
    center_tolerance_ratio: float = 0.20,
) -> bool:
    """
    Decide whether two detections refer to the same physical marker.

    Instead of comparing marker IDs, this compares image location.

    center_tolerance_ratio:
      0.20 means the centers can differ by up to 20% of the smaller marker's
      average side length and still be considered the same physical marker.
    """
    ax, ay = marker_center(corners_a)
    bx, by = marker_center(corners_b)

    center_distance = math.hypot(ax - bx, ay - by)

    side_a = marker_average_side_length(corners_a)
    side_b = marker_average_side_length(corners_b)
    tolerance = min(side_a, side_b) * center_tolerance_ratio

    return center_distance <= tolerance


def is_duplicate_physical_marker(
    corners: List[List[float]],
    accepted_corners: List[List[List[float]]],
) -> bool:
    return any(same_physical_marker(corners, existing) for existing in accepted_corners)


def make_detection_result(
    marker_id: int,
    dictionary_name: str,
    corners: Optional[List[List[float]]] = None,
    include_corners: bool = False,
) -> Dict[str, Any]:
    dictionary_name = normalize_dictionary_name(dictionary_name)
    marker_size, dictionary_capacity = dictionary_metadata(dictionary_name)

    result: Dict[str, Any] = {
        "id": marker_id,
        "dictionary": f"DICT_{dictionary_name}",
        "marker_size": marker_size,
        "dictionary_capacity": dictionary_capacity,
    }

    if corners is not None:
        result["bounding_box"] = marker_bounding_box(corners)

    if include_corners and corners is not None:
        result["corners"] = corners

    return result


def detect_markers_for_dictionary(
    image,
    dictionary_name: str,
    include_corners: bool = False,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []

    for marker_id, corners in detect_marker_raw(image, dictionary_name):
        detections.append(
            make_detection_result(
                marker_id=marker_id,
                dictionary_name=dictionary_name,
                corners=corners,
                include_corners=include_corners,
            )
        )

    return detections


def detect_markers_auto(
    image,
    include_corners: bool = False,
) -> List[Dict[str, Any]]:
    """
    Try every built-in OpenCV ArUco dictionary, but return only the first
    dictionary match for each physical marker location.

    Example:
      If the same physical marker is detected as DICT_4X4_50 and DICT_4X4_100,
      only the DICT_4X4_50 result is kept because it appears first in
      ARUCO_DICTIONARY_NAMES.
    """
    results: List[Dict[str, Any]] = []
    accepted_corners: List[List[List[float]]] = []

    for dictionary_name in available_aruco_dictionary_names():
        raw_detections = detect_marker_raw(image, dictionary_name)

        for marker_id, corners in raw_detections:
            if is_duplicate_physical_marker(corners, accepted_corners):
                continue

            accepted_corners.append(corners)

            results.append(
                make_detection_result(
                    marker_id=marker_id,
                    dictionary_name=dictionary_name,
                    corners=corners,
                    include_corners=include_corners,
                )
            )

    return results


def decode_images(
    image_paths: Iterable[Path],
    dictionary_name: str,
    include_corners: bool = False,
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
            results[str(image_path)] = detect_markers_auto(
                image,
                include_corners=include_corners,
            )
        else:
            results[str(image_path)] = detect_markers_for_dictionary(
                image,
                dictionary_name,
                include_corners=include_corners,
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
        "--include-corners",
        action="store_true",
        help="Include each marker's detected corner coordinates in the JSON output.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    return parser

def marker_bounding_box(corners: List[List[float]]) -> Dict[str, float]:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
        "center_x": (x_min + x_max) / 2.0,
        "center_y": (y_min + y_max) / 2.0,
    }

def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def annotated_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")


def annotate_detected_markers(
    image_path: Path,
    detections: List[Dict[str, Any]],
    box_thickness: int = 5,
) -> Optional[Path]:
    """
    Create an annotated copy of image_path showing detected ArUco markers.

    This does not modify the original image.

    The annotation includes:
      - a rectangle around the marker bounding box
      - marker ID
      - ArUco dictionary name

    For markers in the top half of the image, the label is drawn below.
    For markers in the bottom half of the image, the label is drawn above.

    Returns the output path if an annotated image was written, otherwise None.
    """
    if not detections:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    annotated = image.copy()
    image_height, image_width = annotated.shape[:2]

    box_color = (0, 255, 0)
    text_color = (255, 255, 255)
    label_background_color = (0, 0, 0)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    text_thickness = 2
    label_padding = 8
    label_gap = 8

    drew_anything = False

    for detection in detections:
        bounding_box = detection.get("bounding_box")
        if not bounding_box:
            continue

        x_min = clamp_int(bounding_box["x_min"], 0, image_width - 1)
        y_min = clamp_int(bounding_box["y_min"], 0, image_height - 1)
        x_max = clamp_int(bounding_box["x_max"], 0, image_width - 1)
        y_max = clamp_int(bounding_box["y_max"], 0, image_height - 1)

        center_y = float(
            bounding_box.get(
                "center_y",
                (bounding_box["y_min"] + bounding_box["y_max"]) / 2.0,
            )
        )

        marker_id = detection.get("id", "?")
        dictionary = detection.get("dictionary", "?")
        label_text = f"ID {marker_id} | {dictionary}"

        # Draw the marker box.
        cv2.rectangle(
            annotated,
            (x_min, y_min),
            (x_max, y_max),
            box_color,
            box_thickness,
        )

        text_size, baseline = cv2.getTextSize(
            label_text,
            font_face,
            font_scale,
            text_thickness,
        )

        text_width, text_height = text_size

        label_width = text_width + (2 * label_padding)
        label_height = text_height + baseline + (2 * label_padding)

        # Keep the label horizontally inside the image.
        label_x_min = clamp_int(
            x_min,
            0,
            max(0, image_width - label_width),
        )
        label_x_max = label_x_min + label_width

        marker_is_in_bottom_half = center_y >= (image_height / 2.0)

        if marker_is_in_bottom_half:
            # Put label above the marker.
            label_y_min = y_min - label_gap - label_height

            # If it would go off the top edge, fall back to below.
            if label_y_min < 0:
                label_y_min = y_max + label_gap
        else:
            # Put label below the marker.
            label_y_min = y_max + label_gap

            # If it would go off the bottom edge, fall back to above.
            if label_y_min + label_height > image_height:
                label_y_min = y_min - label_gap - label_height

        # Final clamp so the label always stays visible.
        label_y_min = clamp_int(
            label_y_min,
            0,
            max(0, image_height - label_height),
        )
        label_y_max = label_y_min + label_height

        # Draw label background.
        cv2.rectangle(
            annotated,
            (label_x_min, label_y_min),
            (label_x_max, label_y_max),
            label_background_color,
            thickness=-1,
        )

        # Draw label text.
        text_origin = (
            label_x_min + label_padding,
            label_y_min + label_padding + text_height,
        )

        cv2.putText(
            annotated,
            label_text,
            text_origin,
            font_face,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

        drew_anything = True

    if not drew_anything:
        return None

    output_path = annotated_output_path(image_path)

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write annotated image: {output_path}")

    return output_path

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_path.exists():
        parser.error(f"Input path does not exist: {args.input_path}")

    image_paths = gather_images(args.input_path)

    results = decode_images(
        image_paths,
        args.dictionary,
        include_corners=args.include_corners,
    )

    for image_path in image_paths:
        annotate_detected_markers(
            image_path=image_path,
            detections=results.get(str(image_path), []),
            box_thickness=5,
        )

    json_kwargs = {"indent": 2} if args.pretty else {}
    print(json.dumps(results, **json_kwargs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())