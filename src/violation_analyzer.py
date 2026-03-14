"""
Violation Analyzer Module
Rule-based logic for helmet-cyclist matching and plate association.
"""

import math
from typing import List, Tuple, Optional


BBox = Tuple[int, int, int, int]


class ViolationAnalyzer:
    """Determines helmet violations using spatial heuristics."""

    def __init__(self, helmet_iou_threshold: float = 0.3,
                 upper_body_ratio: float = 0.4,
                 max_plate_distance: float = 200.0):
        """
        Args:
            helmet_iou_threshold: Min IoU between helmet and cyclist upper
                body to consider the helmet as "worn".
            upper_body_ratio: Fraction of cyclist bbox height treated as
                upper body (measured from top).
            max_plate_distance: Max Euclidean pixel distance to associate
                a license plate with a cyclist.
        """
        self.helmet_iou_threshold = helmet_iou_threshold
        self.upper_body_ratio = upper_body_ratio
        self.max_plate_distance = max_plate_distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, detections: dict) -> list:
        """
        Analyze detections and return a list of violation dicts.

        Args:
            detections: Output of ObjectDetector.detect().

        Returns:
            list[dict]: Each dict has keys:
                - cyclist (the original detection dict)
                - has_helmet (bool)
                - nearest_plate (detection dict or None)
        """
        cyclists = detections.get("cyclist", [])
        helmets = detections.get("helmet", [])
        plates = detections.get("license_plate", [])

        helmet_boxes = [h["bbox"] for h in helmets]
        results = []

        for cyclist in cyclists:
            has_helmet = self._is_wearing_helmet(cyclist["bbox"], helmet_boxes)
            nearest_plate = (
                self._find_nearest_plate(cyclist["bbox"], plates)
                if not has_helmet else None
            )
            results.append({
                "cyclist": cyclist,
                "has_helmet": has_helmet,
                "nearest_plate": nearest_plate,
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_wearing_helmet(self, cyclist_box: BBox,
                           helmet_boxes: List[BBox]) -> bool:
        """Check whether any helmet overlaps the cyclist's upper body."""
        if not helmet_boxes:
            return False

        cx1, cy1, cx2, cy2 = cyclist_box
        upper_h = (cy2 - cy1) * self.upper_body_ratio
        upper_body = (cx1, cy1, cx2, cy1 + upper_h)

        return any(
            self._iou(upper_body, hbox) > self.helmet_iou_threshold
            for hbox in helmet_boxes
        )

    def _find_nearest_plate(self, cyclist_box: BBox,
                            plates: list) -> Optional[dict]:
        """Return the closest license-plate detection within range."""
        cx = (cyclist_box[0] + cyclist_box[2]) / 2
        cy = (cyclist_box[1] + cyclist_box[3]) / 2

        best, best_dist = None, float("inf")
        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            dist = math.hypot(cx - pcx, cy - pcy)

            # Plate should be roughly below or at the same height as cyclist
            if dist < self.max_plate_distance and dist < best_dist:
                if pcy >= cy - 50:
                    best, best_dist = plate, dist

        return best

    @staticmethod
    def _iou(box_a: BBox, box_b: BBox) -> float:
        """Compute Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
