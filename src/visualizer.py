"""
Visualizer Module
Draw detection results (bounding boxes, labels) on images.
"""

import cv2
from typing import Tuple

# Default colour palette (BGR)
COLOR_OK = (0, 255, 0)          # Green  – cyclist with helmet
COLOR_VIOLATION = (0, 0, 255)   # Red    – cyclist without helmet
COLOR_HELMET = (255, 255, 0)    # Cyan   – helmet box
COLOR_PLATE = (255, 0, 0)       # Blue   – license plate box

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2


class Visualizer:
    """Annotate an image with detection & violation results."""

    @staticmethod
    def draw(image, analysis_results: list, detections: dict,
             violation_plates: dict | None = None):
        """
        Annotate and return a copy of the image.

        Args:
            image: BGR numpy array.
            analysis_results: Output of ViolationAnalyzer.analyze().
            detections: Raw detections dict (for un-associated helmets/plates).
            violation_plates: {plate_bbox_tuple: plate_text} for labelling.

        Returns:
            Annotated BGR image (copy).
        """
        canvas = image.copy()
        violation_plates = violation_plates or {}

        drawn_plate_boxes = set()

        # --- Cyclists ---
        for item in analysis_results:
            bbox = item["cyclist"]["bbox"]
            x1, y1, x2, y2 = bbox

            if item["has_helmet"]:
                color = COLOR_OK
                label = "Cyclist (OK)"
            else:
                color = COLOR_VIOLATION
                label = "VIOLATION"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            cv2.putText(canvas, label, (x1, y1 - 10),
                        FONT, FONT_SCALE, color, FONT_THICKNESS)

            # Draw associated plate
            plate = item.get("nearest_plate")
            if plate and not item["has_helmet"]:
                pbox = plate["bbox"]
                px1, py1, px2, py2 = pbox
                cv2.rectangle(canvas, (px1, py1), (px2, py2),
                              COLOR_PLATE, BOX_THICKNESS)
                plate_text = violation_plates.get(pbox, "")
                if plate_text:
                    cv2.putText(canvas, f"Plate: {plate_text}",
                                (px1, max(py1 - 10, 0)),
                                FONT, 0.5, COLOR_PLATE, FONT_THICKNESS)
                drawn_plate_boxes.add(pbox)

        # --- Helmets ---
        for helmet in detections.get("helmet", []):
            hx1, hy1, hx2, hy2 = helmet["bbox"]
            cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2),
                          COLOR_HELMET, BOX_THICKNESS)
            cv2.putText(canvas, "Helmet", (hx1, hy1 - 10),
                        FONT, 0.5, COLOR_HELMET, FONT_THICKNESS)

        # --- Un-associated license plates ---
        for plate in detections.get("license_plate", []):
            if plate["bbox"] not in drawn_plate_boxes:
                px1, py1, px2, py2 = plate["bbox"]
                cv2.rectangle(canvas, (px1, py1), (px2, py2),
                              COLOR_HELMET, BOX_THICKNESS)
                cv2.putText(canvas, "Plate", (px1, py1 - 10),
                            FONT, 0.5, COLOR_HELMET, FONT_THICKNESS)

        return canvas
