"""
Pipeline Module
End-to-end orchestration: detect → analyze → OCR → record → visualize.
"""

import cv2
from pathlib import Path
from datetime import datetime

import pytz

from .detector import ObjectDetector
from .violation_analyzer import ViolationAnalyzer
from .ocr_reader import OCRReader
from .visualizer import Visualizer


class HelmetViolationPipeline:
    """High-level pipeline that ties all modules together."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, ocr_lang: str = "vi",
                 output_dir: str = "outputs/violation_records",
                 class_names: dict | None = None,
                 helmet_iou: float = 0.3,
                 upper_body_ratio: float = 0.4,
                 max_plate_distance: float = 200.0):
        self.class_names = class_names or {
            0: "helmet", 1: "license_plate", 2: "cyclist"
        }

        self.detector = ObjectDetector(
            model_path=model_path,
            class_names=self.class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self.analyzer = ViolationAnalyzer(
            helmet_iou_threshold=helmet_iou,
            upper_body_ratio=upper_body_ratio,
            max_plate_distance=max_plate_distance,
        )
        self.ocr = OCRReader(lang=ocr_lang)
        self.visualizer = Visualizer()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Confidence threshold passthrough
    # ------------------------------------------------------------------
    @property
    def conf_threshold(self):
        return self.detector.conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, value):
        self.detector.conf_threshold = value

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_image(self, image, save_violations: bool = True):
        """
        Full pipeline on a single BGR image.

        Returns:
            (annotated_image, violations_list)
        """
        # 1. Detect
        detections = self.detector.detect(image)

        # 2. Analyze (rule-based)
        analysis = self.analyzer.analyze(detections)

        # 3. OCR on violation plates + build records
        violations = []
        plate_texts = {}  # bbox → text for visualizer

        for item in analysis:
            if item["has_helmet"]:
                continue

            plate = item["nearest_plate"]
            plate_text = ""
            plate_bbox = None

            if plate is not None:
                plate_bbox = plate["bbox"]
                plate_text = self.ocr.read_plate(image, plate_bbox)
                if plate_text:
                    plate_texts[plate_bbox] = plate_text

            record = {
                "timestamp": datetime.now(
                    pytz.timezone("Asia/Ho_Chi_Minh")
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "cyclist_box": item["cyclist"]["bbox"],
                "confidence": item["cyclist"]["confidence"],
                "license_plate": plate_bbox,
                "plate_text": plate_text,
            }
            violations.append(record)

            if save_violations:
                self._save_record(image, record)

        # 4. Visualize
        annotated = self.visualizer.draw(
            image, analysis, detections, plate_texts
        )

        return annotated, violations

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, output_path: str = "output_video.mp4",
                      frame_skip: int = 5, save_violations: bool = True):
        """
        Process a video file frame-by-frame.

        Returns:
            (output_video_path, all_violations)
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        all_violations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % frame_skip == 0:
                annotated, violations = self.process_image(
                    frame, save_violations=save_violations
                )
                for v in violations:
                    v["frame"] = frame_idx
                all_violations.extend(violations)
            else:
                annotated = frame

            writer.write(annotated)

        cap.release()
        writer.release()
        return output_path, all_violations

    # ------------------------------------------------------------------
    # Artifact saving
    # ------------------------------------------------------------------

    def _save_record(self, image, record: dict):
        """Persist violation crop, plate crop, and details to disk."""
        ts = record["timestamp"].replace(":", "-").replace(" ", "_")
        folder = self.output_dir / ts
        folder.mkdir(parents=True, exist_ok=True)

        # Violator crop
        x1, y1, x2, y2 = record["cyclist_box"]
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(str(folder / "violator.jpg"), crop)

        # License plate crop
        if record["license_plate"]:
            px1, py1, px2, py2 = record["license_plate"]
            plate_crop = image[py1:py2, px1:px2]
            cv2.imwrite(str(folder / "license_plate.jpg"), plate_crop)

        # Text details
        with open(folder / "violation_details.txt", "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {record['timestamp']}\n")
            f.write(f"Confidence: {record['confidence']:.2f}\n")
            f.write(f"License Plate: "
                    f"{record['plate_text'] or 'Not detected'}\n")
