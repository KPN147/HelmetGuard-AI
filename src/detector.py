"""
Object Detector Module
Wraps YOLO model for helmet / cyclist / license-plate detection.
"""

from collections import defaultdict
from ultralytics import YOLO


class ObjectDetector:
    """Thin wrapper around a YOLO model for structured inference."""

    def __init__(self, model_path: str, class_names: dict,
                 conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Args:
            model_path: Path to YOLO .pt weights.
            class_names: Mapping {int_class_id: str_label}.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for non-maximum suppression.
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image):
        """
        Run YOLO inference on a BGR image.

        Returns:
            dict[str, list[dict]]: Detections grouped by class name.
            Each detection has keys: bbox (x1,y1,x2,y2), confidence, class_id.
        """
        results = self.model(
            image, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]

        detections = defaultdict(list)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.class_names.get(cls_id, f"class_{cls_id}")

            detections[label].append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
            })

        return dict(detections)
