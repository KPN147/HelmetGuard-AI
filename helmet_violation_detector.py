import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path
import pytz


class HelmetViolationDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the helmet violation detector

        Args:
            model_path: Path to YOLOv11 model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Initialize PaddleOCR

        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        # Class names (adjust based on your model)
        self.class_names = {0: 'helmet', 1: 'license_plate', 2: 'cyclist'}

        # Create output directory
        self.output_dir = Path('violation_records')
        self.output_dir.mkdir(exist_ok=True)

    def detect_objects(self, image):
        """
        Detect objects in the image using YOLOv11

        Args:
            image: Input image (BGR format)

        Returns:
            detections: Dictionary of detections by class
        """
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)[0]

        detections = defaultdict(list)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = self.class_names.get(cls_id, f'class_{cls_id}')

            detections[class_name].append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls_id
            })

        return detections

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def is_cyclist_wearing_helmet(self, cyclist_box, helmet_boxes, iou_threshold=0.3):
        """
        Check if a cyclist is wearing a helmet based on spatial proximity

        Args:
            cyclist_box: Bounding box of cyclist (x1, y1, x2, y2)
            helmet_boxes: List of helmet bounding boxes
            iou_threshold: Minimum IOU to consider helmet as worn

        Returns:
            bool: True if wearing helmet, False otherwise
        """
        if not helmet_boxes:
            return False

        # Check if any helmet overlaps with the upper portion of cyclist
        cyclist_x1, cyclist_y1, cyclist_x2, cyclist_y2 = cyclist_box

        # Define upper body region (top 40% of cyclist box)
        upper_body_height = (cyclist_y2 - cyclist_y1) * 0.4
        upper_body_box = (cyclist_x1, cyclist_y1, cyclist_x2, cyclist_y1 + upper_body_height)

        for helmet_box in helmet_boxes:
            iou = self.calculate_iou(upper_body_box, helmet_box)
            if iou > iou_threshold:
                return True

        return False


    def read_license_plate(self, image, bbox):
        """
        Read text from license plate using PaddleOCR

        Args:
            image: Full image
            bbox: License plate bounding box (x1, y1, x2, y2)

        Returns:
            str: Detected license plate text
        """
        x1, y1, x2, y2 = bbox

        # Add padding to license plate crop
        padding = 5
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        plate_img = image[y1:y2, x1:x2]

        if plate_img.size == 0:
            return ""
        cv2.imwrite("Original.jpg", plate_img)
        # Apply preprocessing for better OCR
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("Resize.jpg", plate_img)
        # Run OCR
        try:
            result = self.ocr.ocr(plate_img)

            if result and result[0]:
                # Combine all detected text
                rec_texts = result[0]['rec_texts']
                plate_text = ' '.join(rec_texts)
                return plate_text
        except:
            pass

        return ""

    def process_image(self, image, save_violations=True):
        """
        Process image to detect helmet violations

        Args:
            image: Input image (BGR format)
            save_violations: Whether to save violation records

        Returns:
            annotated_image: Image with annotations
            violations: List of violation records
        """
        # Detect all objects
        detections = self.detect_objects(image)

        # Extract bounding boxes
        cyclists = detections.get('cyclist', [])
        helmets = detections.get('helmet', [])
        license_plates = detections.get('license_plate', [])

        helmet_boxes = [h['bbox'] for h in helmets]

        violations = []
        annotated_image = image.copy()

        # Check each cyclist for helmet violation
        for cyclist in cyclists:
            cyclist_box = cyclist['bbox']
            is_wearing_helmet = self.is_cyclist_wearing_helmet(cyclist_box, helmet_boxes)

            # Draw cyclist box
            color = (0, 255, 0) if is_wearing_helmet else (0, 0, 255)
            x1, y1, x2, y2 = cyclist_box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            label = "Cyclist (OK)" if is_wearing_helmet else "VIOLATION"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not is_wearing_helmet:
              # Đảm bảo có đủ biến
              y1, y2 = cyclist_box[1], cyclist_box[3]
              x1, x2 = cyclist_box[0], cyclist_box[2]

              violation_record = {
                  'timestamp': datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d %H:%M:%S'),
                  'cyclist_box': cyclist_box,
                  'confidence': cyclist['confidence'],
                  'license_plate': None,
                  'plate_text': ""
              }

              # Tìm biển số GẦN NHẤT
              cyclist_center_x = (x1 + x2) / 2
              cyclist_center_y = (y1 + y2) / 2

              best_plate = None
              min_distance = float('inf')

              for plate in license_plates:
                  plate_box = plate['bbox']
                  px1, py1, px2, py2 = plate_box

                  plate_center_x = (px1 + px2) / 2
                  plate_center_y = (py1 + py2) / 2

                  # Tính khoảng cách Euclidean
                  distance = ((cyclist_center_x - plate_center_x)**2 +
                            (cyclist_center_y - plate_center_y)**2)**0.5

                  # Kiểm tra khoảng cách hợp lý (200 pixels)
                  if distance < 200 and distance < min_distance:
                      # Biển số nên ở dưới hoặc ngang người
                      if plate_center_y >= cyclist_center_y - 50:
                          min_distance = distance
                          best_plate = plate_box

              # Đọc biển số tốt nhất
              if best_plate:
                  plate_text = self.read_license_plate(image, best_plate)

                  if plate_text:
                      violation_record['license_plate'] = best_plate
                      violation_record['plate_text'] = plate_text

                      # Vẽ bbox với validation
                      px1, py1, px2, py2 = map(int, best_plate)
                      cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (255, 0, 0), 2)
                      cv2.putText(annotated_image, f"Plate: {plate_text}",
                                (px1, max(py1 - 10, 0)),  # Tránh vẽ ngoài ảnh
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                      print(f"✓ Found plate '{plate_text}' at distance {min_distance:.1f}px")

              violations.append(violation_record)

              if save_violations:
                  self.save_violation_record(image, violation_record)

        # Draw helmet boxes
        for helmet in helmets:
            x1, y1, x2, y2 = helmet['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_image, "Helmet", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw license plates that weren't associated with violations
        for plate in license_plates:
            plate_box = plate['bbox']
            px1, py1, px2, py2 = plate_box
            # Only draw if not already drawn as part of violation
            already_drawn = any(v['license_plate'] == plate_box for v in violations if v['license_plate'])
            if not already_drawn:
                cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (255, 255, 0), 2)
                cv2.putText(annotated_image, "Plate", (px1, py1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return annotated_image, violations

    def save_violation_record(self, image, violation):
        """Save violation record with image and details"""
        timestamp = violation['timestamp'].replace(':', '-').replace(' ', '_')
        violation_dir = self.output_dir / timestamp
        violation_dir.mkdir(exist_ok=True)

        # Crop violator image
        x1, y1, x2, y2 = violation['cyclist_box']
        violator_img = image[y1:y2, x1:x2]

        # Save violator image
        cv2.imwrite(str(violation_dir / 'violator.jpg'), violator_img)

        # Save license plate image if available
        if violation['license_plate']:
            px1, py1, px2, py2 = violation['license_plate']
            plate_img = image[py1:py2, px1:px2]
            # cv2.imwrite(str(violation_dir / 'license_plate'.jpg), plate_img)
            cv2.imwrite(str(violation_dir / 'license_plate.jpg'), plate_img)

        # Save violation details
        details_path = violation_dir / 'violation_details.txt'
        with open(details_path, 'w') as f:
            f.write(f"Timestamp: {violation['timestamp']}\n")
            f.write(f"Confidence: {violation['confidence']:.2f}\n")
            f.write(f"License Plate: {violation['plate_text'] if violation['plate_text'] else 'Not detected'}\n")

        print(f"Violation record saved to: {violation_dir}")


def main():
    """Example usage"""
# Initialize detector
detector = HelmetViolationDetector(
    model_path='./best.pt',  # Your YOLOv11 model path
    conf_threshold=0.5
)

# Process a test image
test_image_path = './test_image.jpg'
if os.path.exists(test_image_path):
    image = cv2.imread(test_image_path)
    annotated_image, violations = detector.process_image(image)

    print(f"Detected {len(violations)} violations")
    for i, v in enumerate(violations, 1):
        print(f"Violation {i}: Plate = {v['plate_text']}")

    # Save result
    cv2.imwrite('result.jpg', annotated_image)
    print("Result saved to result.jpg")


if __name__ == '__main__':
    main()