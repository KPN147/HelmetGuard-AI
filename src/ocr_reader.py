"""
OCR Reader Module
License-plate text extraction using PaddleOCR.
"""

import cv2
from paddleocr import PaddleOCR


class OCRReader:
    """Extract text from license-plate crops."""

    def __init__(self, lang: str = "vi", use_angle_cls: bool = True,
                 upscale_factor: int = 2, padding: int = 5):
        """
        Args:
            lang: PaddleOCR language code ('vi', 'en', 'ch', …).
            use_angle_cls: Enable angle classification for rotated text.
            upscale_factor: Resize multiplier applied before OCR.
            padding: Extra pixels added around the plate crop.
        """
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang,
                             show_log=False)
        self.upscale_factor = upscale_factor
        self.padding = padding

    def read_plate(self, image, bbox) -> str:
        """
        Crop, preprocess, and OCR a license-plate region.

        Args:
            image: Full BGR image (numpy array).
            bbox: (x1, y1, x2, y2) bounding box of the plate.

        Returns:
            Detected text string (empty string on failure).
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Apply padding (clamped to image bounds)
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        # Upscale for better OCR accuracy
        crop = cv2.resize(crop, None,
                          fx=self.upscale_factor, fy=self.upscale_factor,
                          interpolation=cv2.INTER_CUBIC)

        try:
            result = self.ocr.ocr(crop)
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                return " ".join(texts)
        except Exception:
            pass

        return ""
