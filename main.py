"""
main.py — CLI entry point for HelmetGuard AI.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --video path/to/video.mp4
"""

import argparse
import os
import cv2

import config
from src.pipeline import HelmetViolationPipeline
from src.model_loader import get_model_path


def build_pipeline() -> HelmetViolationPipeline:
    model_path = get_model_path(
        repo_id=config.HF_REPO_ID,
        filename=config.HF_MODEL_FILENAME,
        local_dir=config.LOCAL_MODEL_DIR,
    )
    return HelmetViolationPipeline(
        model_path=model_path,
        conf_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        ocr_lang=config.OCR_LANGUAGE,
        output_dir=config.OUTPUT_DIR,
        class_names=config.CLASS_NAMES,
        helmet_iou=config.HELMET_CYCLIST_IOU_THRESHOLD,
        upper_body_ratio=config.UPPER_BODY_RATIO,
        max_plate_distance=config.MAX_CYCLIST_PLATE_DISTANCE,
    )


def run_image(path: str, save: bool):
    pipeline = build_pipeline()
    image = cv2.imread(path)
    if image is None:
        print(f"❌ Cannot read image: {path}")
        return

    annotated, violations = pipeline.process_image(image, save_violations=save)
    out_path = "result.jpg"
    cv2.imwrite(out_path, annotated)

    print(f"✅ Detected {len(violations)} violation(s)")
    for i, v in enumerate(violations, 1):
        plate = v["plate_text"] or "N/A"
        print(f"   #{i}  conf={v['confidence']:.2f}  plate={plate}")
    print(f"📸 Annotated image saved → {out_path}")


def run_video(path: str, save: bool):
    pipeline = build_pipeline()
    out_path = "output_video.mp4"
    _, violations = pipeline.process_video(
        path, output_path=out_path,
        frame_skip=config.VIDEO_FRAME_SKIP,
        save_violations=save,
    )
    print(f"✅ Video processed — {len(violations)} violation(s) detected")
    print(f"🎬 Output saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HelmetGuard AI — detect helmet violations from images/videos"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image")
    group.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save violation records to disk")
    args = parser.parse_args()

    save = not args.no_save

    if args.image:
        run_image(args.image, save)
    else:
        run_video(args.video, save)


if __name__ == "__main__":
    main()
