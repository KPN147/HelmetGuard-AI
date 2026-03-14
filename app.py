"""
app.py — Gradio web interface for HelmetGuard AI.

Usage:
    python app.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import gradio as gr

import config
from src.pipeline import HelmetViolationPipeline
from src.model_loader import get_model_path


class GradioHelmetApp:
    """Gradio front-end that delegates to HelmetViolationPipeline."""

    def __init__(self):
        model_path = get_model_path(
            repo_id=config.HF_REPO_ID,
            filename=config.HF_MODEL_FILENAME,
            local_dir=config.LOCAL_MODEL_DIR,
        )
        self.pipeline = HelmetViolationPipeline(
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

    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #
    def process_image(self, image, conf_threshold, save_violations):
        if image is None:
            return None, "Please upload an image.", None

        try:
            self.pipeline.conf_threshold = conf_threshold
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            annotated, violations = self.pipeline.process_image(
                image_bgr, save_violations=save_violations
            )

            if annotated is None:
                return image, "⚠️ Processing returned None.", None

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB).astype(np.uint8)

            # Build report
            if violations:
                report = f"⚠️ DETECTED {len(violations)} HELMET VIOLATION(S)\n\n"
                rows = []
                for i, v in enumerate(violations, 1):
                    plate = v["plate_text"] or "Not detected"
                    report += (f"Violation #{i}:\n"
                               f"  • Timestamp: {v['timestamp']}\n"
                               f"  • Confidence: {v['confidence']:.2%}\n"
                               f"  • License Plate: {plate}\n\n")
                    rows.append({
                        "Violation #": i,
                        "Timestamp": v["timestamp"],
                        "Confidence": f"{v['confidence']:.2%}",
                        "License Plate": plate,
                    })
                df = pd.DataFrame(rows)
            else:
                report = "✅ No helmet violations detected!"
                df = pd.DataFrame()

            return annotated_rgb, report, df

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {e}", None

    # ------------------------------------------------------------------ #
    # Video
    # ------------------------------------------------------------------ #
    def process_video(self, video, conf_threshold, save_violations):
        if video is None:
            return None, "Please upload a video."

        self.pipeline.conf_threshold = conf_threshold
        out_path, violations = self.pipeline.process_video(
            video, output_path="output_video.mp4",
            frame_skip=config.VIDEO_FRAME_SKIP,
            save_violations=save_violations,
        )

        if violations:
            plates = {v["plate_text"] for v in violations if v["plate_text"]}
            report = (f"📹 VIDEO PROCESSING COMPLETE\n\n"
                      f"Total Violations: {len(violations)}\n"
                      f"Unique Plates: {len(plates)}\n\n")
            if plates:
                report += "Detected Plates:\n"
                for p in plates:
                    report += f"  • {p}\n"
        else:
            report = "✅ No helmet violations detected in video!"

        return out_path, report

    # ------------------------------------------------------------------ #
    # Launch
    # ------------------------------------------------------------------ #
    def launch(self):
        with gr.Blocks(title="HelmetGuard AI") as demo:
            gr.Markdown(
                "# 🛵 HelmetGuard AI\n"
                "**Automated helmet-violation detection with license-plate OCR**"
            )

            with gr.Tabs():
                # --- Image tab ---
                with gr.Tab("📷 Image"):
                    with gr.Row():
                        with gr.Column():
                            img_in = gr.Image(label="Upload Image", type="numpy")
                            conf_img = gr.Slider(0.1, 0.9, 0.5, step=0.05,
                                                 label="Confidence")
                            save_img = gr.Checkbox(label="Save Records", value=True)
                            btn_img = gr.Button("🔍 Detect", variant="primary")
                        with gr.Column():
                            img_out = gr.Image(label="Result", type="numpy",
                                               format="png")
                    with gr.Row():
                        report_img = gr.Textbox(label="Report", lines=8)
                        table_img = gr.Dataframe(label="Details")
                    btn_img.click(self.process_image,
                                  [img_in, conf_img, save_img],
                                  [img_out, report_img, table_img])

                # --- Video tab ---
                with gr.Tab("🎥 Video"):
                    with gr.Row():
                        with gr.Column():
                            vid_in = gr.Video(label="Upload Video")
                            conf_vid = gr.Slider(0.1, 0.9, 0.5, step=0.05,
                                                 label="Confidence")
                            save_vid = gr.Checkbox(label="Save Records", value=True)
                            btn_vid = gr.Button("🔍 Process", variant="primary")
                        with gr.Column():
                            vid_out = gr.Video(label="Processed Video")
                    report_vid = gr.Textbox(label="Report", lines=8)
                    btn_vid.click(self.process_video,
                                  [vid_in, conf_vid, save_vid],
                                  [vid_out, report_vid])

        demo.launch(server_name=config.GRADIO_SERVER_NAME,
                     server_port=config.GRADIO_SERVER_PORT,
                     share=config.GRADIO_SHARE)


def main():
    print("🚀 Launching HelmetGuard AI …")
    GradioHelmetApp().launch()


if __name__ == "__main__":
    main()
