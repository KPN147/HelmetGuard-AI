"""
Gradio Web Interface for Helmet Violation Detection System
"""

import gradio as gr
import cv2
import numpy as np
from helmet_violation_detector import HelmetViolationDetector
import os
from pathlib import Path
import pandas as pd

# Check Gradio version compatibility
import importlib.metadata
gradio_version = importlib.metadata.version('gradio')
print(f"Using Gradio version: {gradio_version}")


class GradioHelmetApp:
    def __init__(self, model_path):
        """Initialize the Gradio application"""
        self.detector = HelmetViolationDetector(
            model_path=model_path,
            conf_threshold=0.5
        )
        

    def process_image_interface(self, image, conf_threshold, save_violations):
        if image is None:
            return None, "Please upload an image", None
        
        try:
            # Update confidence threshold
            self.detector.conf_threshold = conf_threshold
            
            # Convert RGB to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process image
            annotated_image, violations = self.detector.process_image(
                image_bgr, 
                save_violations=save_violations
            )
            
            # ✅ Kiểm tra annotated_image có None không
            if annotated_image is None:
                return image, "⚠️ Processing returned None - check detector code", None
            
            # Convert back to RGB
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # ✅ Đảm bảo là uint8
            if annotated_image_rgb.dtype != np.uint8:
                annotated_image_rgb = annotated_image_rgb.astype(np.uint8)
            
            # Generate violation report
            if violations:
                report = f"⚠️ DETECTED {len(violations)} HELMET VIOLATION(S)\n\n"
                for i, v in enumerate(violations, 1):
                    report += f"Violation #{i}:\n"
                    report += f"  • Timestamp: {v['timestamp']}\n"
                    report += f"  • Confidence: {v['confidence']:.2%}\n"
                    report += f"  • License Plate: {v['plate_text'] if v['plate_text'] else 'Not detected'}\n"
                    if save_violations:
                        report += f"  • Record saved ✓\n"
                    report += "\n"
                
                df_data = []
                for i, v in enumerate(violations, 1):
                    df_data.append({
                        'Violation #': i,
                        'Timestamp': v['timestamp'],
                        'Confidence': f"{v['confidence']:.2%}",
                        'License Plate': v['plate_text'] if v['plate_text'] else 'Not detected'
                    })
                violation_df = pd.DataFrame(df_data)
            else:
                report = "✅ No helmet violations detected!"
                violation_df = pd.DataFrame()
            
            return annotated_image_rgb, report, violation_df
        
        except Exception as e:
            print(f"❌ Error in process_image_interface: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", None
    
    def process_video_interface(self, video, conf_threshold, save_violations):
        """
        Process video through Gradio interface
        
        Args:
            video: Input video from Gradio
            conf_threshold: Confidence threshold
            save_violations: Whether to save violations
            
        Returns:
            output_video_path: Path to processed video
            summary_report: Text summary of violations
        """
        if video is None:
            return None, "Please upload a video"
        
        # Update confidence threshold
        self.detector.conf_threshold = conf_threshold
        
        # Open video
        cap = cv2.VideoCapture(video)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video path
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_violations = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame (or skip frames for faster processing)
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                annotated_frame, violations = self.detector.process_image(
                    frame, 
                    save_violations=save_violations
                )
                
                if violations:
                    for v in violations:
                        v['frame'] = frame_count
                    total_violations.extend(violations)
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        # Generate summary report
        if total_violations:
            unique_plates = set(v['plate_text'] for v in total_violations if v['plate_text'])
            report = f"📹 VIDEO PROCESSING COMPLETE\n\n"
            report += f"Total Frames: {frame_count}\n"
            report += f"Total Violations Detected: {len(total_violations)}\n"
            report += f"Unique License Plates: {len(unique_plates)}\n\n"
            
            if unique_plates:
                report += "Detected License Plates:\n"
                for plate in unique_plates:
                    report += f"  • {plate}\n"
        else:
            report = "✅ No helmet violations detected in video!"
        
        return output_path, report
    
    def launch(self):
        """Launch Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        """
        
        # Create Gradio interface using Blocks
        demo = gr.Blocks(title="Helmet Violation Detection System", css=custom_css)
        
        with demo:
            gr.Markdown(
                """
                # 🛵 Helmet Violation Detection System
                
                **Automated detection of cyclists without helmets with license plate recognition**
                
                This system uses YOLOv11 for object detection and PaddleOCR for license plate reading.
                """
            )
            
            with gr.Tabs():
                # Image Processing Tab
                with gr.Tab("📷 Image Processing"):
                    gr.Markdown("### Upload an image to detect helmet violations")
                    
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                label="Upload Image",
                                type="numpy"
                            )
                            
                            conf_slider_img = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.05,
                                label="Confidence Threshold"
                            )
                            
                            save_violations_img = gr.Checkbox(
                                label="Save Violation Records",
                                value=True
                            )
                            
                            process_btn_img = gr.Button(
                                "🔍 Detect Violations",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            output_image = gr.Image(
                                label="Detection Results",
                                type="numpy",
                                format="png",
                                show_label=True
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            violation_report_img = gr.Textbox(
                                label="Violation Report",
                                lines=10,
                                max_lines=15
                            )
                        
                        with gr.Column():
                            violation_table = gr.Dataframe(
                                label="Violation Details",
                                headers=['Violation #', 'Timestamp', 'Confidence', 'License Plate']
                            )
                    
                    process_btn_img.click(
                        fn=self.process_image_interface,
                        inputs=[image_input, conf_slider_img, save_violations_img],
                        outputs=[output_image, violation_report_img, violation_table]
                    )
                    
                    gr.Markdown(
                        """
                        ### Instructions:
                        1. Upload an image containing cyclists
                        2. Adjust confidence threshold if needed (lower = more detections, higher = more accurate)
                        3. Check "Save Violation Records" to save detected violations to disk
                        4. Click "Detect Violations" to process
                        
                        **Legend:**
                        - 🟢 Green box: Cyclist wearing helmet (OK)
                        - 🔴 Red box: Cyclist without helmet (VIOLATION)
                        - 🟡 Yellow box: Detected helmet
                        - 🔵 Blue box: License plate
                        """
                    )
                
                # Video Processing Tab
                with gr.Tab("🎥 Video Processing"):
                    gr.Markdown("### Upload a video to detect helmet violations")
                    
                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Video(
                                label="Upload Video"
                            )
                            
                            conf_slider_video = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.05,
                                label="Confidence Threshold"
                            )
                            
                            save_violations_video = gr.Checkbox(
                                label="Save Violation Records",
                                value=True
                            )
                            
                            process_btn_video = gr.Button(
                                "🔍 Process Video",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            output_video = gr.Video(
                                label="Processed Video"
                            )
                    
                    video_report = gr.Textbox(
                        label="Video Analysis Report",
                        lines=10,
                        max_lines=15
                    )
                    
                    process_btn_video.click(
                        fn=self.process_video_interface,
                        inputs=[video_input, conf_slider_video, save_violations_video],
                        outputs=[output_video, video_report]
                    )
                    
                    gr.Markdown(
                        """
                        ### Instructions:
                        1. Upload a video file
                        2. Adjust confidence threshold
                        3. Click "Process Video" (this may take a few minutes)
                        4. Download the processed video with annotations
                        
                        **Note:** Video processing analyzes every 5th frame for efficiency.
                        """
                    )
                
                # Information Tab
                with gr.Tab("ℹ️ Information"):
                    gr.Markdown(
                        """
                        ## System Information
                        
                        ### How It Works
                        
                        1. **Object Detection (YOLOv11)**
                           - Detects cyclists, helmets, and license plates in images/videos
                           - Uses spatial analysis to determine if cyclists are wearing helmets
                        
                        2. **Helmet Violation Logic**
                           - Checks if a helmet is present in the upper body region of each cyclist
                           - If no helmet detected → Flags as violation
                        
                        3. **License Plate Recognition (PaddleOCR)**
                           - Automatically reads license plates near violators
                           - Stores plate text for violation records
                        
                        4. **Violation Recording**
                           - Saves cropped images of violators
                           - Saves license plate images
                           - Creates timestamped violation reports
                        
                        ### Output Structure
                        
                        When violations are saved, they are stored in:
                        ```
                        violation_records/
                        ├── 2025-11-11_14-30-45/
                        │   ├── violator.jpg
                        │   ├── license_plate.jpg
                        │   └── violation_details.txt
                        └── 2025-11-11_14-32-10/
                            ├── violator.jpg
                            └── violation_details.txt
                        ```
                        
                        ### Model Classes
                        - **Class 0:** Helmet
                        - **Class 1:** Cyclist
                        - **Class 2:** License Plate
                        
                        ### Tips for Best Results
                        - Use clear, well-lit images
                        - Ensure license plates are visible and not too small
                        - Adjust confidence threshold based on your needs:
                          - Lower (0.3-0.4): More detections, some false positives
                          - Medium (0.5): Balanced
                          - Higher (0.6-0.7): Fewer detections, higher accuracy
                        
                        ### Requirements
                        - Python 3.8+
                        - YOLOv11 (ultralytics)
                        - PaddleOCR
                        - OpenCV
                        - Gradio
                        
                        ---
                        
                        **Developed for Traffic Safety Enforcement**
                        """
                    )
            
            gr.Markdown(
                """
                ---
                <center>
                <small>⚖️ For authorized traffic enforcement use only. Ensure compliance with local privacy and data protection laws.</small>
                </center>
                """
            )
        
        # Launch the app
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )


def main():
    # Path to your trained YOLOv11 model
    MODEL_PATH = './best.pt'  # Change this to your model path
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please update MODEL_PATH in gradio_app.py to point to your YOLOv11 model.")
        return
    
    # Create and launch the app
    app = GradioHelmetApp(model_path=MODEL_PATH)
    print("🚀 Launching Helmet Violation Detection System...")
    print("📱 Open your browser and go to the URL shown below")
    app.launch()


if __name__ == '__main__':
    main()