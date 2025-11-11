# 🛵 Helmet Violation Detection System

An automated system to detect cyclists without helmets, capture violator images, and read license plates using YOLOv11 and PaddleOCR, deployed with a Gradio web interface.

## 🎯 Features

- **Real-time Detection**: Detect cyclists, helmets, and license plates using YOLOv11
- **Violation Logic**: Automatically identifies cyclists not wearing helmets
- **License Plate Recognition**: Uses PaddleOCR to read license plates of violators
- **Evidence Recording**: Saves timestamped violation records with images
- **Web Interface**: User-friendly Gradio interface for image and video processing
- **Batch Processing**: Process videos frame-by-frame for comprehensive monitoring

## 📋 System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 4GB RAM minimum (8GB recommended)

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# If you have git
git clone <your-repo-url>
cd helmet-violation-detection

# Or download and extract the files
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PaddleOCR installation, try:
```bash
# For CPU version
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# For GPU version (CUDA 11.2)
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

### Step 4: Prepare Your Model

Place your trained YOLOv11 model file in the project directory:

```bash
# Your model file should be named:
best.pt

# Or update the MODEL_PATH in gradio_app.py to match your model filename
```

## 📁 Project Structure

```
helmet-violation-detection/
├── helmet_violation_detector.py   # Core detection logic
├── gradio_app.py                  # Gradio web interface
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── best.pt                        # Your YOLOv11 model (you need to add this)
└── violation_records/             # Generated violation records (created automatically)
    └── YYYY-MM-DD_HH-MM-SS/
        ├── violator.jpg
        ├── license_plate.jpg
        └── violation_details.txt
```

## 🎮 Usage

### Starting the Web Interface

```bash
python gradio_app.py
```

The system will start and display a URL (typically `http://127.0.0.1:7860`). Open this URL in your web browser.

### Using the Interface

#### Image Processing
1. Navigate to the **"Image Processing"** tab
2. Upload an image containing cyclists
3. Adjust the confidence threshold (0.5 is default)
4. Enable/disable "Save Violation Records"
5. Click **"Detect Violations"**
6. Review the annotated image and violation report

#### Video Processing
1. Navigate to the **"Video Processing"** tab
2. Upload a video file
3. Adjust the confidence threshold
4. Click **"Process Video"** (may take several minutes)
5. Download the processed video with annotations
6. Review the summary report

### Command Line Usage

You can also use the detector programmatically:

```python
from helmet_violation_detector import HelmetViolationDetector
import cv2

# Initialize detector
detector = HelmetViolationDetector(
    model_path='best.pt',
    conf_threshold=0.5
)

# Process an image
image = cv2.imread('test_image.jpg')
annotated_image, violations = detector.process_image(image)

# Print violations
print(f"Detected {len(violations)} violations")
for i, v in enumerate(violations, 1):
    print(f"Violation {i}:")
    print(f"  Timestamp: {v['timestamp']}")
    print(f"  License Plate: {v['plate_text']}")

# Save result
cv2.imwrite('result.jpg', annotated_image)
```

## 🔧 Configuration

### Model Classes

The system expects your YOLOv11 model to have the following classes:
- **Class 0**: Helmet
- **Class 1**: Cyclist
- **Class 2**: License Plate

If your model uses different class IDs, update the `class_names` dictionary in `helmet_violation_detector.py`:

```python
self.class_names = {
    0: 'helmet',
    1: 'cyclist', 
    2: 'license_plate'
}
```

### Detection Parameters

Adjust these parameters in the code or via the Gradio interface:

- **Confidence Threshold** (0.1-0.9): Minimum confidence for detections
  - Lower: More detections, may include false positives
  - Higher: Fewer detections, higher accuracy

- **IOU Threshold** (default 0.45): For non-maximum suppression
- **Helmet Association IOU** (default 0.3): Threshold for helmet-cyclist matching

### OCR Settings

PaddleOCR settings in `helmet_violation_detector.py`:

```python
self.ocr = PaddleOCR(
    use_angle_cls=True,  # Enable angle classification
    lang='en',           # Language (change to your region)
    show_log=False       # Disable verbose logging
)
```

## 📊 Output Format

### Violation Records

When violations are detected and saved, the system creates:

**Directory**: `violation_records/YYYY-MM-DD_HH-MM-SS/`

**Files**:
- `violator.jpg`: Cropped image of the cyclist without helmet
- `license_plate.jpg`: Cropped image of the license plate (if detected)
- `violation_details.txt`: Text file with violation information

**Example violation_details.txt**:
```
Timestamp: 2025-11-11 14:30:45
Confidence: 0.87
License Plate: ABC1234
```

## 🎨 Detection Visualization

The system uses color-coded bounding boxes:

- 🟢 **Green**: Cyclist wearing helmet (OK)
- 🔴 **Red**: Cyclist without helmet (VIOLATION)
- 🟡 **Yellow**: Detected helmet or license plate (not associated with violation)
- 🔵 **Blue**: License plate associated with violation

## ⚙️ Advanced Usage

### Batch Processing Multiple Images

```python
import glob
from helmet_violation_detector import HelmetViolationDetector
import cv2

detector = HelmetViolationDetector('best.pt')

for img_path in glob.glob('images/*.jpg'):
    image = cv2.imread(img_path)
    annotated, violations = detector.process_image(image)
    
    if violations:
        output_path = f'results/{os.path.basename(img_path)}'
        cv2.imwrite(output_path, annotated)
        print(f"Processed {img_path}: {len(violations)} violations")
```

### Custom Violation Criteria

Modify the `is_cyclist_wearing_helmet` method to adjust the helmet detection logic:

```python
def is_cyclist_wearing_helmet(self, cyclist_box, helmet_boxes, iou_threshold=0.3):
    # Adjust upper_body_height for different helmet positioning
    upper_body_height = (cyclist_y2 - cyclist_y1) * 0.4  # Change 0.4 to suit your needs
    # ... rest of the logic
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model file not found
```
Error: Model file 'best.pt' not found!
```
**Solution**: Make sure your YOLOv11 model is in the project directory or update `MODEL_PATH` in `gradio_app.py`

**Issue**: PaddleOCR installation fails
**Solution**: 
```bash
# Try installing with specific version
pip install paddlepaddle==2.5.0 -i https://mirror.baidu.com/pypi/simple
pip install paddleocr==2.7.0
```

**Issue**: CUDA out of memory
**Solution**: 
- Reduce image/video resolution
- Process fewer frames in videos (increase the skip interval)
- Use CPU version of PaddlePaddle

**Issue**: OCR not reading license plates correctly
**Solution**:
- Ensure license plates are clear and well-lit in images
- Increase image resolution
- Adjust OCR language setting in code: `lang='en'` to your language
- Try preprocessing the license plate images (contrast enhancement)

**Issue**: Too many false positives
**Solution**:
- Increase confidence threshold
- Retrain your model with more diverse data
- Adjust the IOU threshold for helmet-cyclist association

## 📈 Performance Tips

1. **GPU Acceleration**: Install CUDA and use GPU version of libraries for faster processing
2. **Batch Processing**: Process multiple images/frames together for efficiency
3. **Frame Skipping**: For videos, process every Nth frame (adjust in `process_video_interface`)
4. **Resolution**: Resize large images to 1280x720 or lower if accuracy permits

## 🔐 Legal and Privacy Considerations

**⚠️ Important**: This system captures and stores personal data. Ensure compliance with:

- Local traffic enforcement regulations
- Data protection laws (GDPR, CCPA, etc.)
- Privacy regulations regarding image capture and storage
- License plate data retention policies

Always obtain proper authorization before deploying this system for surveillance or enforcement purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Support for multiple language license plates
- Real-time streaming video processing
- Database integration for violation management
- Mobile app interface
- Advanced analytics and reporting
- Multi-object tracking for better video analysis

## 📝 License

This project is for educational and authorized traffic enforcement purposes only.

## 👤 Author

Your Name - Traffic Safety AI System

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv11 nano object detection
- **PaddlePaddle**: PaddleOCR for license plate recognition
- **Gradio**: Web interface framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments
3. Open an issue on GitHub (if applicable)


