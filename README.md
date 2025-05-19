# Diagnosica
Simple Flask app for brain tumor detection from MRI images using a pre-trained deep learning model.

## Features
- Upload MRI images through a clean web interface
- Get instant tumor prediction with confidence score
- Responsive UI with Google Fonts
- CORS enabled for flexibility

## Installation
1. Clone repository:
   git clone <your-repo-url>
   cd <your-repo-folder>/backend

2. Create virtual environment:
   python3 -m venv tf-venv
   source tf-venv/bin/activate  # macOS/Linux
   .\tf-venv\Scripts\activate   # Windows

3. Install dependencies:
   pip install -r requirements.txt
   Requirements: Flask, flask-cors, numpy, Pillow, tensorflow

4. Place `brain_tumor_model.h5` in the `model/` directory.

## Usage
Run the app:
   python app.py
Visit http://127.0.0.1:5000 in your browser.

## How It Works
- Images are resized to 224x224 RGB arrays
- Model outputs probability between 0-1
- Score interpretation:
  - <0.5: No tumor
  - ≥0.5: Tumor detected

## Troubleshooting
- Check dependencies are installed in your virtual environment
- Verify Python 3.8+ is used
- Confirm model path is correct
- Check terminal for error messages

## Future Improvements
- Support for multiple image formats
- Add segmentation overlays
- Cloud deployment
- Enhanced UI with React/Vue
- User authentication