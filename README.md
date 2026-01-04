# 🧠 NeuroScan AI - Alzheimer's Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art deep learning system for detecting and classifying Alzheimer's disease from MRI brain scans using EfficientNetB0 transfer learning architecture.

## 🎯 Project Overview

NeuroScan AI is an advanced medical image analysis system that leverages deep learning to classify Alzheimer's disease stages from MRI scans with **99.2% accuracy**. The system uses transfer learning with EfficientNetB0 architecture, fine-tuned on a comprehensive dataset of brain MRI images.

### Key Features

- 🎯 **High Accuracy**: 99.2% classification accuracy with 1.00 AUC score
- 🧠 **4-Class Classification**: Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented
- 🚀 **Real-time Analysis**: Fast inference with optimized model architecture
- 💻 **User-Friendly Interface**: Modern web interface with intuitive design
- 📊 **Detailed Results**: Comprehensive analysis with confidence scores and recommendations
- 🔒 **Secure Processing**: Client-side image validation and server-side security

## 📋 Table of Contents

- [Technology Stack](#technology-stack)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **Flask**: Web application framework
- **NumPy**: Numerical computing
- **Pillow (PIL)**: Image processing

### Frontend
- **HTML5**: Markup structure
- **CSS3**: Modern styling with animations
- **JavaScript (ES6+)**: Interactive functionality
- **Font Awesome**: Icon library
- **Google Fonts**: Typography (Inter)

### Model
- **EfficientNetB0**: Base architecture (transfer learning)
- **ImageNet**: Pre-trained weights
- **Custom Classification Head**: Fine-tuned for Alzheimer's detection

## 🧠 Model Architecture

### Base Model: EfficientNetB0
```
Input Layer: (224, 224, 3)
    ↓
EfficientNetB0 (Pre-trained on ImageNet)
    ↓
Fine-tuning from: block5a_expand_activation
    ↓
Global Average Pooling 2D
    ↓
Batch Normalization
    ↓
Dense Layer (512 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Batch Normalization
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (4 units, Softmax)
```

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Batch Size**: 32
- **Image Size**: 224x224x3
- **Data Augmentation**: Rotation, Zoom, Flip, Shift

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.2% |
| **Precision** | 99.1% |
| **Recall** | 99.0% |
| **F1-Score** | 99.0% |
| **AUC-ROC** | 1.00 |

### Classification Classes
1. **Non-Demented**: No signs of cognitive impairment
2. **Very Mild Demented**: Early-stage cognitive decline
3. **Mild Demented**: Moderate cognitive impairment
4. **Moderate Demented**: Significant cognitive decline

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/alzheimer-detection-system.git
cd alzheimer-detection-system
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model File
Ensure the trained model file is present:
```
model/Alzheimer_Detection_model.h5
```

### Step 5: Run the Application
```bash
python app.py
```

The application will start at: `http://localhost:5000`

## 💻 Usage

### Web Interface

1. **Navigate to Homepage**
   - Open your browser and go to `http://localhost:5000`
   - View system overview and features

2. **Upload MRI Scan**
   - Click on "Detection" or "Try Demo"
   - Upload a brain MRI image (JPEG, PNG)
   - Supported formats: JPEG, PNG
   - Maximum file size: 16MB

3. **Analyze Image**
   - Click "Analyze Image" button
   - Wait for processing (typically 2-5 seconds)
   - View detailed results

4. **Review Results**
   - Classification result with confidence score
   - All class probabilities
   - Recommendations based on diagnosis
   - Option to analyze another image

### API Usage

#### Upload and Analyze Endpoint
```python
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)

Response:
{
    "prediction": "Mild Cognitive Impairment",
    "confidence": 95.67,
    "all_confidences": {
        "Non-Demented": 2.1,
        "Very Mild Demented": 1.23,
        "Mild Demented": 95.67,
        "Moderate Demented": 1.0
    },
    "recommendation": "Consult with a healthcare professional..."
}
```

## 📁 Project Structure

```
alzheimer-detection-system/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── STARTUP.md                  # Startup guide
│
├── model/
│   └── Alzheimer_Detection_model.h5    # Trained model
│
├── utils/
│   └── image_processor.py      # Image preprocessing utilities
│
├── static/
│   ├── css/
│   │   ├── style.css          # Main stylesheet
│   │   └── style-3d.css       # 3D effects stylesheet
│   ├── js/
│   │   ├── script.js          # Main JavaScript
│   │   └── 3d-effects.js      # 3D animations
│   └── uploads/               # Temporary upload directory
│
└── templates/
    ├── index.html             # Homepage
    ├── detection.html         # Detection page
    ├── results.html           # Results page
    ├── realtime.html          # Real-time analysis
    └── developers.html        # Documentation page
```

## 🔌 API Documentation

### Endpoints

#### 1. Home Page
```
GET /
Returns: Homepage with system overview
```

#### 2. Detection Page
```
GET /detection
Returns: Image upload and analysis interface
```

#### 3. Upload and Analyze
```
POST /upload
Content-Type: multipart/form-data
Body: file (image)
Returns: JSON with prediction results
```

#### 4. Results Page
```
GET /results
Returns: Detailed analysis results page
```

#### 5. Documentation
```
GET /developer
Returns: API and system documentation
```

## 🎨 Features

### Image Processing Pipeline
1. **Validation**: File type and size verification
2. **Preprocessing**: 
   - Resize to 224x224
   - RGB conversion
   - Contrast enhancement
   - Noise reduction
3. **Normalization**: EfficientNet-specific preprocessing
4. **Inference**: Model prediction with confidence scores

### Security Features
- File type validation (JPEG, PNG only)
- File size limits (16MB maximum)
- Secure filename handling
- Input sanitization
- Error handling and logging

### User Experience
- Drag-and-drop file upload
- Real-time preview
- Loading animations
- Detailed error messages
- Responsive design
- Mobile-friendly interface

## 🧪 Testing

### Manual Testing
1. Test with various MRI image formats
2. Verify classification accuracy
3. Check edge cases (corrupted images, wrong formats)
4. Test responsive design on different devices

### Performance Testing
- Average inference time: 2-3 seconds
- Maximum concurrent users: 50+
- Memory usage: ~2GB with model loaded

## 🔧 Configuration

### Environment Variables
Create a `.env` file (optional):
```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
MAX_CONTENT_LENGTH=16777216
```

### Model Configuration
Edit `app.py` to modify:
- Upload folder path
- Allowed file extensions
- Maximum file size
- Model path

## 📈 Future Enhancements

- [ ] Multi-model ensemble for improved accuracy
- [ ] 3D MRI scan support (DICOM format)
- [ ] Batch processing for multiple images
- [ ] User authentication and history
- [ ] Export reports as PDF
- [ ] Integration with PACS systems
- [ ] Mobile application (iOS/Android)
- [ ] Cloud deployment (AWS/Azure/GCP)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Write unit tests for new features
- Update documentation

## ⚠️ Disclaimer

**Important**: This system is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Muhammad Hamza Nawaz**
- GitHub: [@hamzanawazsangha](https://github.com/hamzanawazsangha/)
- LinkedIn: [Muhammad Hamza Nawaz](https://www.linkedin.com/in/muhammad-hamza-nawaz-a434501b3/)
- Instagram: [@iam_hamzanawaz](https://instagram.com/iam_hamzanawaz)

## 🙏 Acknowledgments

- Dataset: Alzheimer's Disease Neuroimaging Initiative (ADNI)
- EfficientNet: Google Research
- TensorFlow and Keras teams
- Flask framework developers
- Open-source community

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact via LinkedIn
- Email: [iamhamzanawaz14@gmail.com]

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐

---

**Made with ❤️ for advancing AI in healthcare**

*Last Updated: January 2026*
