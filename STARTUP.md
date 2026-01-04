# 🚀 NeuroScan AI - Quick Startup Guide

This guide will help you get the Alzheimer's Detection System up and running in minutes.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] pip (Python package manager)
- [ ] 4GB+ RAM available
- [ ] 2GB+ free disk space
- [ ] Internet connection (for initial setup)

## ⚡ Quick Start (5 Minutes)

### Step 1: Verify Python Installation

```bash
python --version
# Should show Python 3.8 or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Mac**: `brew install python3`
- **Linux**: `sudo apt-get install python3`

### Step 2: Navigate to Project Directory

```bash
cd path/to/Alzehiemer_Detectection_System
```

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow
- Flask
- NumPy
- Pillow
- And other required packages

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 5: Verify Model File

Ensure the model file exists:
```
model/Alzheimer_Detection_model.h5
```

If missing, contact the project maintainer for the trained model file.

### Step 6: Start the Application

```bash
python app.py
```

You should see output like:
```
✅ Model loaded successfully!
   Model input shape: (None, 224, 224, 3)
   Model output shape: (None, 4)
 * Running on http://127.0.0.1:5000
```

### Step 7: Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

🎉 **Congratulations!** The system is now running.

## 🖥️ Using the System

### 1. Homepage
- View system overview
- Check performance metrics
- Navigate to detection page

### 2. Upload MRI Image
- Click "Detection" in the navigation
- Drag & drop or click to upload an MRI scan
- Supported formats: JPEG, PNG
- Max size: 16MB

### 3. Analyze
- Click "Analyze Image" button
- Wait 2-5 seconds for processing
- View results with confidence scores

### 4. Interpret Results
The system classifies into 4 categories:
- **Non-Demented**: No cognitive impairment
- **Very Mild Demented**: Early-stage decline
- **Mild Demented**: Moderate impairment
- **Moderate Demented**: Significant decline

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "Model file not found"

**Solution:**
- Verify `model/Alzheimer_Detection_model.h5` exists
- Check file path in `app.py` (line ~30)
- Ensure model file is not corrupted

### Issue: Port 5000 already in use

**Solution:**
```bash
# Change port in app.py (last line)
app.run(debug=True, port=5001)  # Use different port
```

### Issue: Out of memory

**Solution:**
- Close other applications
- Restart your computer
- Ensure 4GB+ RAM available

### Issue: Slow inference

**Solution:**
- Check CPU usage
- Ensure model is loaded correctly
- Try smaller image sizes
- Consider GPU acceleration (requires TensorFlow-GPU)

## 📦 Dependencies

### Core Requirements
```
tensorflow>=2.8.0
flask>=2.0.0
numpy>=1.21.0
pillow>=9.0.0
werkzeug>=2.0.0
```

### Optional (for development)
```
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
```

## 🔄 Updating the System

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Pull Latest Changes (if using Git)
```bash
git pull origin main
pip install -r requirements.txt
```

## 🛑 Stopping the Application

### Method 1: Keyboard Interrupt
Press `Ctrl + C` in the terminal

### Method 2: Close Terminal
Simply close the terminal window

### Deactivate Virtual Environment
```bash
deactivate
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Windows 10, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: Quad-core 2.5 GHz or higher
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional)

## 🌐 Network Configuration

### Running on Local Network

To access from other devices on your network:

```python
# In app.py, change the last line to:
app.run(debug=True, host='0.0.0.0', port=5000)
```

Then access from other devices using:
```
http://YOUR_LOCAL_IP:5000
```

Find your local IP:
- **Windows**: `ipconfig`
- **Mac/Linux**: `ifconfig` or `ip addr`

### Security Warning
⚠️ Only use `host='0.0.0.0'` on trusted networks!

## 📝 Configuration Options

### Change Upload Folder
```python
# In app.py
UPLOAD_FOLDER = 'your/custom/path'
```

### Change Maximum File Size
```python
# In app.py
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

### Enable Debug Mode
```python
# In app.py (last line)
app.run(debug=True)  # Shows detailed errors
```

### Disable Debug Mode (Production)
```python
# In app.py (last line)
app.run(debug=False)  # Hide error details
```

## 🧪 Testing the Installation

### Test 1: Check Model Loading
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### Test 2: Verify Flask
```bash
python -c "import flask; print('Flask:', flask.__version__)"
```

### Test 3: Test Image Processing
```bash
python -c "from PIL import Image; print('Pillow: OK')"
```

### Test 4: Full System Test
1. Start the application
2. Upload a test MRI image
3. Verify prediction is returned
4. Check console for any errors

## 📱 Mobile Access

The web interface is mobile-responsive. Access from your phone:
1. Ensure phone is on same Wi-Fi network
2. Use your computer's local IP address
3. Navigate to `http://LOCAL_IP:5000`

## 🔐 Security Considerations

### For Development
- ✅ Debug mode enabled
- ✅ Detailed error messages
- ✅ Auto-reload on code changes

### For Production
- ❌ Disable debug mode
- ✅ Use production WSGI server (Gunicorn, uWSGI)
- ✅ Enable HTTPS
- ✅ Implement authentication
- ✅ Rate limiting
- ✅ Input validation

## 📚 Additional Resources

### Documentation
- **README.md**: Comprehensive project documentation
- **Code Comments**: Inline documentation in source files
- **API Docs**: Available at `/developer` route

### Getting Help
1. Check troubleshooting section above
2. Review error messages in terminal
3. Check browser console (F12)
4. Open an issue on GitHub
5. Contact the developer

## 🎯 Next Steps

After successful startup:

1. **Explore the Interface**
   - Navigate through all pages
   - Test with sample images
   - Review documentation

2. **Understand the Code**
   - Read `app.py` for backend logic
   - Check `utils/image_processor.py` for preprocessing
   - Review templates for frontend

3. **Customize**
   - Modify CSS for different styling
   - Add new features
   - Integrate with other systems

4. **Deploy** (Optional)
   - Set up on cloud platform (AWS, Azure, GCP)
   - Configure domain name
   - Enable HTTPS
   - Set up monitoring

## ⚠️ Important Notes

### Medical Disclaimer
This system is for **educational and research purposes only**. It should NOT replace professional medical diagnosis. Always consult qualified healthcare professionals.

### Data Privacy
- Images are processed locally
- No data is stored permanently (unless configured)
- Temporary uploads are cleared on restart
- Implement proper data handling for production use

### Model Limitations
- Trained on specific dataset
- May not generalize to all MRI types
- Requires proper image quality
- Should be validated by medical professionals

## 🔄 Regular Maintenance

### Weekly
- Check for dependency updates
- Review error logs
- Test with new images

### Monthly
- Update dependencies
- Review security patches
- Backup model and data

### As Needed
- Retrain model with new data
- Update documentation
- Add new features

## 📞 Support

If you encounter issues:

1. **Check Logs**: Review terminal output for errors
2. **Browser Console**: Press F12 and check for JavaScript errors
3. **Dependencies**: Ensure all packages are installed correctly
4. **Model File**: Verify model file is present and not corrupted
5. **Contact**: Reach out via GitHub issues or LinkedIn

## ✅ Startup Checklist

Before considering the system ready:

- [ ] Python 3.8+ installed and verified
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Model file present and loaded
- [ ] Application starts without errors
- [ ] Homepage loads in browser
- [ ] Test image uploads successfully
- [ ] Prediction returns results
- [ ] All pages accessible
- [ ] No console errors

---

**Ready to detect Alzheimer's with AI!** 🧠✨

*For detailed information, see README.md*

*Last Updated: January 2026*

