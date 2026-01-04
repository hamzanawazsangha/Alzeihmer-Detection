// File Upload Handling
document.addEventListener('DOMContentLoaded', function() {
    // File upload functionality
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const previewContainer = document.getElementById('previewContainer');
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');

    if (uploadArea) {
        // Drag and drop functionality
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelection(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
        
        uploadBtn.addEventListener('click', uploadFile);
    }

    // Real-time detection canvas
    initializeCanvas();
});

function handleFileSelection(file) {
    if (!file.type.match('image.*')) {
        showAlert('Please select an image file (JPEG, PNG, etc.)', 'error');
        return;
    }
    
    const preview = document.getElementById('previewImage');
    const reader = new FileReader();
    
    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = 'block';
        uploadBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a file first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        if (data.success) {
            displayResults(data);
        } else {
            showAlert(data.error || 'Error processing image', 'error');
        }
    })
    .catch(error => {
        showLoading(false);
        showAlert('Network error: ' + error.message, 'error');
    });
}

function displayResults(data) {
    const resultsElement = document.getElementById('results');
    const confidenceColor = data.confidence > 90 ? 'success' : 
                           data.confidence > 75 ? 'warning' : 'error';
    
    resultsElement.innerHTML = `
        <div class="results">
            <div class="image-preview">
                <img src="${data.image_data}" alt="Uploaded MRI Scan">
            </div>
            <div class="prediction-result alert alert-${confidenceColor}">
                Diagnosis: <strong>${data.prediction}</strong>
            </div>
            <div class="confidence">
                Confidence: <strong>${data.confidence}%</strong>
            </div>
            <div class="tips-section">
                <h3>💡 Medical Recommendations</h3>
                <ul class="tips-list">
                    ${data.tips.map(tip => `<li>${tip}</li>`).join('')}
                </ul>
            </div>
            <button onclick="resetDetection()" class="btn btn-primary" style="margin-top: 1rem;">
                Analyze Another Image
            </button>
        </div>
    `;
    resultsElement.style.display = 'block';
}

function resetDetection() {
    document.getElementById('fileInput').value = '';
    document.getElementById('previewImage').style.display = 'none';
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('results').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
}

// Real-time Detection Canvas
function initializeCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    if (!canvas) return;
    
    let drawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Set canvas background to white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing functionality
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    predictBtn.addEventListener('click', predictDrawing);
    clearBtn.addEventListener('click', clearCanvas);
    
    function startDrawing(e) {
        drawing = true;
        [lastX, lastY] = getCoordinates(e);
    }
    
    function draw(e) {
        if (!drawing) return;
        
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        const [x, y] = getCoordinates(e);
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
    }
    
    function stopDrawing() {
        drawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        
        if (e.type === 'touchstart') {
            startDrawing(mouseEvent);
        } else if (e.type === 'touchmove') {
            draw(mouseEvent);
        }
    }
    
    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        let clientX, clientY;
        
        if (e.type.includes('touch')) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        
        return [
            (clientX - rect.left) * scaleX,
            (clientY - rect.top) * scaleY
        ];
    }
    
    function predictDrawing() {
        const imageData = canvas.toDataURL();
        
        showLoading(true);
        
        fetch('/realtime_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_data: imageData })
        })
        .then(response => response.json())
        .then(data => {
            showLoading(false);
            if (data.success) {
                displayRealTimeResults(data);
            } else {
                showAlert(data.error || 'Error processing drawing', 'error');
            }
        })
        .catch(error => {
            showLoading(false);
            showAlert('Network error: ' + error.message, 'error');
        });
    }
    
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        document.getElementById('realtimeResults').style.display = 'none';
    }
}

function displayRealTimeResults(data) {
    const resultsElement = document.getElementById('realtimeResults');
    const confidenceColor = data.confidence > 90 ? 'success' : 
                           data.confidence > 75 ? 'warning' : 'error';
    
    resultsElement.innerHTML = `
        <div class="results">
            <div class="prediction-result alert alert-${confidenceColor}">
                Predicted Condition: <strong>${data.prediction}</strong>
            </div>
            <div class="confidence">
                Confidence: <strong>${data.confidence}%</strong>
            </div>
            <div class="tips-section">
                <h3>💡 Medical Recommendations</h3>
                <ul class="tips-list">
                    ${data.tips.map(tip => `<li>${tip}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    resultsElement.style.display = 'block';
}

// Utility Functions
function showLoading(show) {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = show ? 'block' : 'none';
    }
}

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlert = document.querySelector('.alert');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    
    // Insert at the top of the main content
    const mainContent = document.querySelector('.container');
    mainContent.insertBefore(alert, mainContent.firstChild);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const header = document.querySelector('.main-header');

    // Toggle mobile menu
    hamburger.addEventListener('click', function() {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // Header scroll effect
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});