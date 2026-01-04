from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json
from utils.image_processor import AlzheimerImageProcessor, image_processor
import hashlib
import time

# Force unbuffered output for debugging
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

app = Flask(__name__)
app.secret_key = 'neuroscan-ai-alzheimer-detection-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Load the trained model - OPTIMIZED
try:
    # Disable eager execution for faster inference
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation for faster execution
    
    print("⏳ Loading model...")
    model = tf.keras.models.load_model('model/Alzheimer_Detection_model.h5', compile=False)
    
    # Ensure model is in inference mode
    model.trainable = False
    
    print("✅ Model loaded successfully!")
    print(f"   Input: {model.input_shape} → Output: {model.output_shape}")
    
    # Warm up the model with a single prediction (makes subsequent predictions faster)
    print("⏳ Warming up model...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_input = tf.keras.applications.efficientnet.preprocess_input(dummy_input)
    _ = model(dummy_input, training=False)  # Warm-up call
    print("✅ Model ready for predictions")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Class names
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Medical tips for each condition
MEDICAL_TIPS = {
    'Mild Impairment': [
        "Engage in regular cognitive exercises like puzzles and memory games",
        "Maintain a healthy diet rich in omega-3 fatty acids and antioxidants",
        "Stay socially active and engage in group activities regularly",
        "Consider cognitive therapy and schedule regular medical check-ups",
        "Practice stress management techniques like meditation and yoga",
        "Establish a consistent daily routine to reduce confusion",
        "Use memory aids like calendars, notes, and reminder apps"
    ],
    'Moderate Impairment': [
        "Work closely with healthcare providers for comprehensive medication management",
        "Implement home safety measures to prevent accidents and ensure security",
        "Establish structured daily routines with clear schedules and reminders",
        "Use comprehensive memory aids including digital reminders and family support",
        "Join specialized support groups for both patients and caregivers",
        "Consider professional in-home care assistance if needed",
        "Focus on maintaining physical health with supervised exercise routines"
    ],
    'No Impairment': [
        "Continue with brain-healthy activities and cognitive exercises",
        "Maintain regular physical exercise routine (150 minutes weekly)",
        "Stay socially engaged and mentally active with new learning",
        "Schedule annual cognitive health check-ups and screenings",
        "Follow a Mediterranean-style diet rich in fruits, vegetables, and fish",
        "Manage cardiovascular risk factors like blood pressure and cholesterol",
        "Get adequate sleep (7-9 hours nightly) for brain health maintenance"
    ],
    'Very Mild Impairment': [
        "Begin structured cognitive training exercises regularly",
        "Monitor symptoms systematically and maintain a health journal",
        "Stay physically active with daily walks or light aerobic exercise",
        "Consider nutritional supplements after consulting healthcare providers",
        "Engage in mentally stimulating activities like reading and puzzles",
        "Participate in social activities and maintain strong social connections",
        "Discuss potential early intervention strategies with your doctor"
    ]
}

# Severity information for visualization
SEVERITY_INFO = {
    'No Impairment': {
        'level': 0,
        'color': '#27ae60',
        'description': 'Normal cognitive function with no signs of impairment'
    },
    'Very Mild Impairment': {
        'level': 1,
        'color': '#f39c12',
        'description': 'Early stage with minimal cognitive changes'
    },
    'Mild Impairment': {
        'level': 2,
        'color': '#e67e22',
        'description': 'Noticeable cognitive decline affecting daily activities'
    },
    'Moderate Impairment': {
        'level': 3,
        'color': '#e74c3c',
        'description': 'Significant cognitive impairment requiring support'
    }
}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Legacy preprocessing function for backward compatibility"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (EfficientNet input size)
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Apply EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_alzheimer_legacy(image):
    """Legacy prediction function"""
    processed_image = preprocess_image(image)
    
    # Force fresh computation by ensuring input is a new numpy array
    processed_image_fresh = np.array(processed_image, copy=True, dtype=np.float32)
    
    # Use model in inference mode - ensure no caching
    with tf.device('/CPU:0'):  # Explicit device to avoid GPU caching issues
        predictions = model(processed_image_fresh, training=False)
    
    # Convert tensor to numpy immediately and ensure it's a fresh array
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy().copy()
    else:
        predictions = np.array(predictions, copy=True)
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return CLASS_NAMES[predicted_class], float(confidence)

# Global variable to track previous prediction for comparison
_previous_prediction_data = None

def predict_alzheimer_enhanced(image):
    """OPTIMIZED: Enhanced prediction using the image processor"""
    global _previous_prediction_data
    
    # Disable verbose logging for speed (can be re-enabled for debugging)
    ENABLE_DEBUG_LOGGING = False  # Set to True for debugging
    
    if ENABLE_DEBUG_LOGGING:
        log_path = os.path.join(os.getcwd(), 'debug.log')
        session_id = 'debug-session'
        run_id = 'run1'
        timestamp_ms = int(time.time() * 1000)
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id':f'log_{timestamp_ms}_entry','timestamp':timestamp_ms,'location':'app.py:147','message':'predict_alzheimer_enhanced entry','data':{'image_mode':image.mode if hasattr(image,'mode') else 'unknown','image_size':list(image.size) if hasattr(image,'size') else 'unknown'},'sessionId':session_id,'runId':run_id,'hypothesisId':'D'})+'\n')
            print(f"[DEBUG] ===== PREDICTION FUNCTION CALLED =====", flush=True)
            print(f"[DEBUG] Image mode: {image.mode if hasattr(image,'mode') else 'unknown'}", flush=True)
            print(f"[DEBUG] Image size: {image.size if hasattr(image,'size') else 'unknown'}", flush=True)
            sys.stdout.flush()
        except Exception as e: 
            print(f"[DEBUG] Log error: {e}")
    else:
        log_path = None
        session_id = None
        run_id = None
        timestamp_ms = None
    try:
        is_valid, message = image_processor.validate_image(image)
        if not is_valid:
            raise ValueError(f"Image validation failed: {message}")

        # OPTIMIZED: Preprocess image (faster without quality enhancement)
        processed_image, enhanced_image = image_processor.preprocess_for_model(image)
        
        # No need to copy again - already fresh from preprocessing
        # processed_image = np.array(processed_image, copy=True, dtype=np.float32)
        
        # #region agent log
        img_hash = hashlib.md5(processed_image.tobytes()).hexdigest()[:16]
        img_stats = {
            'mean': float(np.mean(processed_image)),
            'std': float(np.std(processed_image)),
            'min': float(np.min(processed_image)),
            'max': float(np.max(processed_image)),
            'shape': list(processed_image.shape),
            'hash': img_hash
        }
        
        # Compare with previous input
        comparison = {}
        if _previous_prediction_data and 'input_hash' in _previous_prediction_data:
            prev_hash = _previous_prediction_data['input_hash']
            comparison = {
                'previous_hash': prev_hash,
                'current_hash': img_hash,
                'hashes_match': (prev_hash == img_hash),
                'previous_mean': _previous_prediction_data.get('input_mean', 0),
                'current_mean': img_stats['mean'],
                'means_differ': abs(_previous_prediction_data.get('input_mean', 0) - img_stats['mean']) > 0.001
            }
            print(f"[DEBUG] ⚠️ INPUT COMPARISON:")
            print(f"[DEBUG]   Previous hash: {prev_hash}")
            print(f"[DEBUG]   Current hash:  {img_hash}")
            print(f"[DEBUG]   Hashes match: {comparison['hashes_match']} {'❌ SAME INPUT!' if comparison['hashes_match'] else '✅ Different input'}")
            print(f"[DEBUG]   Previous mean: {comparison['previous_mean']:.6f}")
            print(f"[DEBUG]   Current mean:  {comparison['current_mean']:.6f}")
            print(f"[DEBUG]   Means differ: {comparison['means_differ']} {'❌ SAME VALUES!' if not comparison['means_differ'] else '✅ Different values'}")
        
        log_entry = {'id':f'log_{int(time.time()*1000)}_preprocessed','timestamp':int(time.time()*1000),'location':'app.py:165','message':'processed_image stats','data':{**img_stats, 'comparison': comparison},'sessionId':session_id,'runId':run_id,'hypothesisId':'A'}
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry)+'\n')
            print(f"[DEBUG] Processed image hash: {img_hash}, mean: {img_stats['mean']:.4f}, std: {img_stats['std']:.4f}")
        except Exception as e: 
            print(f"[DEBUG] Log error: {e}")
            print(f"[DEBUG] Processed image hash: {img_hash}, mean: {img_stats['mean']:.4f}")
        # #endregion

        print(f"\n{'='*60}", flush=True)
        print(f"NEW PREDICTION RUN - {time.strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Input image stats: {img_stats}", flush=True)
        print(f"Processed image hash: {img_hash}", flush=True)
        sys.stdout.flush()

        # Use model in inference mode - ensure no caching
        # The processed_image is already a fresh copy, use it directly
        # Use model.__call__ with training=False for proper inference mode
        # This ensures batch normalization and dropout layers behave correctly
        predictions = model(processed_image, training=False)
        
        # Convert tensor to numpy immediately and ensure it's a fresh array
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy().copy()
        else:
            predictions = np.array(predictions, copy=True)
        
        # ===== CRITICAL DEBUG: Verify model output =====
        print(f"\n{'#'*80}", flush=True)
        print(f"🔵 STEP 3: VERIFYING MODEL OUTPUT", flush=True)
        print(f"{'#'*80}", flush=True)
        
        pred_array = predictions[0].tolist()
        pred_hash = hashlib.md5(str(pred_array).encode()).hexdigest()[:16]
        
        print(f"✅ Raw model output: {pred_array}", flush=True)
        print(f"✅ Predictions hash: {pred_hash}", flush=True)
        print(f"✅ Detailed predictions:", flush=True)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"   {class_name}: {pred_array[i]:.8f} ({pred_array[i]*100:.4f}%)", flush=True)
        sys.stdout.flush()
        
        # Compare with previous predictions
        pred_comparison = {}
        if _previous_prediction_data and 'predictions' in _previous_prediction_data:
            prev_pred = _previous_prediction_data['predictions']
            prev_pred_hash = _previous_prediction_data.get('pred_hash', '')
            pred_comparison = {
                'previous_predictions': prev_pred,
                'current_predictions': pred_array,
                'previous_hash': prev_pred_hash,
                'current_hash': pred_hash,
                'predictions_match': (prev_pred_hash == pred_hash),
                'max_diff': max([abs(prev_pred[i] - pred_array[i]) for i in range(len(pred_array))]) if len(prev_pred) == len(pred_array) else -1
            }
            print(f"\n⚠️  PREDICTION COMPARISON:", flush=True)
            print(f"   Previous predictions: {prev_pred}", flush=True)
            print(f"   Current predictions:  {pred_array}", flush=True)
            print(f"   Previous hash: {prev_pred_hash}", flush=True)
            print(f"   Current hash:  {pred_hash}", flush=True)
            print(f"   Predictions match: {pred_comparison['predictions_match']}", flush=True)
            if pred_comparison['predictions_match']:
                print(f"   ❌❌❌ CRITICAL: Model outputting IDENTICAL predictions! ❌❌❌", flush=True)
            else:
                print(f"   ✅ Different predictions (max diff: {pred_comparison['max_diff']:.8f})", flush=True)
            sys.stdout.flush()
        
        log_entry = {'id':f'log_{int(time.time()*1000)}_predictions','timestamp':int(time.time()*1000),'location':'app.py:180','message':'model predictions raw','data':{'predictions':pred_array,'pred_hash':pred_hash,'comparison': pred_comparison},'sessionId':session_id,'runId':run_id,'hypothesisId':'B'}
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry)+'\n')
            print(f"[DEBUG] Model predictions: {pred_array}")
            print(f"[DEBUG] Predictions hash: {pred_hash}")
            print(f"[DEBUG] Detailed predictions:")
            for i, class_name in enumerate(CLASS_NAMES):
                print(f"[DEBUG]   {class_name}: {pred_array[i]:.8f} ({pred_array[i]*100:.4f}%)")
        except Exception as e: 
            print(f"[DEBUG] Log error: {e}")
            print(f"[DEBUG] Model predictions: {pred_array}")
            print(f"[DEBUG] Predictions hash: {pred_hash}")
        # #endregion
        
        # ===== CRITICAL DEBUG: Verify class selection =====
        print(f"\n{'#'*80}", flush=True)
        print(f"🔵 STEP 4: VERIFYING CLASS SELECTION", flush=True)
        print(f"{'#'*80}", flush=True)
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print(f"✅ Argmax selected index: {predicted_class}", flush=True)
        print(f"✅ Selected class: '{CLASS_NAMES[predicted_class]}'", flush=True)
        print(f"✅ Confidence: {confidence:.8f} ({confidence*100:.4f}%)", flush=True)
        sys.stdout.flush()

        # Compare selected class with previous
        class_comparison = {}
        if _previous_prediction_data and 'predicted_class' in _previous_prediction_data:
            prev_class = _previous_prediction_data['predicted_class']
            prev_class_name = _previous_prediction_data.get('predicted_class_name', '')
            class_comparison = {
                'previous_class_idx': prev_class,
                'previous_class_name': prev_class_name,
                'current_class_idx': int(predicted_class),
                'current_class_name': CLASS_NAMES[predicted_class],
                'classes_match': (prev_class == int(predicted_class))
            }
            print(f"\n⚠️  CLASS SELECTION COMPARISON:", flush=True)
            print(f"   Previous: idx={prev_class}, '{prev_class_name}'", flush=True)
            print(f"   Current:  idx={predicted_class}, '{CLASS_NAMES[predicted_class]}'", flush=True)
            print(f"   Classes match: {class_comparison['classes_match']}", flush=True)
            if class_comparison['classes_match']:
                print(f"   ❌❌❌ CRITICAL: Always selecting same class! ❌❌❌", flush=True)
            else:
                print(f"   ✅ Different class selected", flush=True)
            sys.stdout.flush()
        
        log_entry = {'id':f'log_{int(time.time()*1000)}_argmax','timestamp':int(time.time()*1000),'location':'app.py:186','message':'argmax result','data':{'predicted_class_idx':int(predicted_class),'predicted_class_name':CLASS_NAMES[predicted_class],'confidence':float(confidence),'all_predictions':pred_array,'comparison': class_comparison},'sessionId':session_id,'runId':run_id,'hypothesisId':'C'}
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry)+'\n')
            print(f"[DEBUG] Argmax: idx={predicted_class}, class='{CLASS_NAMES[predicted_class]}', conf={confidence:.4f}")
            print(f"[DEBUG] All predictions: {[f'{CLASS_NAMES[i]}: {pred_array[i]:.4f}' for i in range(len(CLASS_NAMES))]}")
        except Exception as e: 
            print(f"[DEBUG] Log error: {e}")
            print(f"[DEBUG] Argmax: idx={predicted_class}, class='{CLASS_NAMES[predicted_class]}', conf={confidence:.4f}")
            print(f"[DEBUG] All predictions: {[f'{CLASS_NAMES[i]}: {pred_array[i]:.4f}' for i in range(len(CLASS_NAMES))]}")
        # #endregion
        
        # Store current prediction data for next comparison
        _previous_prediction_data = {
            'input_hash': img_hash,
            'input_mean': img_stats['mean'],
            'predictions': pred_array,
            'pred_hash': pred_hash,
            'predicted_class': int(predicted_class),
            'predicted_class_name': CLASS_NAMES[predicted_class],
            'timestamp': timestamp_ms
        }

        all_confidences = {
            CLASS_NAMES[i]: float(predictions[0][i]) * 100 
            for i in range(len(CLASS_NAMES))
        }

        print(f"RAW MODEL OUTPUT: {predictions[0]}")
        print(f"Predicted class index: {predicted_class}")
        print(f"Predicted → {CLASS_NAMES[predicted_class]}")
        print(f"Confidence → {confidence * 100:.2f}%")
        print(f"All class probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"  {class_name}: {predictions[0][i]*100:.2f}%")
        print(f"{'='*60}\n")

        # #region agent log
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id':f'log_{int(time.time()*1000)}_exit','timestamp':int(time.time()*1000),'location':'app.py:200','message':'predict_alzheimer_enhanced exit','data':{'final_prediction':CLASS_NAMES[predicted_class],'final_confidence':float(confidence)*100},'sessionId':session_id,'runId':run_id,'hypothesisId':'E'})+'\n')
        except Exception as e: print(f"Log error: {e}")
        # #endregion

        return {
            'prediction': CLASS_NAMES[predicted_class],
            'confidence': float(confidence) * 100,
            'enhanced_image': enhanced_image,
            'all_confidences': all_confidences,
            'validation_message': message
        }

    except Exception as e:
        # #region agent log
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'id':f'log_{int(time.time()*1000)}_error','timestamp':int(time.time()*1000),'location':'app.py:210','message':'prediction error','data':{'error':str(e)},'sessionId':session_id,'runId':run_id,'hypothesisId':'ALL'})+'\n')
        except Exception as e: print(f"Log error: {e}")
        # #endregion
        raise ValueError(f"Enhanced prediction failed: {str(e)}")


@app.route('/')
def index():
    """Home page with system overview"""
    return render_template('index.html')

@app.route('/detection')
def detection():
    """Image upload and detection page"""
    return render_template('detection.html')

@app.route('/realtime')
def realtime():
    """Real-time drawing analysis page"""
    return render_template('realtime.html')

@app.route('/developer')
def developer():
    """Developer documentation page"""
    return render_template('developers.html')

@app.route('/results')
def results_page():
    """Detailed results page"""
    # Get parameters from query string or session
    prediction = request.args.get('prediction', session.get('prediction', 'Unknown'))
    confidence = float(request.args.get('confidence', session.get('confidence', 0)))
    image_data = request.args.get('image_data', session.get('image_data', ''))
    tips = request.args.getlist('tips') or session.get('tips', [])
    
    # Get severity information
    severity_info = SEVERITY_INFO.get(prediction, {})
    
    return render_template('results.html', 
                         prediction=prediction,
                         confidence=confidence,
                         image_data=image_data,
                         tips=tips,
                         severity_info=severity_info,
                         all_severity=SEVERITY_INFO)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and processing"""
    sys.stdout.flush()
    print("\n" + "="*80, flush=True)
    print("🔵 UPLOAD ROUTE CALLED", flush=True)
    print("="*80, flush=True)
    
    if 'file' not in request.files:
        print("❌ No file in request.files")
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    print(f"✅ File received: {file.filename}")
    print(f"✅ File type: {file.content_type}")
    
    if file and allowed_file(file.filename):
        print(f"✅ File is allowed")
        try:
            # Read and process image - ensure we read fresh data
            # Reset stream position if possible
            if hasattr(file.stream, 'seek'):
                file.stream.seek(0)
            # Read image from stream
            image = Image.open(file.stream)
            # Ensure image is loaded into memory
            image.load()
            
            print(f"✅ Image loaded: {image.size}, mode: {image.mode}")
            
            # #region agent log
            log_path = os.path.join(os.getcwd(), 'debug.log')
            try:
                img_bytes = BytesIO()
                image.save(img_bytes, format='PNG')
                img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()[:16]
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id':f'log_{int(time.time()*1000)}_upload','timestamp':int(time.time()*1000),'location':'app.py:260','message':'file uploaded','data':{'filename':file.filename,'image_size':list(image.size),'image_mode':image.mode,'image_hash':img_hash},'sessionId':'debug-session','runId':'run1','hypothesisId':'D'})+'\n')
                print(f"✅ [DEBUG] File uploaded: {file.filename}, hash: {img_hash}")
            except Exception as e: 
                print(f"❌ [DEBUG] Log error: {e}")
                import traceback
                traceback.print_exc()
            # #endregion
            
            # Choose prediction method (enhanced for production)
            use_enhanced = request.form.get('enhanced_processing', 'true').lower() == 'true'
            print(f"🔵 use_enhanced: {use_enhanced}")
            print(f"🔵 image_processor exists: {image_processor is not None}")
            
            if use_enhanced and image_processor:
                print(f"✅ Using ENHANCED prediction method")
                # Use enhanced processing
                print(f"\n{'#'*80}")
                print(f"🔵 [UPLOAD DEBUG] ===== CALLING predict_alzheimer_enhanced =====")
                print(f"🔵 [UPLOAD DEBUG] File: {file.filename}")
                print(f"🔵 [UPLOAD DEBUG] Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'#'*80}")
                
                try:
                    result = predict_alzheimer_enhanced(image)
                    print(f"✅ [UPLOAD DEBUG] predict_alzheimer_enhanced returned successfully")
                    predicted_class = result['prediction']
                    confidence = result['confidence']
                    enhanced_image = result['enhanced_image']
                    all_confidences = result['all_confidences']
                    validation_message = result['validation_message']
                    
                    print(f"✅ [UPLOAD DEBUG] Prediction result: {predicted_class}, Confidence: {confidence}%")
                    print(f"✅ [UPLOAD DEBUG] All confidences: {all_confidences}")
                except Exception as e:
                    print(f"❌❌❌ ERROR in predict_alzheimer_enhanced: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Convert enhanced image to base64
                img_str = image_processor.image_to_base64(enhanced_image)
                
            else:
                print(f"⚠️ Using LEGACY prediction method")
                # Use legacy processing
                predicted_class, confidence_float = predict_alzheimer_legacy(image)
                confidence = confidence_float * 100
                all_confidences = {name: 0 for name in CLASS_NAMES}  # Placeholder
                validation_message = "Processed with standard method"
                print(f"✅ Legacy prediction: {predicted_class}, confidence: {confidence}")
                
                # Convert original image to base64
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_str = f"data:image/jpeg;base64,{img_str}"
            
            # Get medical tips
            tips = MEDICAL_TIPS.get(predicted_class, [])
            
            # Store in session for results page
            session.update({
                'prediction': predicted_class,
                'confidence': confidence,
                'image_data': img_str,
                'tips': tips,
                'all_confidences': all_confidences
            })
            
            response_data = {
                'success': True,
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'image_data': img_str,
                'tips': tips,
                'all_confidences': all_confidences,
                'validation_message': validation_message,
                'severity_info': SEVERITY_INFO.get(predicted_class, {})
            }
            
            print(f"[UPLOAD DEBUG] Returning response with prediction: {predicted_class}")
            print(f"[UPLOAD DEBUG] Response all_confidences: {all_confidences}")
            
            return jsonify(response_data)
            
        except Exception as e:
            error_message = f'Error processing image: {str(e)}'
            print(f"❌ Upload error: {error_message}")
            return jsonify({'success': False, 'error': error_message}), 500
    else:
        return jsonify({
            'success': False, 
            'error': f'File type not allowed. Please use: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

@app.route('/realtime_predict', methods=['POST'])
def realtime_predict():
    """Handle real-time drawing predictions"""
    try:
        # Get image data from canvas
        image_data = request.json.get('image_data', '')
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Use enhanced processing for real-time analysis
        result = predict_alzheimer_enhanced(image)
        predicted_class = result['prediction']
        confidence = result['confidence']
        all_confidences = result['all_confidences']
        
        # Get medical tips
        tips = MEDICAL_TIPS.get(predicted_class, [])
        
        # Store in session
        session.update({
            'prediction': predicted_class,
            'confidence': confidence,
            'tips': tips,
            'all_confidences': all_confidences
        })
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'tips': tips,
            'all_confidences': all_confidences,
            'severity_info': SEVERITY_INFO.get(predicted_class, {})
        })
        
    except Exception as e:
        error_message = f'Error processing drawing: {str(e)}'
        print(f"❌ Real-time prediction error: {error_message}")
        return jsonify({'success': False, 'error': error_message}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            image = Image.open(file.stream)
            result = predict_alzheimer_enhanced(image)
            
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': round(result['confidence'], 2),
                'all_confidences': result['all_confidences'],
                'timestamp': np.datetime64('now').astype(str)
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    model_status = "healthy" if model else "unavailable"
    return jsonify({
        'status': 'operational',
        'model': model_status,
        'version': '1.0.0',
        'timestamp': np.datetime64('now').astype(str)
    })

@app.route('/test_debug')
def test_debug():
    """Test endpoint to verify debugging is working"""
    print("="*80, flush=True)
    print("🔵 TEST DEBUG ENDPOINT CALLED", flush=True)
    print("="*80, flush=True)
    sys.stdout.flush()
    return jsonify({
        'message': 'Debug test successful',
        'debugging_active': True,
        'timestamp': time.time()
    })

@app.route('/batch_upload', methods=['GET', 'POST'])
def batch_upload():
    """Batch upload page for multiple image processing"""
    if request.method == 'POST':
        # Handle batch uploads
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    image = Image.open(file.stream)
                    result = predict_alzheimer_enhanced(image)
                    
                    results.append({
                        'filename': file.filename,
                        'prediction': result['prediction'],
                        'confidence': round(result['confidence'], 2),
                        'status': 'success'
                    })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'prediction': 'Error',
                        'confidence': 0,
                        'status': 'error',
                        'error': str(e)
                    })
        
        return jsonify({'success': True, 'results': results})
    
    return render_template('batch_upload.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Utility routes
@app.route('/clear_session')
def clear_session():
    """Clear user session"""
    session.clear()
    return jsonify({'success': True, 'message': 'Session cleared'})

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model:
        return jsonify({
            'model_type': 'EfficientNetB0',
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'layers': len(model.layers),
            'classes': CLASS_NAMES
        })
    else:
        return jsonify({'error': 'Model not loaded'}), 503

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    # Test logging
    log_path = r'c:\Users\ahsan\Documents\Desktop\Alzehiemer_Detectection_System\.cursor\debug.log'
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'message':'Flask app started','timestamp':int(time.time()*1000)})+'\n')
        print(f"✅ Debug logging initialized: {log_path}")
    except Exception as e:
        print(f"⚠️ Debug logging failed: {e}")
    
    print("🚀 Starting NeuroScan AI Alzheimer Detection System...")
    print("📊 Model Information:")
    if model:
        print(f"   ✅ Model loaded: {model.input_shape} -> {model.output_shape}")
    else:
        print("   ❌ Model not found - please ensure Alzheimer_Detection_model.h5 is in model/ directory")
    
    print("🌐 Starting Flask server...")
    print("📍 Access the application at: http://localhost:5000")
    print("📚 Developer docs at: http://localhost:5000/developer")
    print("🔧 API Health check: http://localhost:5000/api/health")
    
    # Run the application
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )