import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import cv2
import io
import base64
import json
import time
import hashlib

class AlzheimerImageProcessor:
    """
    Advanced image processor for Alzheimer's MRI scan analysis
    Handles preprocessing, augmentation, and quality checks
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.expected_channels = 3  # RGB for EfficientNet
        
    def preprocess_for_model(self, image):
        """
        Main preprocessing pipeline for Alzheimer's detection model
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                # If it's a file path
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                # If it's a numpy array
                image = Image.fromarray(image)
            elif isinstance(image, bytes):
                # If it's bytes data
                image = Image.open(io.BytesIO(image))
            
            # Apply preprocessing steps
            image = self._ensure_rgb(image)
            image = self._resize_image(image)
            image = self._enhance_quality(image)
            
            # Convert to numpy array for model input
            img_array = np.array(image)
            
            # #region agent log
            log_path = r'c:\Users\ahsan\Documents\Desktop\Alzehiemer_Detectection_System\.cursor\debug.log'
            try:
                import os
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                img_before_prep_hash = hashlib.md5(img_array.tobytes()).hexdigest()[:16]
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id':f'log_{int(time.time()*1000)}_before_effnet','timestamp':int(time.time()*1000),'location':'image_processor.py:43','message':'before efficientnet preprocessing','data':{'mean':float(np.mean(img_array)),'std':float(np.std(img_array)),'hash':img_before_prep_hash},'sessionId':'debug-session','runId':'run1','hypothesisId':'A'})+'\n')
            except Exception as e: print(f"Log error: {e}")
            # #endregion
            
            # Apply EfficientNet specific preprocessing
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            
            # #region agent log
            try:
                import os
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                img_after_prep_hash = hashlib.md5(img_array.tobytes()).hexdigest()[:16]
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'id':f'log_{int(time.time()*1000)}_after_effnet','timestamp':int(time.time()*1000),'location':'image_processor.py:50','message':'after efficientnet preprocessing','data':{'mean':float(np.mean(img_array)),'std':float(np.std(img_array)),'hash':img_after_prep_hash},'sessionId':'debug-session','runId':'run1','hypothesisId':'A'})+'\n')
            except Exception as e: print(f"Log error: {e}")
            # #endregion
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, image
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def _ensure_rgb(self, image):
        """Convert image to RGB format if needed"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _resize_image(self, image):
        """OPTIMIZED: Resize image using faster method"""
        # Use BILINEAR instead of LANCZOS for 3x faster resizing
        # LANCZOS is higher quality but much slower
        return image.resize(self.target_size, Image.Resampling.BILINEAR)
    
    def _enhance_quality(self, image):
        """
        OPTIMIZED: Skip quality enhancement for faster processing
        Medical images are typically high quality already
        """
        # Skip enhancement for speed - medical MRI scans are usually high quality
        # If needed, can be re-enabled by uncommenting below
        return image
        
        # Original code (disabled for speed):
        # img_array = np.array(image)
        # quality_score = self._assess_image_quality(img_array)
        # if quality_score < 0.6:
        #     img_array = self._enhance_contrast(img_array)
        #     img_array = self._reduce_noise(img_array)
        #     image = Image.fromarray(img_array)
        # return image
    
    def _assess_image_quality(self, img_array):
        """
        Assess image quality using multiple metrics
        Returns a score between 0 and 1
        """
        try:
            # Convert to grayscale for quality assessment
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate quality metrics
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Contrast (standard deviation)
            contrast = np.std(gray)
            
            # 3. Brightness (mean intensity)
            brightness = np.mean(gray)
            
            # Normalize metrics
            sharpness_norm = min(sharpness / 1000, 1.0)  # Normalize sharpness
            contrast_norm = min(contrast / 80, 1.0)      # Normalize contrast
            brightness_norm = 1 - abs(brightness - 127) / 127  # Ideal around 127
            
            # Combined quality score
            quality_score = (sharpness_norm + contrast_norm + brightness_norm) / 3
            
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5  # Default quality score if assessment fails
    
    def _enhance_contrast(self, img_array):
        """Enhance image contrast using CLAHE"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception:
            # Fallback: simple histogram equalization
            if len(img_array.shape) == 3:
                yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                return enhanced
            else:
                return cv2.equalizeHist(img_array)
    
    def _reduce_noise(self, img_array):
        """Reduce image noise while preserving edges"""
        try:
            return cv2.medianBlur(img_array, 3)
        except Exception:
            return img_array
    
    def validate_image(self, image):
        """
        Validate image for Alzheimer's detection
        Returns (is_valid, message)
        """
        try:
            # Basic checks
            if image is None:
                return False, "No image provided"
            
            # Size checks
            if image.size[0] < 50 or image.size[1] < 50:
                return False, "Image too small (minimum 50x50 pixels required)"
            
            if image.size[0] > 5000 or image.size[1] > 5000:
                return False, "Image too large (maximum 5000x5000 pixels)"
            
            # Aspect ratio check (should be roughly square for MRI scans)
            aspect_ratio = image.size[0] / image.size[1]
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return False, "Unusual aspect ratio detected"
            
            # Quality assessment
            img_array = np.array(image)
            quality_score = self._assess_image_quality(img_array)
            
            if quality_score < 0.3:
                return False, "Image quality too low for accurate analysis"
            
            return True, f"Image validated (Quality score: {quality_score:.2f})"
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    def extract_brain_region(self, image):
        """
        Attempt to extract brain region from MRI scan
        This is a simplified version - in production, you'd use more advanced techniques
        """
        try:
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply threshold to isolate brain region
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours and get the largest one (assuming it's the brain)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create mask
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                
                # Apply mask to original image
                if len(img_array.shape) == 3:
                    result = cv2.bitwise_and(img_array, img_array, mask=mask)
                else:
                    result = cv2.bitwise_and(gray, gray, mask=mask)
                
                # Convert back to PIL
                brain_image = Image.fromarray(result)
                return brain_image
            else:
                return image  # Return original if no contours found
                
        except Exception:
            return image  # Return original if extraction fails
    
    def create_heatmap_overlay(self, original_image, attention_map):
        """
        Create heatmap overlay to show model's areas of focus
        """
        try:
            # Resize attention map to match original image
            attention_resized = cv2.resize(attention_map, original_image.size)
            
            # Normalize attention map
            attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
            
            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * attention_normalized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend with original image
            original_array = np.array(original_image)
            overlay = cv2.addWeighted(original_array, 0.7, heatmap, 0.3, 0)
            
            return Image.fromarray(overlay)
            
        except Exception:
            return original_image  # Return original if heatmap creation fails
    
    def image_to_base64(self, image):
        """Convert PIL image to base64 string for web display"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            raise ValueError(f"Base64 conversion failed: {str(e)}")

# Utility functions
def create_sample_mri_pattern():
    """Create a sample MRI-like pattern for testing"""
    # Create a synthetic MRI-like image
    img = np.random.rand(224, 224, 3) * 50 + 100  # Gray background
    img = img.astype(np.uint8)
    
    # Add some brain-like structures
    center_y, center_x = 112, 112
    y, x = np.ogrid[:224, :224]
    
    # Create ventricle-like dark regions
    mask1 = ((x - center_x)**2 + (y - center_y)**2) < 2500
    mask2 = ((x - center_x + 30)**2 + (y - center_y - 20)**2) < 1600
    mask3 = ((x - center_x - 30)**2 + (y - center_y - 20)**2) < 1600
    
    img[mask1] = 50
    img[mask2] = 50
    img[mask3] = 50
    
    return Image.fromarray(img)

def get_image_statistics(image):
    """Get comprehensive statistics about the image"""
    img_array = np.array(image)
    
    stats = {
        'dimensions': image.size,
        'mode': image.mode,
        'format': getattr(image, 'format', 'Unknown'),
        'mean_intensity': float(np.mean(img_array)),
        'std_intensity': float(np.std(img_array)),
        'min_intensity': float(np.min(img_array)),
        'max_intensity': float(np.max(img_array)),
    }
    
    if len(img_array.shape) == 3:
        # Color image statistics
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            stats[f'{channel}_mean'] = float(np.mean(img_array[:,:,i]))
            stats[f'{channel}_std'] = float(np.std(img_array[:,:,i]))
    
    return stats

# Singleton instance for easy access
image_processor = AlzheimerImageProcessor()