#!/usr/bin/env python3
"""
Working Kannada Character Recognition Application
Fixed version with proper model loading and prediction
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import fitz  # PyMuPDF for PDF processing
from werkzeug.utils import secure_filename
import tempfile
import zipfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json

# TTS imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. TTS will use pyttsx3 fallback.")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not available. TTS functionality will be limited.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variables
model = None
class_names = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uploaded_files = {}  # Store uploaded files in memory

TRAIN_CLASSES = [
  "ಅ","ಅಂ","ಅಃ","ಆ","ಇ","ಈ","ಉ","ಊ","ಋ","ಎ","ಏ","ಐ","ಒ","ಓ","ಔ",
  "ಕ","ಕಂ","ಕಃ","ಕಾ","ಕಿ","ಕೀ","ಕು","ಕೂ","ಕೃ","ಕೆ","ಕೊ","ಕೋ","ಕೇ","ಕೈ","ಕೌ","ಕ್","ಕ್ಕ",
  "ಖ","ಖಂ","ಖಃ","ಖಾ","ಖಿ","ಖೀ","ಖು","ಖೂ","ಖೃ","ಖೆ","ಖೊ","ಖೋ","ಖೇ","ಖೈ","ಖೌ","ಖ್","ಖ್ಖ",
  "ಗ","ಗಂ","ಗಃ","ಗಾ","ಗಿ","ಗೀ","ಗು","ಗೂ","ಗೃ","ಗೆ","ಗೊ","ಗೋ","ಗೇ","ಗೈ","ಗೌ","ಗ್","ಗ್ಗ",
  "ಘ","ಘಂ","ಘಃ","ಘಾ","ಘಿ","ಘೀ","ಘು","ಘೂ","ಘೃ","ಘೆ","ಘೊ","ಘೋ","ಘೇ","ಘೈ","ಘೌ","ಘ್ಘ",
  "ಙ","ಙಂ","ಙಃ","ಙಾ","ಙಿ","ಙೀ","ಙು","ಙೂ","ಙೃ","ಙೆ","ಙೊ","ಙೋ","ಙೇ","ಙೈ","ಙೌ","ಙ್ಙ",
  "ಚ","ಚಂ","ಚಃ","ಚಾ","ಚಿ","ಚೀ","ಚು","ಚೂ","ಚೃ","ಚೆ","ಚೊ","ಚೋ","ಚೇ","ಚೈ","ಚೌ","ಚ್","ಚ್ಚ",
  "ಛ","ಛಂ","ಛಃ","ಛಾ","ಛಿ","ಛೀ","ಛು","ಛೂ","ಛೃ","ಛೆ","ಛೊ","ಛೋ","ಛೇ","ಛೈ","ಛೌ","ಛ್ಛ",
  "ಜ","ಜಂ","ಜಃ","ಜಾ","ಜಿ","ಜೀ","ಜು","ಜೂ","ಜೃ","ಜೆ","ಜೊ","ಜೋ","ಜೇ","ಜೈ","ಜೌ","ಜ್ಜ","ಜ್",
  "ಝ","ಝಂ","ಝಃ","ಝಾ","ಝಿ","ಝೀ","ಝು","ಝೂ","ಝೃ","ಝೆ","ಝೊ","ಝೋ","ಝೇ","ಝೈ","ಝೌ","ಝ್ಝ",
  "ಞ","ಞಂ","ಞಃ","ಞಾ","ಞಿ","ಞೀ","ಞು","ಞೂ","ಞೃ","ಞೆ","ಞೊ","ಞೋ","ಞೇ","ಞೈ","ಞೌ","ಞ್ಞ",
  "ಟ","ಟಂ","ಟಃ","ಟಾ","ಟಿ","ಟೀ","ಟು","ಟೂ","ಟೃ","ಟೆ","ಟೊ","ಟೋ","ಟೇ","ಟೈ","ಟೌ","ಟ್","ಟ್ಟ",
  "ಠ","ಠಂ","ಠಃ","ಠಾ","ಠಿ","ಠೀ","ಠು","ಠೂ","ಠೃ","ಠೆ","ಠೊ","ಠೋ","ಠೇ","ಠೈ","ಠೌ","ಠ್ಠ",
  "ಡ","ಡಂ","ಡಃ","ಡಾ","ಡಿ","ಡೀ","ಡು","ಡೃ","ಡೆ","ಡೊ","ಡೋ","ಡೇ","ಡೈ","ಡೌ","ಡ್ಡ","ಡ್",
  "ಢ","ಢಂ","ಢಃ","ಢಾ","ಢಿ","ಢೀ","ಢು","ಢೂ","ಢೃ","ಢೆ","ಢೊ","ಢೋ","ಢೇ","ಢೈ","ಢೌ","ಢ್ಢ",
  "ಣ","ಣಂ","ಣಃ","ಣಾ","ಣಿ","ಣೀ","ಣು","ಣೂ","ಣೃ","ಣೆ","ಣೊ","ಣೋ","ಣೇ","ಣೈ","ಣೌ","ಣ್","ಣ್ಣ",
  "ತ","ತಂ","ತಃ","ತಾ","ತಿ","ತೀ","ತು","ತೂ","ತೃ","ತೆ","ತೊ","ತೋ","ತೇ","ತೈ","ತೌ","ತ್","ತ್ತ",
  "ಥ","ಥಂ","ಥಃ","ಥಾ","ಥಿ","ಥೀ","ಥು","ಥೂ","ಥೃ","ಥೆ","ಥೊ","ಥೋ","ಥೇ","ಥೈ","ಥೌ","ಥ್ಥ",
  "ದ","ದಂ","ದಃ","ದಾ","ದಿ","ದೀ","ದು","ದೂ","ದೃ","ದೆ","ದೊ","ದೋ","ದೇ","ದೈ","ದೌ","ದ್","ದ್ದ",
  "ಧ","ಧಂ","ಧಃ","ಧಾ","ಧಿ","ಧೀ","ಧು","ಧೂ","ಧೃ","ಧೆ","ಧೊ","ಧೋ","ಧೇ","ಧೈ","ಧೌ","ಧ್ಧ",
  "ನ","ನಂ","ನಃ","ನಾ","ನಿ","ನೀ","ನು","ನೂ","ನೃ","ನೆ","ನೊ","ನೋ","ನೇ","ನೈ","ನೌ","ನ್","ನ್ನ",
  "ಪ","ಪಂ","ಪಃ","ಪಾ","ಪಿ","ಪೀ","ಪು","ಪೂ","ಪೃ","ಪೆ","ಪೊ","ಪೋ","ಪೇ","ಪೈ","ಪೌ","ಪ್","ಪ್ಪ",
  "ಫ","ಫಂ","ಫಃ","ಫಾ","ಫಿ","ಫೀ","ಫು","ಫೂ","ಫೃ","ಫೆ","ಫೊ","ಫೋ","ಫೇ","ಫೈ","ಫೌ","ಫ್ಫ",
  "ಬ","ಬಂ","ಬಃ","ಬಾ","ಬಿ","ಬೀ","ಬು","ಬೂ","ಬೃ","ಬೆ","ಬೊ","ಬೋ","ಬೇ","ಬೈ","ಬೌ","ಬ್","ಬ್ಬ",
  "ಭ","ಭಂ","ಭಃ","ಭಾ","ಭಿ","ಭೀ","ಭು","ಭೂ","ಭೃ","ಭೆ","ಭೊ","ಭೋ","ಭೇ","ಭೈ","ಭೌ","ಭ್","ಭ್ಭ",
  "ಮ","ಮಂ","ಮಃ","ಮಾ","ಮಿ","ಮೀ","ಮು","ಮೂ","ಮೃ","ಮೆ","ಮೊ","ಮೋ","ಮೇ","ಮೈ","ಮೌ","ಮ್","ಮ್ಮ",
  "ಯ","ಯಂ","ಯಃ","ಯಾ","ಯಿ","ಯೀ","ಯು","ಯೂ","ಯೃ","ಯೆ","ಯೊ","ಯೋ","ಯೇ","ಯೈ","ಯೌ","ಯ್","ಯ್ಯ",
  "ರ","ರಂ","ರಃ","ರಾ","ರಿ","ರೀ","ರು","ರೂ","ರೃ","ರೆ","ರೊ","ರೋ","ರೇ","ರೈ","ರೌ","ರ್","ರ್ರ",
  "ಲ","ಲಂ","ಲಃ","ಲಾ","ಲಿ","ಲೀ","ಲು","ಲೂ","ಲೃ","ಲೆ","ಲೊ","ಲೋ","ಲೇ","ಲೈ","ಲೌ","ಲ್","ಲ್ಲ",
  "ಳ","ಳಂ","ಳಃ","ಳಾ","ಳಿ","ಳೀ","ಳು","ಳೂ","ಳೃ","ಳೆ","ಳೊ","ಳೋ","ಳೇ","ಳೈ","ಳೌ","ಳ್ಳ",
  "ವ","ವಂ","ವಃ","ವಾ","ವಿ","ವೀ","ವು","ವೂ","ವೃ","ವೆ","ವೊ","ವೋ","ವೇ","ವೈ","ವೌ","ವ್","ವ್ವ",
  "ಶ","ಶಂ","ಶಃ","ಶಾ","ಶಿ","ಶೀ","ಶು","ಶೂ","ಶೃ","ಶೆ","ಶೊ","ಶೋ","ಶೇ","ಶೈ","ಶೌ","ಶ್","ಶ್ಶ",
  "ಷ","ಷಂ","ಷಃ","ಷಾ","ಷಿ","ಷೀ","ಷು","ಷೂ","ಷೃ","ಷೆ","ಷೊ","ಷೋ","ಷೇ","ಷೈ","ಷೌ","ಷ್","ಷ್ಷ",
  "ಸ","ಸಂ","ಸಃ","ಸಾ","ಸಿ","ಸೀ","ಸು","ಸೂ","ಸೃ","ಸೆ","ಸೊ","ಸೋ","ಸೇ","ಸೈ","ಸೌ","ಸ್","ಸ್ಸ",
  "ಹ","ಹಂ","ಹಃ","ಹಾ","ಹಿ","ಹೀ","ಹು","ಹೂ","ಹೃ","ಹೆ","ಹೊ","ಹೋ","ಹೇ","ಹೈ","ಹೌ","ಹ್ಹ",
  "ೠ","೦","೧","೨","೩","೪","೫","೬","೭","೮","೯"
]

# Canvas / preprocessing defaults
DEFAULT_TARGET_SIZE = 64  # align with existing checkpoint (previous resolution)
CANVAS_BG = 255

# Import the improved model
import sys
sys.path.append('src')
from models.cnn import ImprovedKannadaCNN, KannadaCNN

# Working CNN Architecture (compatible with existing models)
class WorkingKannadaCNN(torch.nn.Module):
    def __init__(self, num_classes=391):
        super().__init__()
        self.features = torch.nn.Sequential(
            # First block
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.1),
            
            # Second block
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.2),
            
            # Third block
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.3),
            
            # Fourth block
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model():
    global model, class_names

    print("Loading improved model...")

    checkpoint_path = "checkpoints/best_improved.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Checkpoint loaded successfully")
    print("Architecture:", checkpoint.get("architecture"))
    print("Number of classes:", checkpoint.get("num_classes"))
    print("Validation accuracy:", checkpoint.get("val_acc"))

    num_classes = checkpoint["num_classes"]

    # ----- CREATE MODEL (MATCH CHECKPOINT EXACTLY) -----
    model = KannadaCNN(
        num_classes=num_classes,
        embedding_dim=256   # MUST match checkpoint
    ).to(device)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # ----- NO CLASS FILE (OLD BEHAVIOR) -----
    # Use numeric labels as strings: "0", "1", "2", ...
    class_names = [str(i) for i in range(num_classes)]

    print(f"Model loaded successfully with {num_classes} classes (index-based labels)")
    return True


def _to_square_with_padding(img_array: np.ndarray, target_size: int = DEFAULT_TARGET_SIZE, margin: int = 6) -> np.ndarray:
    """Crop to content bbox, pad, and resize to a square while preserving aspect."""
    # Invert if background dark? assume white bg
    # Find non-white pixels
    coords = cv2.findNonZero(255 - img_array)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + w + margin, img_array.shape[1])
        y1 = min(y + h + margin, img_array.shape[0])
        cropped = img_array[y0:y1, x0:x1]
    else:
        cropped = img_array

    h, w = cropped.shape
    # Preserve aspect: pad to square before resize
    size = max(h, w)
    padded = np.full((size, size), CANVAS_BG, dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = cropped

    resized = cv2.resize(
        padded,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4
    )
    return resized


def preprocess_image(image: Image.Image, enhanced: bool = True, target_size: int = DEFAULT_TARGET_SIZE):
    """
    Enhanced preprocessing for prediction with fine detail preservation for ottaksharas/words.
    - Gentle denoising
    - BBox crop with padding
    - Aspect-ratio preserving pad -> square resize (higher res)
    """
    if image.mode != 'L':
        image = image.convert('L')

    img_array = np.array(image)

    # Gentle denoising that preserves thin strokes
    if enhanced:
        img_array = cv2.bilateralFilter(img_array, 3, 30, 30)

    # Normalize background to white
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    processed = _to_square_with_padding(img_array, target_size=target_size, margin=10)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])  # match original checkpoint normalization
    ])

    tensor = transform(Image.fromarray(processed)).unsqueeze(0)
    return tensor.to(device)

def predict_character(image_tensor):
    if model is None:
        return None

    try:
        model.eval()
        with torch.no_grad():
            # Model returns (features, logits) tuple
            features, logits = model(image_tensor)
            predicted_idx = logits.argmax(dim=1).item()
            
            # Get confidence score
            probs = F.softmax(logits, dim=1)
            confidence = probs[0][predicted_idx].item()

            predicted_char = TRAIN_CLASSES[predicted_idx]
            return predicted_char, confidence

    except Exception as e:
        print(f"Error in predict_character: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 0.0




def segment_characters(image):
    """Segment multiple characters from an image"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('L'))
        
        # Apply threshold
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        character_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 18 and h >= 18:  # Minimum size filter
                character_boxes.append((x, y, w, h))
        
        # Sort by x-coordinate (left to right)
        character_boxes.sort(key=lambda box: box[0])
        
        # Extract character images
        characters = []
        for x, y, w, h in character_boxes:
            char_img = img_array[y:y+h, x:x+w]
            # Add padding
            char_img = cv2.copyMakeBorder(char_img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            char_pil = Image.fromarray(char_img)
            characters.append(char_pil)
        
        return characters, character_boxes
        
    except Exception as e:
        print(f"Error in segment_characters: {e}")
        return [], []


def segment_with_spaces(image, space_ratio: float = 1.25):
    """
    Segment characters and infer spaces based on horizontal gaps.
    Returns list of (char_image, is_space_flag).
    """
    chars, boxes = segment_characters(image)
    if not chars or not boxes:
        return []

    # Sort by x
    sorted_items = sorted(zip(chars, boxes), key=lambda x: x[1][0])
    widths = [b[2] for _, b in sorted_items]
    median_w = np.median(widths) if widths else 0
    space_thresh = median_w * space_ratio if median_w > 0 else 25

    with_spaces = []
    prev_x2 = None
    for (char_img, (x, y, w, h)) in sorted_items:
        if prev_x2 is not None:
            gap = x - prev_x2
            if gap > space_thresh:
                with_spaces.append((None, True))
        with_spaces.append((char_img, False))
        prev_x2 = x + w
    return with_spaces

def process_pdf(pdf_file):
    """Process PDF file and extract pages as images"""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pages = []
        
        for page_num in range(min(pdf_document.page_count, 40)):  # Limit to 40 pages
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            pages.append({
                'page_num': page_num + 1,
                'image': img,
                'data': img_data
            })
        
        pdf_document.close()
        return pages
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

# Flask Routes
@app.route('/')
def index():
    return render_template('optimized_index.html')

@app.route('/upload_single', methods=['POST'])
def upload_single():
    """Handle single image upload"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        
        # Store in session
        session_id = str(hash(file.filename + str(os.urandom(8))))
        uploaded_files[session_id] = {
            'type': 'single',
            'images': [image],
            'current_index': 0
        }
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image': img_str,
            'total_images': 1,
            'current_index': 0
        })
        
    except Exception as e:
        print(f"Error in upload_single: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    """Handle multiple images upload (up to 40)"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        files = request.files.getlist('images')
        if len(files) > 40:
            return jsonify({'error': 'Maximum 40 images allowed'}), 400
        
        images = []
        for file in files:
            if file.filename != '':
                image = Image.open(io.BytesIO(file.read()))
                images.append(image)
        
        if not images:
            return jsonify({'error': 'No valid images found'}), 400
        
        # Store in session
        session_id = str(hash(str(len(images)) + str(os.urandom(8))))
        uploaded_files[session_id] = {
            'type': 'multiple',
            'images': images,
            'current_index': 0
        }
        
        # Convert first image to base64 for display
        buffered = io.BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image': img_str,
            'total_images': len(images),
            'current_index': 0
        })
        
    except Exception as e:
        print(f"Error in upload_multiple: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload (up to 40 pages)"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF uploaded'}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'error': 'No PDF selected'}), 400
        
        # Process PDF
        pages = process_pdf(file)
        if not pages:
            return jsonify({'error': 'Failed to process PDF'}), 400
        
        # Store in session
        session_id = str(hash(file.filename + str(os.urandom(8))))
        uploaded_files[session_id] = {
            'type': 'pdf',
            'images': [page['image'] for page in pages],
            'current_index': 0,
            'pages': pages
        }
        
        # Convert first page to base64 for display
        buffered = io.BytesIO()
        pages[0]['image'].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image': img_str,
            'total_images': len(pages),
            'current_index': 0
        })
        
    except Exception as e:
        print(f"Error in upload_pdf: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/navigate', methods=['POST'])
def navigate():
    """Navigate between images"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        direction = data.get('direction')  # 'prev' or 'next'
        
        if session_id not in uploaded_files:
            return jsonify({'error': 'Session not found'}), 400
        
        session_data = uploaded_files[session_id]
        current_index = session_data['current_index']
        total_images = len(session_data['images'])
        
        if direction == 'prev':
            new_index = (current_index - 1) % total_images
        else:  # next
            new_index = (current_index + 1) % total_images
        
        session_data['current_index'] = new_index
        
        # Convert current image to base64
        current_image = session_data['images'][new_index]
        buffered = io.BytesIO()
        current_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': img_str,
            'current_index': new_index,
            'total_images': total_images
        })
        
    except Exception as e:
        print(f"Error in navigate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize text in current image.
    Returns final predicted text.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        mode = data.get('mode', 'word')
        
        if session_id not in uploaded_files:
            return jsonify({'error': 'Session not found'}), 400
        
        session_data = uploaded_files[session_id]
        current_image = session_data['images'][session_data['current_index']]
        
        # -------- SEQUENCE / SENTENCE MODE --------
        if mode in ['sequence', 'sentence']:
            char_items = segment_with_spaces(current_image)
            if not char_items:
                return jsonify({'error': 'No characters detected'}), 400
            
            sequence = ""
            total_confidence = 0.0
            char_count = 0
            for char_img, is_space in char_items:
                if is_space:
                    sequence += " "
                    continue
                
                char_tensor = preprocess_image(
                    char_img,
                    enhanced=True,
                    target_size=DEFAULT_TARGET_SIZE
                )
                predicted_char, confidence = predict_character(char_tensor)
                sequence += predicted_char
                total_confidence += confidence
                char_count += 1
            
            final_text = sequence.strip()
            avg_confidence = total_confidence / char_count if char_count > 0 else 0.0
            
            return jsonify({
                'success': True,
                'mode': mode,
                'sequence': final_text,  # Frontend expects this
                'predicted_text': final_text,
                'text': final_text,
                'confidence': avg_confidence
            })
        
        # -------- SINGLE / WORD MODE --------
        else:
            char_tensor = preprocess_image(
                current_image,
                enhanced=True,
                target_size=DEFAULT_TARGET_SIZE
            )
            predicted_char, confidence = predict_character(char_tensor)
            
            return jsonify({
                'success': True,
                'mode': 'word',
                'predicted_character': predicted_char,  # Frontend expects this
                'predicted_text': predicted_char,
                'text': predicted_char,
                'confidence': confidence
            })
        
    except Exception as e:
        print(f"Error in recognize: {e}")
        return jsonify({
            'error': str(e),
            'text': 'Error',
            'confidence': 0
        }), 500



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Direct predict endpoint. Accepts an image file (form-data 'image').
    Returns only the final predicted text.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        tensor = preprocess_image(image, enhanced=True, target_size=DEFAULT_TARGET_SIZE)
        predicted_char, confidence = predict_character(tensor)
        return jsonify({'success': True, 'predicted_text': predicted_char, 'confidence': confidence})
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_text', methods=['POST'])
def download_text():
    """Download recognized text as text file"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        temp_file.write(text)
        temp_file.close()
        
        return send_file(temp_file.name, as_attachment=True, download_name='recognized_text.txt')
        
    except Exception as e:
        print(f"Error in download_text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """Download recognized text as PDF with Unicode support"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Create temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        # Create PDF with Unicode support
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        from reportlab.lib.units import inch
        
        # Register a Unicode-capable font for Kannada support
        font_name = 'Helvetica'  # Default fallback
        
        # List of font paths to try (in order of preference)
        font_paths = [
            # Windows paths - Arial Unicode is the best for Kannada on Windows
            'C:/Windows/Fonts/ARIALUNI.TTF',  # Arial Unicode - BEST for Kannada
            'C:/Windows/Fonts/NotoSansKannada-Regular.ttf',
            'C:/Windows/Fonts/NotoSans-Regular.ttf', 
            'C:/Windows/Fonts/dejavu-sans.ttf',
            'C:/Windows/Fonts/arial.ttf',
            'C:/Windows/Fonts/calibri.ttf',
            # Linux paths
            '/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            # macOS paths
            '/System/Library/Fonts/Helvetica.ttc',
            '/Library/Fonts/Arial.ttf'
        ]
        
        # Try to register a Unicode font
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font_name = 'KannadaFont'
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    print(f"Successfully registered font: {font_path}")
                    break
                else:
                    print(f"Font not found: {font_path}")
            except Exception as e:
                print(f"Error registering font {font_path}: {e}")
                continue
        
        if font_name == 'Helvetica':
            print("Warning: Using fallback font. Kannada characters may not display correctly.")
            print("Consider installing Noto Sans Kannada font for better Unicode support.")
        
        # Create PDF canvas
        c = canvas.Canvas(temp_file.name, pagesize=letter)
        width, height = letter
        
        # Set font and size
        c.setFont(font_name, 12)
        
        # Split text into lines and handle long lines
        lines = text.split('\n')
        y_position = height - 50
        line_height = 20
        
        for line in lines:
            # Handle very long lines by wrapping them
            if len(line) > 80:  # Adjust based on font and page width
                # Simple word wrapping
                words = line.split()
                wrapped_lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + word) <= 80:
                        current_line += word + " "
                    else:
                        if current_line:
                            wrapped_lines.append(current_line.strip())
                        current_line = word + " "
                
                if current_line:
                    wrapped_lines.append(current_line.strip())
                
                lines_to_write = wrapped_lines
            else:
                lines_to_write = [line]
            
            # Write each line
            for line_to_write in lines_to_write:
                if y_position < 50:  # New page
                    c.showPage()
                    c.setFont(font_name, 12)
                    y_position = height - 50
                
                # Use drawString with proper encoding
                try:
                    c.drawString(50, y_position, line_to_write)
                except UnicodeEncodeError:
                    # If Unicode fails, try with UTF-8 encoding
                    try:
                        encoded_line = line_to_write.encode('utf-8').decode('utf-8')
                        c.drawString(50, y_position, encoded_line)
                    except:
                        # Last resort: replace unsupported characters
                        safe_line = line_to_write.encode('ascii', 'replace').decode('ascii')
                        c.drawString(50, y_position, safe_line)
                
                y_position -= line_height
        
        c.save()
        
        return send_file(temp_file.name, as_attachment=True, download_name='recognized_text.pdf')
        
    except Exception as e:
        print(f"Error in download_pdf: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert recognized text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        lang = data.get('lang', 'kn')  # Default to Kannada
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Try gTTS first (requires internet)
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang=lang, slow=False)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts.save(temp_file.name)
                temp_file.close()
                
                return send_file(
                    temp_file.name,
                    mimetype='audio/mpeg',
                    as_attachment=True,
                    download_name='speech.mp3'
                )
            except Exception as e:
                print(f"gTTS error: {e}, trying pyttsx3 fallback")
        
        # Fallback to pyttsx3 (offline, but may not support Kannada well)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                # Save to temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_file.close()
                
                engine.save_to_file(text, temp_file.name)
                engine.runAndWait()
                
                return send_file(
                    temp_file.name,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='speech.wav'
                )
            except Exception as e:
                print(f"pyttsx3 error: {e}")
                return jsonify({'error': f'TTS failed: {str(e)}'}), 500
        
        return jsonify({'error': 'TTS libraries not available'}), 500
        
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(class_names) if class_names else 0,
        'model_type': 'WorkingKannadaCNN',
        'tts_available': GTTS_AVAILABLE or PYTTSX3_AVAILABLE
    })

if __name__ == '__main__':
    if load_model():
        print("Starting WORKING Kannada Recognition Flask app...")
        print("Open your browser and go to: http://localhost:5000")
        print("Features: Single image, Multiple images (40 max), PDF (40 pages max)")
        print("Optimized algorithms with working model!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check your model files.")