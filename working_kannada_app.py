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
    """Load the improved model"""
    global model, class_names

    print("Loading improved model...")

    # Try to load the best available model
    checkpoint_paths = [
        "checkpoints/best_improved.pt",
        "checkpoints/best.pt"
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"Found checkpoint: {path}")
            break

    if checkpoint_path is None:
        print("No trained model found. Creating a new improved model...")
        model = ImprovedKannadaCNN(num_classes=391).to(device)
        class_names = [f"Class_{i}" for i in range(391)]
        return True

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")
        print(f"   Architecture: {checkpoint.get('architecture', 'Unknown')}")
        print(f"   Number of classes: {checkpoint.get('num_classes', 'Unknown')}")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 'Unknown')}%")

        # ===== CREATE MODEL (MUST MATCH TRAINING) =====
        architecture = checkpoint.get("architecture", "WorkingKannadaCNN")
        num_classes = checkpoint["num_classes"]

        if architecture == "ImprovedKannadaCNN":
            model = ImprovedKannadaCNN(num_classes=num_classes, use_bilstm=True).to(device)
            print("Using ImprovedKannadaCNN with BiLSTM and attention")

        elif architecture == "KannadaCNN":
            model = KannadaCNN(num_classes=num_classes).to(device)
            print("Using KannadaCNN")

        else:
            model = WorkingKannadaCNN(num_classes=num_classes).to(device)
            print("Using WorkingKannadaCNN (fallback)")

        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        # ===== LOAD CLASS NAMES (PRIORITY ORDER) =====

        # 1️⃣ FIRST: try idx_to_class.json (BEST & SAFE)
        idx_path = "checkpoints/idx_to_class.json"
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx_to_class = json.load(f)

            # Ensure correct index order: "0","1","2",...
            class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
            print("Loaded class names from idx_to_class.json")

        # 2️⃣ FALLBACK: checkpoint classes
        else:
            class_names = checkpoint.get(
                "classes",
                [f"Class_{i}" for i in range(num_classes)]
            )
            print("Loaded class names from checkpoint")

        # ===== FINAL SAFETY CHECK =====
        if len(class_names) != num_classes:
            raise ValueError(
                f"Class count mismatch: model expects {num_classes}, "
                f"but got {len(class_names)} labels"
            )

        print(f"Model loaded with {len(class_names)} classes")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = ImprovedKannadaCNN(num_classes=391).to(device)
        class_names = [f"Class_{i}" for i in range(391)]
        return True

def _to_square_with_padding(img_array: np.ndarray, target_size: int = DEFAULT_TARGET_SIZE, margin: int = 6) -> np.ndarray:
    """
    Crop to content bbox, pad, and resize to a square while preserving aspect.
    Enhanced for ottaksharas: preserves thin strokes and compound characters.
    """
    # Normalize to ensure white background (255)
    # Handle both white-on-black and black-on-white inputs
    if np.mean(img_array) < 128:
        # Likely black background, invert
        img_array = 255 - img_array
    
    # Find non-white pixels (content)
    # Use a threshold slightly below 255 to catch near-white pixels
    non_white_mask = img_array < 250
    coords = cv2.findNonZero((255 - img_array).astype(np.uint8))
    
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        # Increased margin for ottaksharas to preserve context
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + w + margin, img_array.shape[1])
        y1 = min(y + h + margin, img_array.shape[0])
        cropped = img_array[y0:y1, x0:x1].copy()
    else:
        cropped = img_array.copy()

    h, w = cropped.shape
    # Preserve aspect: pad to square before resize
    # Use max dimension to ensure no cropping
    size = max(h, w)
    if size == 0:
        size = target_size
    
    padded = np.full((size, size), CANVAS_BG, dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = cropped

    # Use high-quality interpolation to preserve thin strokes
    resized = cv2.resize(
        padded,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4  # Best quality for preserving details
    )
    return resized


def preprocess_image(image: Image.Image, enhanced: bool = True, target_size: int = DEFAULT_TARGET_SIZE):
    """
    Enhanced preprocessing for prediction with fine detail preservation for ottaksharas/words.
    - Gentle denoising (preserves thin strokes)
    - NO aggressive thresholding (preserves virama marks and compound characters)
    - BBox crop with padding
    - Aspect-ratio preserving pad -> square resize
    - Normalization matches training exactly
    """
    if image.mode != 'L':
        image = image.convert('L')

    img_array = np.array(image).astype(np.uint8)
    
    # Normalize background to white (255) - handle both formats
    if np.mean(img_array) < 128:
        # Likely inverted (black bg), invert to white bg
        img_array = 255 - img_array
    
    # Gentle denoising that preserves thin strokes and edges
    # Critical for ottaksharas: avoid aggressive filtering
    if enhanced:
        # Very gentle bilateral filter - preserves fine details
        img_array = cv2.bilateralFilter(img_array, d=3, sigmaColor=30, sigmaSpace=30)
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # NO aggressive thresholding - preserves compound characters and virama marks
    # The model was trained on grayscale images, not binary
    
    # Enhanced padding for ottaksharas (compound characters need more context)
    processed = _to_square_with_padding(img_array, target_size=target_size, margin=12)

    # Transform matching training pipeline exactly
    # Training uses: ToTensor -> Normalize(mean=[0.485], std=[0.229])
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229])  # Match training normalization exactly
    ])

    tensor = transform(Image.fromarray(processed)).unsqueeze(0)
    return tensor.to(device)

def predict_character_with_top5(image_tensor, print_top5=False):
    """
    Predict a single character/word with top-5 predictions and confidence scores.
    Returns tuple: (predicted_text, top5_list)
    top5_list: [(class_name, confidence), ...]
    """
    if model is None:
        return None, []
    
    try:
        with torch.no_grad():
            # Handle models that return (features, logits) vs just logits
            # ImprovedKannadaCNN returns (z, logits), KannadaCNN returns logits
            outputs = model(image_tensor)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                # Model returns (features, logits) - use logits
                _, outputs = outputs
            
            # Get probabilities using softmax
            probs = F.softmax(outputs, dim=1)
            
            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, k=min(5, len(class_names)), dim=1)
            
            top5_list = []
            for i in range(top5_indices.shape[1]):
                idx = top5_indices[0, i].item()
                conf = top5_probs[0, i].item()
                class_name = class_names[idx] if idx < len(class_names) else "Unknown"
                top5_list.append((class_name, conf))
            
            # Best prediction
            predicted_idx = top5_indices[0, 0].item()
            predicted_character = class_names[predicted_idx] if predicted_idx < len(class_names) else "Unknown"
            
            # Top-5 predictions available but not printed (kept for internal use)
            
            return predicted_character, top5_list
            
    except Exception as e:
        print(f"Error in predict_character_with_top5: {e}")
        import traceback
        traceback.print_exc()
        return "Error", []

def predict_character(image_tensor):
    """Predict a single character/word. Returns only final text (backward compatibility)."""
    predicted_text, _ = predict_character_with_top5(image_tensor, print_top5=False)
    return predicted_text

def segment_characters(image):
    """
    Segment multiple characters from an image.
    Works with both Canvas images (white bg, black strokes) and uploaded images.
    Returns: (list of PIL Images, list of bounding boxes)
    """
    try:
        # Convert PIL to OpenCV format (grayscale)
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        else:
            img_array = np.array(image)
        
        # Ensure we have a proper grayscale image
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Canvas images: white background (255), black strokes (0)
        # Normalize to ensure consistent format
        mean_val = np.mean(img_array)
        if mean_val < 128:
            # Likely inverted (black background) - invert to white bg
            img_array = 255 - img_array
        
        # Apply threshold to get binary image (black strokes on white background)
        # Use OTSU for automatic threshold selection
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (external only to avoid nested contours)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in image")
            return [], []
        
        # Filter and sort contours
        character_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Minimum size filter - reduced for Canvas which may have smaller strokes
            # But ensure we don't capture noise
            if w >= 15 and h >= 15:
                character_boxes.append((x, y, w, h))
        
        if not character_boxes:
            print(f"No valid character boxes found (filtered {len(contours)} contours)")
            return [], []
        
        # Sort by x-coordinate (left to right) for proper reading order
        character_boxes.sort(key=lambda box: box[0])
        
        print(f"Segmented {len(character_boxes)} characters")
        
        # Extract character images from original grayscale image
        characters = []
        for x, y, w, h in character_boxes:
            # Extract character region
            char_img = img_array[y:y+h, x:x+w].copy()
            
            # Ensure white background for extracted character
            if np.mean(char_img) < 128:
                char_img = 255 - char_img
            
            # Add padding around character for better recognition
            char_img = cv2.copyMakeBorder(char_img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            
            # Convert back to PIL Image
            char_pil = Image.fromarray(char_img)
            characters.append(char_pil)
        
        return characters, character_boxes
        
    except Exception as e:
        print(f"Error in segment_characters: {e}")
        import traceback
        traceback.print_exc()
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
    Returns final predicted text plus confidence metadata.
    
    mode:
        - 'word' / 'single': treat full image as a single class (ottaksharas supported)
        - 'sequence' or 'sentence': segment into characters and infer spaces
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        mode = data.get('mode', 'word')  # default to word/full-image
        
        if session_id not in uploaded_files:
            return jsonify({'error': 'Session not found'}), 400
        
        session_data = uploaded_files[session_id]
        current_image = session_data['images'][session_data['current_index']]
        
        # Default confidence metadata
        confidence = None
        is_uncertain = False
        uncertainty_message = ""
        
        # Confidence threshold for uncertainty messaging
        UNCERTAIN_THRESHOLD = 0.6
        
        if mode in ['sequence', 'sentence']:
            # Segment image into characters (works for both Canvas and Upload)
            # Canvas images are stored as PIL Images, same as uploads
            char_items = segment_with_spaces(current_image)
            
            # STRICT: If segmentation fails, return error (no silent fallback)
            if not char_items:
                print(f"Error: Segmentation failed for {mode} mode - no characters detected")
                return jsonify({
                    'success': False,
                    'error': 'Canvas sequence requires multiple separated characters',
                    'mode': mode
                }), 400
            
            # Verify we have at least 2 characters for sequence/sentence modes
            # (single character should use Word/Single mode)
            actual_char_count = sum(1 for _, is_space in char_items if not is_space)
            if actual_char_count < 2:
                print(f"Error: Only {actual_char_count} character(s) detected - use Word/Single mode for single characters")
                return jsonify({
                    'success': False,
                    'error': 'Canvas sequence requires multiple separated characters',
                    'mode': mode
                }), 400
            
            sequence = ""
            char_confidences = []
            
            # Process each segmented character INDIVIDUALLY
            # Each character is preprocessed and predicted separately
            for idx, (char_img, is_space) in enumerate(char_items):
                if is_space:
                    # Add space ONLY for sentence mode when gap is detected
                    if mode == 'sentence':
                        sequence += " "
                    continue
                
                # Preprocess EACH character individually
                if char_img is None:
                    continue  # Skip None entries (shouldn't happen, but safety check)
                
                char_tensor = preprocess_image(char_img, enhanced=True, target_size=DEFAULT_TARGET_SIZE)
                # Prediction is based ONLY on model output tensor
                predicted_char, top5_list = predict_character_with_top5(char_tensor, print_top5=False)
                
                # Append predicted character to sequence
                sequence += predicted_char
                
                # Track confidence for each character
                if top5_list:
                    char_confidences.append(top5_list[0][1])
            
            # Calculate overall confidence from individual character confidences
            if char_confidences:
                confidence = float(np.mean(char_confidences))
                if confidence < UNCERTAIN_THRESHOLD:
                    is_uncertain = True
                    uncertainty_message = "Uncertain prediction – ottakshara may require clearer strokes"
            else:
                confidence = None
            
            return jsonify({
                'success': True,
                'mode': mode,
                'predicted_text': sequence.strip(),
                'confidence': confidence,
                'is_uncertain': is_uncertain,
                'uncertainty_message': uncertainty_message
            })
        else:
            # Word/single mode: treat entire image as single class (no segmentation)
            # This is critical for ottaksharas which are compound characters
            char_tensor = preprocess_image(current_image, enhanced=True, target_size=DEFAULT_TARGET_SIZE)
            
            # Prediction is based only on model output tensor
            predicted_char, top5_list = predict_character_with_top5(char_tensor, print_top5=False)
            
            if top5_list:
                confidence = float(top5_list[0][1])
                if confidence < UNCERTAIN_THRESHOLD:
                    is_uncertain = True
                    uncertainty_message = "Uncertain prediction – ottakshara may require clearer strokes"
            
            return jsonify({
                'success': True,
                'mode': 'word',
                'predicted_text': predicted_char,
                'confidence': confidence,
                'is_uncertain': is_uncertain,
                'uncertainty_message': uncertainty_message
            })
        
    except Exception as e:
        print(f"Error in recognize: {e}")
        return jsonify({'error': str(e)}), 500


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
        predicted_char = predict_character(tensor)
        return jsonify({'success': True, 'predicted_text': predicted_char})
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