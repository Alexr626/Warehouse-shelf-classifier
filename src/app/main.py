import os
import sys
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64

# Add the parent directory to the path to import model classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

# Global variable to store the loaded model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model from checkpoints directory"""
    global model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    
    # Look for .pt files in checkpoints directory
    pt_file = "warehouse_classifier.pth"
    
    if not pt_file:
        raise FileNotFoundError("No .pt model file found in src/checkpoints directory. Try using the training script in the Google Colab file specified in the README, then place the resulting saved model in src/checkpoints")
    
    # Use the first .pt file found
    model_path = os.path.join(checkpoint_dir, pt_file)
    
    try:
        # Add the workspace directory to Python path to resolve 'src' imports
        import sys
        workspace_path = '/workspace'
        if workspace_path not in sys.path:
            sys.path.append(workspace_path)
        
        # Load the model
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to expected input size (adjust as needed based on your model)
    image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor.to(device)

@app.route('/author', methods=['GET'])
def get_author():
    """GET /author endpoint - returns author information"""
    return jsonify({
        "author": "Nokia Assessment API",
        "description": "Warehouse shelf classification service",
        "version": "1.0.0"
    })

@app.route('/classify', methods=['POST'])
def post_classify():
    """POST /classify endpoint - classifies warehouse shelf images as empty or full"""
    global model
    
    if model is None:
        print("Model not loaded, attempting to load...")
        if not load_model():
            return jsonify({"error": "Model could not be loaded"}), 500
    
    try:
        # Check if request contains file or base64 image
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read image from file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
        elif 'image' in request.json:
            # Handle base64 encoded image
            image_data = request.json['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_np)
        
        # Run inference
        with torch.no_grad():
            outputs = model(processed_image)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map prediction to class names
        class_names = ["empty", "full"]  # Adjust based on your model's output
        predicted_label = class_names[predicted_class]
        
        return jsonify({
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                "empty": float(probabilities[0][0]),
                "full": float(probabilities[0][1])
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    print("Starting warehouse shelf classification API...")
    
    # Try to load the model on startup
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {str(e)}")
        print("Place a .pt model file in src/checkpoints directory")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)