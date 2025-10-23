from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Dense
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Set environment variable to avoid oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_exact_model():
    """Recreate the exact model architecture from your training"""
    model = Sequential()
    
    # Exact architecture from your training code
    model.add(Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=(150,150,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(Conv2D(128, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile with the same settings as your training
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Try to load model directly first, if fails recreate architecture
try:
    print("🔄 Attempting to load model directly...")
    model = tf.keras.models.load_model('my_model.h5')
    print("✅ Model loaded directly!")
except Exception as e:
    print(f"❌ Direct load failed: {e}")
    print("🔄 Recreating model architecture and loading weights...")
    model = create_exact_model()
    model.load_weights('my_model.h5')
    print("✅ Model loaded with recreated architecture!")

# Model configuration
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

def preprocess_image(image):
    """Preprocess image to match training data format"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    
    # Resize to match training size
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize pixel values (same as your training: /255)
    image = image / 255.0
    
    # Reshape for model input
    image = image.reshape(img_size, img_size, 1)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        if 'base64,' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
            
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image then to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Add batch dimension for prediction
        input_image = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction_prob = model.predict(input_image, verbose=0)[0][0]
        
        # Convert probability to class (0 = PNEUMONIA, 1 = NORMAL)
        predicted_class = 1 if prediction_prob >= 0.5 else 0
        confidence = float(prediction_prob if predicted_class == 1 else 1 - prediction_prob)
        
        # Get class label
        predicted_label = labels[predicted_class]
        
        # Calculate probabilities for both classes
        class_probabilities = {
            'PNEUMONIA': float(1 - prediction_prob),
            'NORMAL': float(prediction_prob)
        }
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'prediction_probability': float(prediction_prob),
            'class_probabilities': class_probabilities,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to check model information"""
    try:
        return jsonify({
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'layers': len(model.layers),
            'model_type': 'Pneumonia Detection CNN',
            'classes': labels,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True,
        'message': 'Pneumonia detection model is ready'
    })

if __name__ == '__main__':
    print("🚀 Starting Pneumonia Detection API...")
    print(f"📊 Model input shape: {model.input_shape}")
    print(f"🎯 Detection classes: {labels}")
    print("🔌 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)