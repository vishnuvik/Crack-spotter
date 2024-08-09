from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Directory for saving uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_path = "C:/Users/vishn/OneDrive/Desktop/Crack detection/Crack_detection.h5"
model = None

try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Preprocess image
def preprocess_image(image_path):
    try:
        print(f"Preprocessing image: {image_path}")
        img = load_img(image_path, target_size=(299, 299))  # Ensure the target size matches your model input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape required by the model
        img_array = img_array / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Predict image
def predict_image(image_path):
    if model is None:
        print("Model not loaded properly.")
        return 'Model not loaded'
    
    img_array = preprocess_image(image_path)
    if img_array is not None:
        try:
            print(f"Predicting image: {image_path}")
            prediction = model.predict(img_array)
            print(f"Prediction raw output: {prediction}")
            return 'Cracked' if prediction[0][0] > 0.5 else 'Not Cracked'
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 'Error during prediction'
    else:
        return 'Error in preprocessing'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in the request.")
            return redirect(url_for('home'))

        file = request.files['file']
        if file.filename == '':
            print("No file selected for uploading.")
            return redirect(url_for('home'))

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                print(f"Saving image to: {file_path}")
                file.save(file_path)
                
                # Verify that the file is a valid image before prediction
                prediction = predict_image(file_path)
                print(f"Prediction result: {prediction}")
                return render_template('predict.html', prediction=prediction, image_path=file.filename)
            except Exception as e:
                print(f"Error uploading file: {e}")
                return redirect(url_for('home'))
        else:
            print("Invalid file type. Only .png, .jpg, and .jpeg are allowed.")
            return redirect(url_for('home'))
    return render_template('predict.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
