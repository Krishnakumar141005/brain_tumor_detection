from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import cv2, os
from PIL import Image
import matplotlib.pyplot as plt

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------- Load Model -------------------
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please run train.py first to create the model.")

model = load_model(MODEL_PATH)

# Make sure model is built (important for Grad-CAM)
dummy_input = np.zeros((1, 128, 128, 1), dtype=np.float32)
_ = model.predict(dummy_input, verbose=0)

# ------------------- Grad-CAM Heatmap Function -------------------
def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """Generate Grad-CAM heatmap for the model prediction."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in the model.")
    
    # Run model once to make sure outputs are defined
    _ = model(img_array, training=False)

    # Create a model that maps the input to the activations of the last conv layer and the output predictions
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model. Available layers: {[l.name for l in model.layers]}")
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        # For binary classification with sigmoid, use the prediction value directly
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    # Compute gradients of the top predicted class with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight the channels by corresponding gradients
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize between 0 & 1
    heatmap = np.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.numpy()

# ------------------- Overlay Heatmap -------------------
def overlay_heatmap(heatmap, image_path, alpha=0.4):
    """Overlay heatmap on the original image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    # Use matplotlib's colormap (updated API)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.cvtColor((jet_heatmap * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
    heatmap_path = "static/heatmap_result.jpg"
    cv2.imwrite(heatmap_path, superimposed_img)
    return heatmap_path

# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="No file uploaded!")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result="No selected file!")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # -------- Preprocess the Image --------
    IMG_SIZE = (128, 128)
    img = Image.open(filepath).convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1,128,128,1)

    # -------- Model Prediction --------
    preds = model.predict(img_array, verbose=0)
    # Model uses sigmoid output (single value), not softmax
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
        label = "Tumor" if prob >= 0.5 else "No Tumor"
        # For display, show confidence as probability of the predicted class
        confidence = prob if prob >= 0.5 else (1 - prob)
    else:
        prob = float(np.max(preds))
        label = "Tumor" if np.argmax(preds) == 1 else "No Tumor"
        confidence = prob

    # -------- Grad-CAM Visualization --------
    try:
        heatmap = get_gradcam_heatmap(img_array, model)
        heatmap_image = overlay_heatmap(heatmap, filepath)
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        heatmap_image = None

    return render_template(
        "index.html",
        result=label,
        prob=f"{confidence*100:.2f}%",
        original_image=filepath,
        heatmap_image=heatmap_image
    )

# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(debug=True)
