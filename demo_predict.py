
# demo_predict.py - runs prediction on all images in data/val and prints results
import tensorflow as tf
import numpy as np
import os
from PIL import Image

MODEL = "model.h5"
if not os.path.exists(MODEL):
    print("model.h5 not found. Train first.")
else:
    model = tf.keras.models.load_model(MODEL)
    folder = "data/val"
    
    if not os.path.exists(folder):
        print(f"Validation folder '{folder}' not found.")
    else:
        for cls in ["tumor", "no_tumor"]:
            p = os.path.join(folder, cls)
            if not os.path.exists(p):
                print(f"Class folder '{p}' not found. Skipping...")
                continue
            
            for fname in os.listdir(p):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                
                try:
                    img = Image.open(os.path.join(p, fname)).convert("L").resize((128, 128))
                    arr = np.array(img).astype("float32") / 255.0
                    arr = arr.reshape((1, 128, 128, 1))
                    pred = model.predict(arr, verbose=0)[0][0]
                    label = "tumor" if pred >= 0.5 else "no_tumor"
                    print(f"{fname:30s} {cls:10s} -> {label:10s} (p={pred:.3f})")
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
