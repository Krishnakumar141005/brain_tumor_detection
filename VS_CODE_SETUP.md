# How to Run This Project in VS Code

## 📋 Prerequisites
- Python 3.8 or higher
- VS Code installed
- VS Code Python extension (recommended)

---

## 🚀 Step-by-Step Setup

### Step 1: Open Project in VS Code
1. Open VS Code
2. Click `File` → `Open Folder`
3. Navigate to and select the `brain_tumor_project` folder
4. Click `Select Folder`

### Step 2: Create Virtual Environment
1. Open the terminal in VS Code:
   - Press `` Ctrl + ` `` (backtick) or
   - Go to `Terminal` → `New Terminal`
   - Or use `View` → `Terminal`

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate
     ```
   - **If you get an execution policy error in PowerShell, run:**
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```

   You should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies
In the activated terminal, run:
```powershell
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- TensorFlow (deep learning)
- Pillow (image processing)
- Matplotlib (visualization)
- NumPy (numerical computing)
- OpenCV (computer vision)

### Step 4: Train the Model
**Yes, this project DOES train on a dataset!**

The project currently uses a **synthetic dataset** located in the `data/` folder:
- **Training data**: `data/train/tumor/` and `data/train/no_tumor/` (16 images each)
- **Validation data**: `data/val/tumor/` and `data/val/no_tumor/` (4 images each)

To train the model:
```powershell
python train.py
```

This will:
- Load images from the `data/` folder
- Train a CNN model for 3 epochs
- Save the trained model as `model.h5`

**Note:** For better accuracy, you can:
- Replace the synthetic dataset with a real MRI dataset
- Increase the number of epochs in `train.py` (currently set to 3)
- Use more training data

### Step 5: Run the Flask Web App
1. Make sure the model is trained (you should have `model.h5` in the project root)

2. Run the Flask app:
   ```powershell
   python app.py
   ```

3. You should see output like:
   ```
   * Running on http://127.0.0.1:5000
   * Debug mode: on
   ```

4. Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

5. Upload a brain MRI image to get predictions!

---

## 🎯 Using VS Code Features

### Running Python Files
1. **Method 1 - Terminal:**
   - Open terminal (`` Ctrl + ` ``)
   - Type: `python filename.py`
   - Press Enter

2. **Method 2 - Run Button:**
   - Open a Python file (e.g., `app.py`)
   - Click the "▶ Run" button in the top-right corner
   - Or press `F5` to debug

3. **Method 3 - Right-click:**
   - Right-click on a Python file in the Explorer
   - Select "Run Python File in Terminal"

### Debugging
1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Select "Python File" as the debug configuration

### Python Interpreter Selection
1. Press `Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `venv\Scripts\python.exe`

---

## 📁 Project Structure
```
brain_tumor_project/
├── app.py              # Flask web application
├── train.py            # Model training script
├── demo_predict.py     # Demo prediction script
├── model.h5            # Trained model (created after training)
├── requirements.txt    # Python dependencies
├── data/               # Dataset folder
│   ├── train/          # Training images
│   │   ├── tumor/      # Tumor images
│   │   └── no_tumor/   # Normal images
│   └── val/            # Validation images
│       ├── tumor/
│       └── no_tumor/
├── templates/          # HTML templates
│   └── index.html
├── static/             # Static files
│   └── uploads/        # User uploaded images
└── venv/               # Virtual environment (created)
```

---

## 🔧 Troubleshooting

### Issue: "model.h5 not found"
**Solution:** Run `python train.py` first to create the model.

### Issue: "No module named 'flask'"
**Solution:** Make sure the virtual environment is activated and run `pip install -r requirements.txt`

### Issue: "Execution policy error" (PowerShell)
**Solution:** Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Port 5000 already in use
**Solution:** 
- Close other applications using port 5000, or
- Change the port in `app.py`:
  ```python
  app.run(debug=True, port=5001)
  ```

### Issue: Training takes too long
**Solution:** The current dataset is small (synthetic). For real datasets, training will take longer. You can reduce epochs in `train.py` for testing.

---

## 📊 Testing the Model

### Test on validation images:
```powershell
python demo_predict.py
```

This will run predictions on all images in `data/val/` and print the results.

---

## 🎓 Using a Real Dataset

To use a real brain MRI dataset:

1. **Download a dataset** (e.g., from Kaggle: "Brain MRI Images for Brain Tumor Detection")

2. **Organize your data** in this structure:
   ```
   data/
   ├── train/
   │   ├── tumor/       # Put tumor images here
   │   └── no_tumor/    # Put normal images here
   └── val/
       ├── tumor/
       └── no_tumor/
   ```

3. **Train the model:**
   ```powershell
   python train.py
   ```

4. **Run the app:**
   ```powershell
   python app.py
   ```

---

## ✅ Quick Checklist

- [ ] Project opened in VS Code
- [ ] Virtual environment created (`venv`)
- [ ] Virtual environment activated (see `(venv)` in terminal)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model trained (`python train.py`)
- [ ] Flask app running (`python app.py`)
- [ ] Browser opened to `http://127.0.0.1:5000`

---

## 🎉 You're All Set!

Your brain tumor detection project is now ready to use in VS Code!

