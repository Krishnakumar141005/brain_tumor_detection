# Brain Tumor Detection (Demo)

This is a minimal, demo-ready project for **Brain Tumor Detection using a small CNN**.
It contains:
- synthetic dataset under `data/` (train & val folders)
- training script `train.py` (produces `model.h5`)
- Flask app `app.py` to upload image and get prediction
- a tiny pre-trained demo model produced by a short training run

## 📚 Does This Project Train on a Dataset?

**YES!** This project trains on a dataset. Currently, it uses a **synthetic dataset** located in the `data/` folder:
- **Training**: 16 tumor images + 16 no_tumor images
- **Validation**: 4 tumor images + 4 no_tumor images

The model trains when you run `python train.py`. For better accuracy, you can replace this with a real MRI dataset (see below).

## 🚀 Quick Start

### For VS Code Users
See **[VS_CODE_SETUP.md](VS_CODE_SETUP.md)** for detailed step-by-step instructions.

### Command Line Quick Start

1. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python train.py
   ```
   This creates `model.h5` by training on the dataset in `data/` folder.

4. **Run the web app:**
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:5000 in your browser. On first launch, an `auth.db` SQLite database is created automatically for user accounts.

## 📁 Project Structure
```
brain_tumor_project/
├── app.py              # Flask web application with authentication + Grad-CAM workflow
├── train.py            # Model training script (trains on data/)
├── demo_predict.py     # Demo prediction script
├── model.h5            # Trained model (created after training)
├── requirements.txt    # Python dependencies
├── data/               # Dataset folder
│   ├── train/          # Training images (tumor/ and no_tumor/)
│   └── val/            # Validation images (tumor/ and no_tumor/)
├── templates/          # HTML templates (includes auth views + dashboard)
└── static/             # Static files (uploads/)
```

## 🔐 Authentication & Dashboard

- Modern UI with gradient branding, responsive layout (Bootstrap 5) and glassmorphism styling.
- User accounts backed by Flask-Login + SQLAlchemy (`auth.db`).
- Secure password hashing via Werkzeug.
- Flash messages for feedback and a navigation bar with login/logout controls.
- MRI upload workflow is restricted to authenticated users.

### Managing Accounts

1. Visit `/register` to create a profile (name, email, password).
2. Log in via `/login`, upload MRIs from the dashboard, and view Grad-CAM explanations.
3. Use `/logout` to end the session.

> **Tip:** Set a stronger secret key for production  
> `set SECRET_KEY="your-production-secret"` (PowerShell) before starting the app.

## 🎯 Using a Real Dataset

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
   ```bash
   python train.py
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

### Recommended Public Datasets:
- Kaggle: "Brain MRI Images for Brain Tumor Detection"
- Kaggle: "Brain Tumor Classification (MRI)"
- Other medical imaging datasets with brain MRI scans

## 🧪 Testing

Test predictions on validation images:
```bash
python demo_predict.py
```

## 📖 For VS Code Users

See **[VS_CODE_SETUP.md](VS_CODE_SETUP.md)** for comprehensive VS Code setup instructions including:
- Virtual environment setup
- Running and debugging Python files
- Troubleshooting common issues
- Step-by-step walkthrough

