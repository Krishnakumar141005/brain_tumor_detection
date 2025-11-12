# Troubleshooting Guide

## NumPy Import Error

### Error Message:
```
ImportError: Error importing numpy: you should not try to import numpy from its source directory
ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'
```

### Solution: Reinstall NumPy with Compatible Version

This error occurs when NumPy 2.x is incompatible with TensorFlow. TensorFlow 2.10-2.20 works best with NumPy 1.24.x.

**Quick Fix:**

1. **Uninstall NumPy:**
   ```powershell
   pip uninstall numpy -y
   ```

2. **Install compatible NumPy version:**
   ```powershell
   pip install "numpy<2.0"
   ```

3. **Verify installation:**
   ```powershell
   python -c "import numpy; print(numpy.__version__)"
   ```

4. **Test TensorFlow:**
   ```powershell
   python -c "import tensorflow as tf; print('Success!')"
   ```

**Alternative: Pin NumPy version in requirements.txt**

Update `requirements.txt` to include:
```
numpy<2.0
```

Then reinstall:
```powershell
pip install -r requirements.txt --force-reinstall
```

---

## ML_Dtypes Module Error

### Error Message:
```
ModuleNotFoundError: No module named 'ml_dtypes._ml_dtypes_ext'
```

### Solution: Reinstall ML_Dtypes or Use Stable TensorFlow Version

This error occurs when `ml_dtypes` (a TensorFlow dependency) is corrupted or incompatible.

**Quick Fix:**

1. **Reinstall ml_dtypes:**
   ```powershell
   pip uninstall ml_dtypes -y
   pip install ml_dtypes --force-reinstall
   ```

2. **Or reinstall TensorFlow with a stable version:**
   ```powershell
   pip uninstall tensorflow-cpu -y
   pip install tensorflow-cpu==2.15.0
   ```

3. **Or reinstall all requirements:**
   ```powershell
   pip install -r requirements.txt --force-reinstall
   ```

**Recommended: Use TensorFlow 2.15.0**

TensorFlow 2.15.0 is more stable and compatible. The `requirements.txt` has been updated to use this version.

---

## Keras Optree Error

### Error Message:
```
ImportError: To use Keras, you need to have `optree` installed. Install it via `pip install optree`
```

### Solution: Install Optree

Keras 3.x requires `optree` package. This has been added to `requirements.txt`.

**Quick Fix:**

1. **Install optree:**
   ```powershell
   pip install optree
   ```

2. **Or reinstall all requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **If you have tensorflow-intel installed, uninstall it first:**
   ```powershell
   pip uninstall tensorflow-intel -y
   pip install -r requirements.txt
   ```

**Note:** If you see dependency conflicts with `tensorflow-intel`, uninstall it as it conflicts with `tensorflow` or `tensorflow-cpu`.

---

## TensorFlow DLL Load Error on Windows

### Error Message:
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal: The specified module could not be found.
```

### Solution 1: Use TensorFlow CPU Version (Recommended for Windows)

The `requirements.txt` has been updated to use `tensorflow-cpu` which is more compatible with Windows.

**Steps:**

1. **Uninstall current TensorFlow:**
   ```powershell
   pip uninstall tensorflow tensorflow-cpu -y
   ```

2. **Reinstall with CPU version:**
   ```powershell
   pip install tensorflow-cpu>=2.10
   ```

   Or reinstall all requirements:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```powershell
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

### Solution 2: Install Visual C++ Redistributables

TensorFlow requires Visual C++ Redistributables on Windows.

1. **Download and install:**
   - Go to: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search for "Microsoft Visual C++ Redistributable 2015-2022"
   - Install the x64 version

2. **Restart your terminal/VS Code**

3. **Try again:**
   ```powershell
   python train.py
   ```

### Solution 3: Check Python Version

TensorFlow 2.10+ requires Python 3.8-3.11.

**Check your Python version:**
```powershell
python --version
```

**If you have Python 3.12+, downgrade TensorFlow:**
```powershell
pip install tensorflow-cpu==2.13.0
```

### Solution 4: Reinstall TensorFlow Completely

If the above solutions don't work:

1. **Uninstall TensorFlow:**
   ```powershell
   pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y
   ```

2. **Clear pip cache:**
   ```powershell
   pip cache purge
   ```

3. **Reinstall:**
   ```powershell
   pip install tensorflow-cpu>=2.10
   ```

### Solution 5: Use Conda (Alternative)

If pip installation continues to fail, use Conda:

1. **Install Miniconda/Anaconda**

2. **Create environment:**
   ```powershell
   conda create -n brain_tumor python=3.10
   conda activate brain_tumor
   ```

3. **Install TensorFlow:**
   ```powershell
   conda install tensorflow
   ```

4. **Install other dependencies:**
   ```powershell
   pip install flask pillow matplotlib numpy opencv-python
   ```

### Solution 6: Check System Requirements

Ensure your system meets requirements:
- **Windows 10/11** (64-bit)
- **Python 3.8-3.11** (64-bit)
- **Visual C++ Redistributables 2015-2022**

### Quick Fix Script

Run this in PowerShell to fix most issues:

```powershell
# Uninstall old TensorFlow
pip uninstall tensorflow tensorflow-cpu tensorflow-gpu -y

# Clear cache
pip cache purge

# Install CPU version
pip install tensorflow-cpu>=2.10

# Verify
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## Other Common Issues

### Issue: "No module named 'flask'"
**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue: "model.h5 not found"
**Solution:**
```powershell
python train.py
```

### Issue: Port 5000 already in use
**Solution:**
Change port in `app.py`:
```python
app.run(debug=True, port=5001)
```

### Issue: "Execution policy error" (PowerShell)
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Training is slow
**Solution:**
- This is normal for CPU-only TensorFlow
- Consider using a smaller dataset for testing
- Reduce epochs in `train.py` for faster testing

---

## Still Having Issues?

1. **Check TensorFlow installation:**
   ```powershell
   python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
   ```

2. **Check Python version:**
   ```powershell
   python --version
   ```

3. **Check pip version:**
   ```powershell
   pip --version
   ```

4. **Update pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

5. **Recreate virtual environment:**
   ```powershell
   # Deactivate current venv
   deactivate
   
   # Remove old venv
   Remove-Item -Recurse -Force venv
   
   # Create new venv
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Install requirements
   pip install -r requirements.txt
   ```

---

## Need More Help?

- TensorFlow Installation Guide: https://www.tensorflow.org/install
- TensorFlow Windows Issues: https://www.tensorflow.org/install/errors
- Stack Overflow: Search for "tensorflow dll load failed windows"

