# Diagnosica

Diagnosica is a medical image diagnosis assistant focused on brain tumor detection using deep learning.  
This project includes training scripts, model files, and a backend API to serve predictions.

---

## Features

- Train a convolutional neural network on brain MRI images
- Save and load the trained TensorFlow model
- REST API backend for serving predictions (Flask/FastAPI/etc.)
- Git LFS support for managing large model files (`.h5`)

---

## Getting Started

### Prerequisites

- Python 3.10+
- [pyenv](https://github.com/pyenv/pyenv) for managing Python versions (optional but recommended)
- [Git Large File Storage (LFS)](https://git-lfs.github.com/) installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ayaannshaikhh/diagnosica.git
   cd diagnosica
   ```

2. Install and activate the Python environment (using `pyenv` or `venv`):

   ```bash
   pyenv install 3.10.13  # if not already installed
   pyenv virtualenv 3.10.13 tf-venv
   pyenv activate tf-venv
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Git LFS is tracking the large model file:

   ```bash
   git lfs install
   git lfs pull
   ```

---

## Usage

### Train the model

```bash
python train_model.py
```

This script loads training images, trains the model, and saves the output to `backend/model/brain_tumor_model.h5`.

### Run the backend API

```bash
python backend/app.py
```

The backend serves predictions via a REST endpoint.

---

## Notes

- The trained model file can be large; use Git LFS to manage it.
- Add any additional configuration or environment variables your app needs.
- Check the logs for warnings about Keras and PIL dependencies.

---

## License

MIT License © Ayaan Shaikh
