from flask import Flask, request, render_template_string
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model('backend/model/brain_tumor_model.h5')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Brain Tumor Detection</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

  * {
    box-sizing: border-box;
  }

  body {
    background: #fff;
    color: #111;
    font-family: 'Inter', sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 1rem;
  }

  .wrapper {
    max-width: 360px;
    width: 100%;
    text-align: center;
  }

  h1 {
    font-weight: 600;
    font-size: 1.8rem;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
  }

  form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  input[type="file"] {
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 0.6rem;
    font-size: 1rem;
  }

  button {
    background: #111;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    border-radius: 6px;
    padding: 0.75rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }

  button:hover,
  button:focus {
    background: #333;
  }

  .result {
    margin-top: 2rem;
    font-weight: 600;
    font-size: 1.25rem;
  }

  .result.positive {
    color: #d32f2f;
  }

  .result.negative {
    color: #2e7d32;
  }

  .result.error {
    color: #f9a825;
  }
</style>
</head>
<body>
  <div class="wrapper">
    <h1>Brain Tumor Detection</h1>
    <form method="post" enctype="multipart/form-data" novalidate>
      <input type="file" name="file" accept="image/*" required />
      <button type="submit">Predict</button>
    </form>

    {% if prediction is not none %}
      {% if error %}
        <div class="result error">{{ prediction }}</div>
      {% else %}
        {% if prediction < 0.5 %}
          <div class="result negative">
            No Tumor Detected<br>
            Confidence: {{ (1 - prediction) * 100 | round(2) }}%
          </div>
        {% else %}
          <div class="result positive">
            Tumor Detected<br>
            Confidence: {{ prediction * 100 | round(2) }}%
          </div>
        {% endif %}
      {% endif %}
    {% endif %}
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = False
    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = "No file uploaded"
            error = True
        else:
            file = request.files['file']
            if file.filename == '':
                prediction = "No file selected"
                error = True
            else:
                try:
                    img = preprocess_image(file.read())
                    preds = model.predict(img)
                    prediction = float(preds[0][0])
                except Exception:
                    prediction = "Error processing image. Please upload a valid image."
                    error = True

    return render_template_string(HTML_PAGE, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
