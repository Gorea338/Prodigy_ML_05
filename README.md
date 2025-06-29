
# PRODIGY\_ML\_05

Develop a **Food Image Recognition Model** that accurately identifies and classifies various types of food items from image data, supporting dietary tracking, calorie estimation, and smart menu suggestions.

# Food Image Recognition Model Results

## Model Information

* **Generated**: 20250629\_164500
* **Architecture**: Deep CNN with Batch Normalization and Dropout
* **Input Shape**: \[224, 224, 3]
* **Number of Classes**: 20
* **Total Parameters**: 4,812,375

## Performance Summary

* **Test Accuracy**: 0.8923
* **Best Validation Accuracy**: 0.9861
* **Training Epochs**: 25

## Food Classes

* 0: Apple
* 1: Banana
* 2: Burger
* 3: Cake
* 4: Dosa
* 5: Fried Rice
* 6: Ice Cream
* 7: Idli
* 8: Noodles
* 9: Omelette
* 10: Orange
* 11: Paneer Curry
* 12: Pasta
* 13: Pizza
* 14: Poha
* 15: Salad
* 16: Sandwich
* 17: Samosa
* 18: Tea
* 19: Upma

## Output Files Description

### Model Files

* `food_image_recognition_model.h5` – Keras model in HDF5 format
* `food_image_recognition_model.keras` – Keras model in native format
* `food_model_savedmodel/` – TensorFlow SavedModel format (recommended for deployment)
* `food_label_encoder.pkl` – Label encoder mapping class names to indices

### Performance Analysis

* `model_results_summary_20250629_164500.json` – Overall performance metrics
* `classification_report_20250629_164500.json` – Per-class precision, recall, F1-score
* `per_class_performance_20250629_164500.csv` – Class-wise accuracy and confidence analysis
* `confusion_matrix_20250629_164500.csv` – Confusion matrix (CSV)

### Training Data

* `training_history_20250629_164500.csv` – Epoch-wise accuracy and loss (train/val)
* `training_plots_20250629_164500.png` – Accuracy and loss curves over epochs

### Predictions

* `test_predictions_20250629_164500.csv` – Test set predictions with:

  * True vs predicted labels
  * Confidence scores
  * Class probabilities
  * Correct/incorrect indicators

### Visualizations

* `confusion_matrix_20250629_164500.png` – Confusion matrix heatmap
* `training_plots_20250629_164500.png` – Training/validation performance graph

## Usage Instructions

### Loading the Model

```python
import tensorflow as tf
import joblib

# Load the model (choose one format)
model = tf.keras.models.load_model('food_image_recognition_model.h5')  # HDF5
# OR
model = tf.keras.models.load_model('food_image_recognition_model.keras')  # Native format
# OR use SavedModel format
model = tf.saved_model.load('food_model_savedmodel')

# Load the label encoder
label_encoder = joblib.load('food_label_encoder.pkl')
```

### Making Predictions

```python
import cv2
import numpy as np

# Load and preprocess image
img = cv2.imread('your_food_image.jpg')
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized.astype(np.float32) / 255.0
img_input = img_normalized.reshape(1, 224, 224, 3)

# Predict
prediction = model.predict(img_input)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)
food_name = label_encoder.classes_[predicted_class]

print(f"Predicted food item: {food_name} (confidence: {confidence:.3f})")
```

## Model Architecture

See `model_architecture_20250629_164500.txt` for full layer details and configurations.

## Notes

* The model was trained on a balanced dataset with various Indian and Western food items.
* Augmentations (rotation, flips, zoom) were applied for generalization.
* Early stopping and learning rate scheduler were used for convergence.
* Future scope includes integrating calorie estimation using regression on detected items.

