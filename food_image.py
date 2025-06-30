import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set memory growth for GPU (if available)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

class SimpleFoodRecognitionModel:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=16):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = []
        
        # Calorie database (calories per 100g)
        self.calorie_db = {
            'apple_pie': 237, 'baby_back_ribs': 297, 'baklava': 307, 'beef_carpaccio': 217,
            'beef_tartare': 196, 'beet_salad': 43, 'beignets': 253, 'bibimbap': 121,
            'bread_pudding': 153, 'breakfast_burrito': 255, 'bruschetta': 195, 'caesar_salad': 190,
            'cannoli': 291, 'caprese_salad': 166, 'carrot_cake': 415, 'ceviche': 87,
            'cheese_plate': 368, 'cheesecake': 321, 'chicken_curry': 165, 'chicken_quesadilla': 206,
            'chicken_wings': 203, 'chocolate_cake': 371, 'chocolate_mousse': 168, 'churros': 117,
            'clam_chowder': 95, 'club_sandwich': 282, 'crab_cakes': 155, 'creme_brulee': 340,
            'croque_madame': 272, 'cup_cakes': 305, 'deviled_eggs': 155, 'donuts': 452,
            'dumplings': 41, 'edamame': 122, 'eggs_benedict': 230, 'escargots': 90,
            'falafel': 333, 'filet_mignon': 227, 'fish_and_chips': 265, 'foie_gras': 462,
            'french_fries': 365, 'french_onion_soup': 57, 'french_toast': 166, 'fried_calamari': 175,
            'fried_rice': 163, 'frozen_yogurt': 127, 'garlic_bread': 350, 'gnocchi': 131,
            'greek_salad': 91, 'grilled_cheese_sandwich': 291, 'grilled_salmon': 231, 'guacamole': 160,
            'gyoza': 64, 'hamburger': 295, 'hot_and_sour_soup': 91, 'hot_dog': 290,
            'huevos_rancheros': 144, 'hummus': 177, 'ice_cream': 207, 'lasagna': 135,
            'lobster_bisque': 104, 'lobster_roll_sandwich': 436, 'macaroni_and_cheese': 164, 'macarons': 397,
            'miso_soup': 84, 'mussels': 172, 'nachos': 346, 'omelette': 154,
            'onion_rings': 411, 'oysters': 68, 'pad_thai': 153, 'paella': 172,
            'pancakes': 227, 'panna_cotta': 186, 'peking_duck': 337, 'pho': 46,
            'pizza': 266, 'pork_chop': 231, 'poutine': 365, 'prime_rib': 220,
            'pulled_pork_sandwich': 225, 'ramen': 436, 'ravioli': 220, 'red_velvet_cake': 478,
            'risotto': 174, 'samosa': 262, 'sashimi': 154, 'scallops': 111,
            'seaweed_salad': 45, 'shrimp_and_grits': 151, 'spaghetti_bolognese': 151, 'spaghetti_carbonara': 151,
            'spring_rolls': 140, 'steak': 271, 'strawberry_shortcake': 166, 'sushi': 143,
            'tacos': 226, 'takoyaki': 112, 'tiramisu': 240, 'tuna_tartare': 144,
            'waffles': 291
        }
    
    def check_dataset_structure(self):
        """Check and validate dataset structure"""
        print("Checking dataset structure...")
        
        # Check if images directory exists
        images_path = os.path.join(self.dataset_path, 'images')
        if not os.path.exists(images_path):
            print(f"Images directory not found at: {images_path}")
            print("Please ensure your dataset structure is:")
            print("food-101/")
            print("  └── images/")
            print("      ├── apple_pie/")
            print("      ├── baby_back_ribs/")
            print("      └── ...")
            return False
        
        # Get class names
        self.class_names = sorted([d for d in os.listdir(images_path) 
                                 if os.path.isdir(os.path.join(images_path, d))])
        
        if len(self.class_names) == 0:
            print("No food class directories found!")
            return False
        
        print(f"Found {len(self.class_names)} food classes")
        
        # Check sample counts
        sample_counts = {}
        for class_name in self.class_names[:5]:  # Check first 5 classes
            class_path = os.path.join(images_path, class_name)
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            sample_counts[class_name] = count
        
        print("Sample counts for first 5 classes:")
        for class_name, count in sample_counts.items():
            print(f"  {class_name}: {count} images")
        
        return True
    
    def create_data_generators(self):
        """Create data generators with error handling"""
        print("Creating data generators...")
        
        images_path = os.path.join(self.dataset_path, 'images')
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            validation_split=0.2
        )
        
        # Validation data generator
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        try:
            # Create generators
            train_generator = train_datagen.flow_from_directory(
                images_path,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            validation_generator = val_datagen.flow_from_directory(
                images_path,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            # Update class names from generator
            self.class_names = list(train_generator.class_indices.keys())
            
            print(f"Training samples: {train_generator.samples}")
            print(f"Validation samples: {validation_generator.samples}")
            print(f"Number of classes: {train_generator.num_classes}")
            
            return train_generator, validation_generator
            
        except Exception as e:
            print(f"Error creating data generators: {e}")
            return None, None
    
    def build_model(self):
        """Build a simple CNN model using MobileNetV2"""
        print("Building model...")
        
        # Use MobileNetV2 as base model (more compatible)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile model with simple metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully!")
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=10):
        """Train the model"""
        print("Starting training...")
        
        # Calculate steps
        steps_per_epoch = max(1, train_generator.samples // self.batch_size)
        validation_steps = max(1, validation_generator.samples // self.batch_size)
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Simple callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'best_food_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_food_and_calories(self, image_path, portion_size=100):
        """Predict food and estimate calories"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Preprocess image
        img_array = self.preprocess_single_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get predicted food class
        predicted_food = self.class_names[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = [
            (self.class_names[idx], float(predictions[0][idx]))
            for idx in top_3_idx
        ]
        
        # Estimate calories
        base_calories = self.calorie_db.get(predicted_food, 200)
        estimated_calories = (base_calories * portion_size) / 100
        
        return {
            'predicted_food': predicted_food,
            'confidence': float(confidence),
            'estimated_calories': estimated_calories,
            'portion_size_g': portion_size,
            'top_3_predictions': top_3_predictions
        }
    
    def save_model(self, filepath='food_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
            
            # Save class names
            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f)
            print("Class names saved to class_names.json")
    
    def load_model(self, filepath='food_model.h5'):
        """Load a trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            
            # Load class names
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)
            print("Class names loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main training function"""
    # Initialize model
    dataset_path = r'C:\Users\vikas\Downloads\archive (3)\food-101\food-101'
    model = SimpleFoodRecognitionModel(dataset_path, batch_size=16)
    
    # Check dataset
    if not model.check_dataset_structure():
        print("Dataset structure check failed!")
        return
    
    # Create data generators
    train_gen, val_gen = model.create_data_generators()
    if train_gen is None or val_gen is None:
        print("Failed to create data generators!")
        return
    
    # Build model
    model.build_model()
    
    # Print model summary
    print("\nModel Summary:")
    model.model.summary()
    
    # Train model
    print("\nStarting training...")
    history = model.train_model(train_gen, val_gen, epochs=10)
    
    # Save model
    model.save_model('food_recognition_model.h5')
    
    # Plot training history
    model.plot_history(history)
    
    print("Training completed!")
    
    return model

# Test prediction function
def test_prediction(model, image_path, portion_size=150):
    """Test prediction on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    result = model.predict_food_and_calories(image_path, portion_size)
    
    if result:
        print(f"\nPrediction Results:")
        print(f"Food: {result['predicted_food']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Estimated Calories: {result['estimated_calories']:.1f} kcal")
        print(f"Portion Size: {result['portion_size_g']}g")
        print(f"\nTop 3 Predictions:")
        for i, (food, conf) in enumerate(result['top_3_predictions'], 1):
            print(f"{i}. {food}: {conf:.2%}")

if __name__ == "__main__":
    # Run training
    trained_model = main()
    
    # Example usage for prediction
    # test_prediction(trained_model, 'path_to_your_test_image.jpg')
