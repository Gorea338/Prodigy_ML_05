import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2

# Configuration
class Config:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    IMG_SIZE = 224
    NUM_CLASSES = 101
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = 'food_recognition_model.pth'
    
    # Data paths - MODIFY THESE PATHS TO MATCH YOUR SETUP
    # Option 1: If you downloaded from Kaggle and extracted
    DATA_PATH = r'C:\Users\vikas\Downloads\archive (3)\food-101\food-101'  # Fixed path
    
    # Option 2: If using Kaggle directly
    # DATA_PATH = '/kaggle/input/food-101/food-101'
    
    # Option 3: If dataset is in current directory
    # DATA_PATH = './food-101'
    
    IMAGES_PATH = os.path.join(DATA_PATH, 'images')
    META_PATH = os.path.join(DATA_PATH, 'meta')

# Calorie database (estimated calories per 100g for Food-101 classes)
CALORIE_DATABASE = {
    'apple_pie': 237, 'baby_back_ribs': 292, 'baklava': 429, 'beef_carpaccio': 114,
    'beef_tartare': 196, 'beet_salad': 58, 'beignets': 316, 'bibimbap': 121,
    'bread_pudding': 212, 'breakfast_burrito': 206, 'bruschetta': 195, 'caesar_salad': 158,
    'cannoli': 297, 'caprese_salad': 168, 'carrot_cake': 415, 'ceviche': 125,
    'cheese_plate': 368, 'cheesecake': 321, 'chicken_curry': 169, 'chicken_quesadilla': 258,
    'chicken_wings': 203, 'chocolate_cake': 371, 'chocolate_mousse': 168, 'churros': 325,
    'clam_chowder': 105, 'club_sandwich': 279, 'crab_cakes': 169, 'creme_brulee': 235,
    'croque_madame': 268, 'cup_cakes': 305, 'deviled_eggs': 155, 'donuts': 452,
    'dumplings': 141, 'edamame': 121, 'eggs_benedict': 274, 'escargots': 184,
    'falafel': 333, 'filet_mignon': 227, 'fish_and_chips': 231, 'foie_gras': 462,
    'french_fries': 365, 'french_onion_soup': 57, 'french_toast': 166, 'fried_calamari': 175,
    'fried_rice': 163, 'frozen_yogurt': 127, 'garlic_bread': 350, 'gnocchi': 131,
    'greek_salad': 89, 'grilled_cheese_sandwich': 291, 'grilled_salmon': 206, 'guacamole': 160,
    'gyoza': 193, 'hamburger': 295, 'hot_and_sour_soup': 45, 'hot_dog': 290,
    'huevos_rancheros': 206, 'hummus': 166, 'ice_cream': 207, 'lasagna': 135,
    'lobster_bisque': 142, 'lobster_roll_sandwich': 436, 'macaroni_and_cheese': 164, 'macarons': 391,
    'miso_soup': 40, 'mussels': 172, 'nachos': 346, 'omelette': 154,
    'onion_rings': 331, 'oysters': 68, 'pad_thai': 153, 'paella': 138,
    'pancakes': 227, 'panna_cotta': 175, 'peking_duck': 337, 'pho': 46,
    'pizza': 266, 'pork_chop': 231, 'poutine': 740, 'prime_rib': 304,
    'pulled_pork_sandwich': 233, 'ramen': 188, 'ravioli': 175, 'red_velvet_cake': 478,
    'risotto': 142, 'samosa': 262, 'sashimi': 127, 'scallops': 111,
    'seaweed_salad': 45, 'shrimp_and_grits': 151, 'spaghetti_bolognese': 151, 'spaghetti_carbonara': 173,
    'spring_rolls': 140, 'steak': 271, 'strawberry_shortcake': 178, 'sushi': 143,
    'tacos': 226, 'takoyaki': 112, 'tiramisu': 240, 'tuna_tartare': 144,
    'waffles': 291
}

# Custom Dataset Class - FIXED MAGIC METHODS
class Food101Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):  # Fixed __init__
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):  # Fixed __len__
        return len(self.image_paths)
    
    def __getitem__(self, idx):  # Fixed __getitem__
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data preprocessing and augmentation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Optimized Model Architecture - FIXED MAGIC METHOD
class FoodRecognitionModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):  # Fixed __init__
        super(FoodRecognitionModel, self).__init__()  # Fixed super call
        
        # Use EfficientNet as backbone for better performance
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Modify classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Additional head for calorie regression
        self.calorie_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.ReLU()  # Calories should be positive
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.backbone.classifier[:-1](features)  # Get features before final layer
        
        # Classification output
        classification = self.backbone.classifier[-1](features)
        
        # Calorie estimation output
        calories = self.calorie_head(features)
        
        return classification, calories

# Data loading function
def load_data():
    """Load and prepare Food-101 dataset"""
    print(f"Looking for dataset at: {Config.DATA_PATH}")
    print(f"Images path: {Config.IMAGES_PATH}")
    print(f"Meta path: {Config.META_PATH}")
    
    # Check if paths exist
    if not os.path.exists(Config.DATA_PATH):
        print(f"ERROR: Dataset path not found: {Config.DATA_PATH}")
        print("Please update the DATA_PATH in Config class to match your dataset location")
        return None, None, None
        
    if not os.path.exists(Config.IMAGES_PATH):
        print(f"ERROR: Images path not found: {Config.IMAGES_PATH}")
        return None, None, None
        
    if not os.path.exists(Config.META_PATH):
        print(f"ERROR: Meta path not found: {Config.META_PATH}")
        return None, None, None
    
    # Read class names
    classes_file = os.path.join(Config.META_PATH, 'classes.txt')
    if not os.path.exists(classes_file):
        print(f"ERROR: Classes file not found: {classes_file}")
        return None, None, None
        
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(classes)} classes")
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Read train and test splits
    train_file = os.path.join(Config.META_PATH, 'train.txt')
    test_file = os.path.join(Config.META_PATH, 'test.txt')
    
    if not os.path.exists(train_file):
        print(f"ERROR: Train file not found: {train_file}")
        return None, None, None
        
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        return None, None, None
    
    with open(train_file, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    with open(test_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(train_files)} training files and {len(test_files)} test files")
    
    # Prepare training data
    train_paths = []
    train_labels = []
    train_calories = []
    
    missing_count = 0
    for file_path in train_files[:1000]:  # Limit to first 1000 for testing
        class_name = file_path.split('/')[0]
        full_path = os.path.join(Config.IMAGES_PATH, file_path + '.jpg')
        if os.path.exists(full_path):
            train_paths.append(full_path)
            train_labels.append(class_to_idx[class_name])
            train_calories.append(CALORIE_DATABASE.get(class_name, 200))  # Default 200 cal/100g
        else:
            missing_count += 1
    
    print(f"Loaded {len(train_paths)} training images, {missing_count} missing")
    
    # Prepare test data
    test_paths = []
    test_labels = []
    test_calories = []
    
    missing_count = 0
    for file_path in test_files[:200]:  # Limit to first 200 for testing
        class_name = file_path.split('/')[0]
        full_path = os.path.join(Config.IMAGES_PATH, file_path + '.jpg')
        if os.path.exists(full_path):
            test_paths.append(full_path)
            test_labels.append(class_to_idx[class_name])
            test_calories.append(CALORIE_DATABASE.get(class_name, 200))
        else:
            missing_count += 1
    
    print(f"Loaded {len(test_paths)} test images, {missing_count} missing")
    
    return (train_paths, train_labels, train_calories), (test_paths, test_labels, test_calories), classes

# Training function
def train_model():
    # Load data
    result = load_data()
    if result[0] is None:
        print("Failed to load data. Please check your dataset paths.")
        return None, None, None, None
        
    (train_paths, train_labels, train_calories), (test_paths, test_labels, test_calories), classes = result
    
    if len(train_paths) == 0:
        print("No training data found. Please check your dataset.")
        return None, None, None, None
    
    # Create transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = Food101Dataset(train_paths, train_labels, train_transform)
    test_dataset = Food101Dataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    model = FoodRecognitionModel().to(Config.DEVICE)
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    # Optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
        {'params': model.calorie_head.parameters(), 'lr': Config.LEARNING_RATE}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    
    # Reduce epochs for testing
    epochs_to_run = min(5, Config.EPOCHS)
    print(f"Starting training for {epochs_to_run} epochs...")
    
    for epoch in range(epochs_to_run):
        model.train()
        running_loss = 0.0
        
        print(f"Starting epoch {epoch+1}/{epochs_to_run}")
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            # Get corresponding calorie values
            calorie_targets = torch.tensor([CALORIE_DATABASE.get(classes[label.item()], 200) 
                                          for label in labels]).float().to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_outputs, calorie_outputs = model(images)
            
            # Calculate losses
            class_loss = classification_criterion(class_outputs, labels)
            calorie_loss = regression_criterion(calorie_outputs.squeeze(), calorie_targets)
            
            # Combined loss
            total_loss = class_loss + 0.1 * calorie_loss  # Weight calorie loss lower
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
        
        # Validation
        print("Running validation...")
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                class_outputs, _ = model(images)
                
                _, predicted = torch.max(class_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs_to_run}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        scheduler.step()
        
        # Save model after each epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy,
            'classes': classes
        }, Config.MODEL_SAVE_PATH)
        print(f"Model saved to {Config.MODEL_SAVE_PATH}")
    
    return model, classes, train_losses, val_accuracies

# Inference function
def predict_food_and_calories(model, image_path, classes, transform):
    """Predict food class and estimate calories"""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None
        
    model.eval()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            class_outputs, calorie_outputs = model(image_tensor)
            
            # Get predictions
            probabilities = torch.softmax(class_outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_food = classes[predicted_class.item()]
            confidence_score = confidence.item()
            estimated_calories_per_100g = calorie_outputs.item()
            
            # Estimate portion size (simple approximation based on image)
            # In a real application, you might use object detection for better estimation
            estimated_portion_g = 150  # Default portion size
            total_calories = (estimated_calories_per_100g * estimated_portion_g) / 100
            
            return {
                'food_item': predicted_food,
                'confidence': confidence_score,
                'calories_per_100g': estimated_calories_per_100g,
                'estimated_portion_g': estimated_portion_g,
                'total_calories': total_calories
            }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Utility function for model evaluation
def evaluate_model(model, test_loader, classes):
    """Comprehensive model evaluation"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            class_outputs, _ = model(images)
            
            _, predicted = torch.max(class_outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    
    return accuracy

# Test function to check if everything works
def test_setup():
    """Test if the setup is working correctly"""
    print("Testing setup...")
    print(f"Using device: {Config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test data loading
    result = load_data()
    if result[0] is None:
        print("Setup test FAILED: Cannot load data")
        return False
    else:
        print("Setup test PASSED: Data loading works")
        return True

# Main execution - FIXED MAGIC METHOD
if __name__ == "__main__":  # Fixed __main__
    print("=" * 50)
    print("FOOD RECOGNITION MODEL TRAINING")
    print("=" * 50)
    
    # Test setup first
    if not test_setup():
        print("\nPlease fix the data path issues before training.")
        print("Update the DATA_PATH in the Config class to point to your Food-101 dataset.")
        exit(1)
    
    print(f"Using device: {Config.DEVICE}")
    print(f"Training Food Recognition Model...")
    
    # Train model
    result = train_model()
    if result[0] is None:
        print("Training failed. Please check the error messages above.")
        exit(1)
        
    model, classes, train_losses, val_accuracies = result
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    
    # Test prediction if you have a test image
    # Uncomment and modify the path below to test prediction
    
    if os.path.exists('test_image.jpg'):
        _, val_transform = get_transforms()
        result = predict_food_and_calories(model, 'test_image.jpg', classes, val_transform)
        if result:
            print(f"\nPrediction Result: {result}")
        else:
            print("\nPrediction failed")
    