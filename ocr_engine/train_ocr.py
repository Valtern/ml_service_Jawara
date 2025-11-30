import cv2
import numpy as np
import os

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
MODEL_OUTPUT = "../models/ocr_zonal_knn.xml"
IMG_SIZE = 32
GRID_SIZE = 4 # 4x4 grid = 16 features

def get_zonal_features(img):
    # 1. Resize to standard
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 2. Split into 4x4 zones
    zone_height = IMG_SIZE // GRID_SIZE
    zone_width = IMG_SIZE // GRID_SIZE
    
    features = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            # Extract zone
            y1 = r * zone_height
            y2 = (r + 1) * zone_height
            x1 = c * zone_width
            x2 = (c + 1) * zone_width
            
            zone = img[y1:y2, x1:x2]
            
            # Calculate pixel density (How much "white" is in this zone?)
            # Count non-zero pixels / Total pixels
            density = cv2.countNonZero(zone) / (zone_height * zone_width)
            features.append(density)
            
    return np.array(features, dtype=np.float32)

def train():
    print("--- Starting Zonal Density Training ---")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset not found at '{DATASET_DIR}'.")
        return

    training_samples = []
    training_labels = []
    
    classes = sorted([d for d in os.listdir(DATASET_DIR) if not d.startswith(".")])
    
    count = 0
    for label_str in classes:
        class_dir = os.path.join(DATASET_DIR, label_str)
        try:
            label_int = int(label_str)
        except ValueError: continue

        print(f"Loading Class {label_int}...", end='\r')
        
        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Extract Zonal Features
            sample = get_zonal_features(img)
            
            training_samples.append(sample)
            training_labels.append(label_int)
            count += 1

    print(f"\nTraining on {count} images...")
    
    train_data = np.array(training_samples, dtype=np.float32)
    train_responses = np.array(training_labels, dtype=np.int32)
    
    # k-NN is perfect for feature vectors
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_responses)
    
    if not os.path.exists("../models"): os.makedirs("../models")
    knn.save(MODEL_OUTPUT)
    print(f"SUCCESS! Zonal Model saved to {os.path.abspath(MODEL_OUTPUT)}")

if __name__ == "__main__":
    train()