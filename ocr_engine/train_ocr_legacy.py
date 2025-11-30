import cv2
import numpy as np
import os

DATASET_DIR = "dataset"
MODEL_OUTPUT = "../models/ocr_hog_svm.xml" # New Model Name
IMG_SIZE = 32

def get_hog(img):
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(img).flatten()

def train():
    print("--- Training SVM (HOG) ---")
    if not os.path.exists(DATASET_DIR): return

    samples = []
    labels = []
    
    classes = sorted([d for d in os.listdir(DATASET_DIR) if not d.startswith(".")])
    
    for label_str in classes:
        class_dir = os.path.join(DATASET_DIR, label_str)
        try:
            label = int(label_str)
        except: continue
            
        print(f"Processing Class {label}...", end='\r')
        
        for f in os.listdir(class_dir):
            path = os.path.join(class_dir, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            fd = get_hog(img)
            samples.append(fd)
            labels.append(label)

    print(f"\nTraining on {len(samples)} samples...")
    train_data = np.array(samples, dtype=np.float32)
    train_labels = np.array(labels, dtype=np.int32)
    
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.1) 
    
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    
    if not os.path.exists("../models"): os.makedirs("../models")
    svm.save(MODEL_OUTPUT)
    print(f"Saved to {MODEL_OUTPUT}")

if __name__ == "__main__":
    train()