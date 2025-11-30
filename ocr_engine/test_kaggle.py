import os
import cv2
import shutil
import sys
from ktp_scanner import KTPScanner

KAGGLE_DIR = "0"
OUTPUT_DIR = "kaggle_results"
LOG_FILE = "kaggle_test_log.txt"

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

def test_kaggle():
    print(f"--- Testing Scanner on Kaggle Data ---")
    print(f"Reading from: {os.path.abspath(KAGGLE_DIR)}")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    try:
        # UPDATED: No model_path needed. 
        # Points to "templates" folder relative to this script
        ocr = KTPScanner(template_dir="templates")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load scanner. {e}")
        return
    
    if not os.path.exists(KAGGLE_DIR):
        print("Error: Kaggle directory not found.")
        return

    files = [f for f in os.listdir(KAGGLE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(files)} images.")
    
    success_count = 0
    error_count = 0
    
    for i, filename in enumerate(files):
        path = os.path.join(KAGGLE_DIR, filename)
        
        with open(path, "rb") as f:
            image_bytes = f.read()
            
        print(f"[{i+1}/{len(files)}] Scanning {filename}...", end=" ")
        
        try:
            result = ocr.scan(image_bytes)
            
            if result["success"]:
                print(f"✅ NIK: {result['nik']}")
                success_count += 1
                shutil.copy(path, os.path.join(OUTPUT_DIR, f"SUCCESS_{result['nik']}_{filename}"))
            else:
                print(f"❌ {result['message']}")
        except Exception as e:
            print(f"🔥 CRASH: {e}")
            error_count += 1
            
    print(f"\n--- TEST COMPLETE ---")
    print(f"Total: {len(files)}")
    print(f"Success: {success_count} ({success_count/len(files)*100:.1f}%)")
    print(f"Crashes: {error_count}")

if __name__ == "__main__":
    test_kaggle()