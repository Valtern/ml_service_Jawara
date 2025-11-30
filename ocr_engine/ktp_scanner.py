import cv2
import numpy as np
import os

class KTPScanner:
    def __init__(self, template_dir="ocr_engine/templates"):
        self.templates = {}
        self.load_templates(template_dir)

    def load_templates(self, template_dir):
        # Fallback if path is relative to main.py vs direct execution
        if not os.path.exists(template_dir):
            if os.path.exists("templates"):
                template_dir = "templates"
            elif os.path.exists("ocr_engine/templates"):
                template_dir = "ocr_engine/templates"
        
        print(f"Loading templates from: {os.path.abspath(template_dir)}")
        
        for i in range(10):
            path = os.path.join(template_dir, f"{i}.jpg")
            if os.path.exists(path):
                self.templates[str(i)] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Warning: Template {i}.jpg not found at {path}")

    def preprocess_image(self, img):
        target_width = 1000 
        h, w = img.shape[:2]
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)))
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        grad = cv2.addWeighted(gray, 1.0, blackhat, 0.5, 0)
        
        return img, cv2.GaussianBlur(grad, (3,3), 0)

    def get_candidate_regions(self, gray_img):
        h_img, w_img = gray_img.shape
        
        thresh = cv2.adaptiveThreshold(gray_img, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)

        k_w = int(w_img / 25) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            if 6.0 < aspect_ratio < 25.0:
                if w_img * 0.25 < w < w_img * 0.95:
                    if y < h_img * 0.6:
                        candidates.append((x, y, w, h))
        
        candidates.sort(key=lambda b: b[1])
        return candidates[:5]

    def match_templates(self, roi_gray):
        roi_h, roi_w = roi_gray.shape
        
        # Resize ROI to a standard height for matching
        target_height = 40
        scale = target_height / roi_h
        target_width = int(roi_w * scale)
        
        roi_resized = cv2.resize(roi_gray, (target_width, target_height))
        
        # Binarize for clean matching
        roi_thresh = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        detected_digits = []

        for char, template in self.templates.items():
            if template is None: continue
            
            t_h, t_w = template.shape
            
            # Scale template to match ROI height (~85%)
            aspect = t_w / t_h
            new_h = int(target_height * 0.85) 
            new_w = int(new_h * aspect)
            t_resized = cv2.resize(template, (new_w, new_h))
            
            # Template Match
            res = cv2.matchTemplate(roi_thresh, t_resized, cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.60 
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                detected_digits.append((pt[0], res[pt[1], pt[0]], char))

        if not detected_digits: return ""

        # Sort by confidence
        detected_digits.sort(key=lambda x: x[1], reverse=True)
        
        final_digits = []
        taken_mask = np.zeros(target_width)
        
        digit_width = int(target_height * 0.6) 

        # Non-Maximum Suppression (Remove duplicate matches in same spot)
        for x, score, char in detected_digits:
            start = max(0, x)
            end = min(target_width, x + digit_width)
            
            overlap = np.mean(taken_mask[start:end])
            if overlap < 0.2: 
                final_digits.append((x, char))
                taken_mask[start:end] = 1

        # Sort Left-to-Right
        final_digits.sort(key=lambda x: x[0])
        
        return "".join([x[1] for x in final_digits])

    def scan(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return {"success": False, "message": "Invalid image"}

        original, gray = self.preprocess_image(img)
        candidates = self.get_candidate_regions(gray)
        
        if not candidates:
            return {"success": False, "message": "No text bars found"}

        best_nik = None
        
        for i, (x, y, w, h) in enumerate(candidates):
            roi = gray[y:y+h, x:x+w]
            
            text = self.match_templates(roi)
            
            # Basic validation
            if len(text) >= 14 and len(text) <= 18:
                best_nik = text
                # Debug output
                if not os.path.exists("debug_output"): os.makedirs("debug_output")
                cv2.imwrite(f"debug_output/nik_match_{i}.jpg", roi)
                break

        if best_nik:
            return {"success": True, "nik": best_nik}

        return {"success": False, "message": "Text found but no NIK detected"}