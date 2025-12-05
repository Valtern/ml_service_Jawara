import cv2
import numpy as np
import easyocr
import re
import base64
import os

class KTPScanner:
    def __init__(self, template_dir=None):
        print("LOADING KTP SCANNER")
        
        # 1. Text Engine: EasyOCR
        self.reader = easyocr.Reader(['id', 'en'], gpu=True)
        
        # 2. Face Engine: Haar Cascade 
        self.face_cascade_path = "models/haarcascade_frontalface_default.xml"
        if not os.path.exists(self.face_cascade_path):
            if os.path.exists(f"../{self.face_cascade_path}"):
                self.face_cascade_path = f"../{self.face_cascade_path}"
            else:
                print(f"WARNING: Haar Cascade not found at {self.face_cascade_path}")
                
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

    def preprocess_image(self, img):
        target_width = 1000
        h, w = img.shape[:2]
        scale = target_width / w
        img = cv2.resize(img, (target_width, int(h * scale)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        return img, gray

    def clean_nik(self, text):
        replacements = {
            'O': '0', 'D': '0', 'Q': '0', 'U': '0', 'C': '0', 'o': '0',
            'L': '1', 'I': '1', '!': '1', '|': '1', 'l': '1', 'i': '1',
            'Z': '2', '?': '7', 'T': '7', '/': '7',
            'S': '5', '$': '5', 's': '5',
            'G': '6', 'b': '6',
            'B': '8', '&': '8',
            'g': '9', 'q': '9', 'P': '9'
        }
        
        text = text.upper()
        clean = ""
        digit_count = 0
        
        for char in text:
            if char.isdigit():
                clean += char
                digit_count += 1
            elif char in replacements:
                clean += replacements[char]
                digit_count += 1
                
        if len(clean) >= 15 and len(clean) <= 17 and digit_count >= 13:
            if clean[0] == '0': return None 
            return clean[:16]
            
        return None

    def extract_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_img = None
        
        # 1. Try Haar Cascade 
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                
                # Add liberal padding 
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.3)
                
                y1 = max(0, y - pad_h)
                y2 = min(img.shape[0], y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(img.shape[1], x + w + pad_w)
                
                face_img = img[y1:y2, x1:x2]

        # 2. Fallback: Fixed Crop 
        if face_img is None or face_img.size == 0:
            h, w = img.shape[:2]
            face_x = int(w * 0.65)
            face_y = int(h * 0.15)
            face_w = int(w * 0.32)
            face_h = int(h * 0.65)
            
            face_x = max(0, min(face_x, w - 10))
            face_y = max(0, min(face_y, h - 10))
            face_w = min(face_w, w - face_x)
            face_h = min(face_h, h - face_y)
            
            face_img = img[face_y:face_y+face_h, face_x:face_x+face_w]

        if face_img is not None and face_img.size > 0:
            _, buffer = cv2.imencode('.jpg', face_img)
            return base64.b64encode(buffer).decode('utf-8')
            
        return None

    def scan(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return {"success": False, "message": "Invalid image"}

        img, gray = self.preprocess_image(img)
        
        # 1. Run EasyOCR
        results = self.reader.readtext(img, detail=0)
        
        nik = None
        name = None
        gender = None
        
        # 2. Global Search for NIK
        full_blob = "".join(results).replace(" ", "")
        
        potential_niks = re.findall(r'[0-9OIl\?]{16}', full_blob.upper())
        for cand in potential_niks:
            cleaned = self.clean_nik(cand)
            if cleaned:
                nik = cleaned
                break
        
        if not nik:
            for line in results:
                cleaned = self.clean_nik(line)
                if cleaned:
                    nik = cleaned
                    break

        # 3. Find Name
        for i, line in enumerate(results):
            if "NAMA" in line.upper():
                if i + 1 < len(results):
                    candidate = results[i+1]
                    if ":" not in candidate and len(candidate) > 3:
                        name = re.sub(r'[^A-Z\s]', '', candidate.upper()).strip()
                        break
                
                if ":" in line:
                    candidate = line.split(":")[1]
                    if len(candidate) > 3:
                        name = re.sub(r'[^A-Z\s]', '', candidate.upper()).strip()
                        break
        
        # 4. Find Gender
        full_str = " ".join(results).upper()
        if "LAKI" in full_str:
            gender = "Laki-laki"
        elif "PEREMPUAN" in full_str or "PUAN" in full_str:
            gender = "Perempuan"

        # 5. Extract Face (Using Haar Cascade + Fallback)
        face_b64 = self.extract_face(img)

        if nik:
            return {
                "success": True,
                "nik": nik,
                "name": name,
                "gender": gender,
                "face_image": face_b64
            }
        
        return {"success": False, "message": "NIK not found. Ensure text is clear."}