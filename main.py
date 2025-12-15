from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import cv2 as cv
import numpy as np
import pickle
import os
import dlib
from ocr_engine.ktp_scanner import KTPScanner

app = FastAPI()

ORB_FEATURES = 5000       
MATCH_RATIO = 0.75        
RANSAC_THRESH = 5.0       

INLIER_THRESHOLD_PER_FRAME = 12  
REQUIRED_CONSENSUS_VOTES = 5     

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
orb = cv.ORB_create(nfeatures=ORB_FEATURES)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

try:
    ocr = KTPScanner() 
except Exception:
    ocr = None

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def get_sharpness(image):
    return cv.Laplacian(image, cv.CV_64F).var()

def process_image(image_bytes, is_enrollment=False):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    if img is None: return None, None, 0.0

    img = adjust_gamma(img, gamma=1.5)

    rects = detector(img, 1)
    if len(rects) == 0:
        img = cv.equalizeHist(img)
        rects = detector(img, 1)
        if len(rects) == 0:
            return None, None, 0.0
        
    rect = max(rects, key=lambda r: r.width() * r.height())
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    
    pad_x, pad_y = int(w*0.15), int(h*0.15)
    img_h, img_w = img.shape
    face_img = img[max(0, y-pad_y):min(img_h, y+h+pad_y), max(0, x-pad_x):min(img_w, x+w+pad_x)]
    
    try:
        face_img = cv.resize(face_img, (300, 300))
    except:
        return None, None, 0.0
    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    face_img = clahe.apply(face_img)
    
    sharpness = get_sharpness(face_img)

    kp, des = orb.detectAndCompute(face_img, None)
    if des is None: return None, None, 0.0
            
    return kp, des, sharpness

@app.post("/enroll")
async def enroll_face(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    print(f"Enrolling User: {user_id}")
    candidates = []

    for file in files:
        content = await file.read()
        kp, des, score = process_image(content, is_enrollment=True)
        
        if des is not None and kp is not None:
            kp_coords = [p.pt for p in kp]
            candidates.append({'des': des, 'kps': kp_coords, 'score': score})
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_faces = candidates[:30] 

    if len(best_faces) < 5:
        print(f"Enrollment Failed: Low Quality Data for User {user_id}")
        return {"status": "failed", "message": "Low quality. Please stay still."}

    final_data = [{'des': f['des'], 'kps': f['kps']} for f in best_faces]

    save_path = os.path.join(MODEL_DIR, f"user_{user_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(final_data, f)
        
    print(f"Success: User {user_id} enrolled with {len(best_faces)} frames.")
    return {"status": "success", "model_path": save_path, "user": user_id}

def compute_inliers(des_login, kp_login, stored_des, stored_kps):
    if stored_des is None or len(stored_des) < 2: return 0
    try:
        matches = bf.knnMatch(des_login, stored_des, k=2)
    except: return 0

    good_matches = []
    for m, n in matches:
        if m.distance < MATCH_RATIO * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 8: return 0

    src_pts = np.float32([ kp_login[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ stored_kps[m.trainIdx] for m in good_matches ]).reshape(-1,1,2)
    
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESH)
        if mask is not None:
            det = np.linalg.det(M[0:2, 0:2])
            if det < 0.2 or det > 3.0: return 0
            
            return sum(mask.ravel().tolist())
    except: pass
    
    return 0

def process_image_with_flip(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    if img is None: return []

    results = []
    
    kp1, des1, _ = process_image(image_bytes)
    if des1 is not None:
        results.append((kp1, des1))

    img_flipped = cv.flip(img, 1)
    _, buf = cv.imencode('.jpg', img_flipped)
    kp2, des2, _ = process_image(buf.tobytes())
    if des2 is not None:
        results.append((kp2, des2))
        
    return results

@app.post("/verify")
async def verify_face(user_id: str = Form(...), file: UploadFile = File(...)):
    print(f"\n--- Verifying User: {user_id} ---")
    
    model_path = os.path.join(MODEL_DIR, f"user_{user_id}.pkl")
    if not os.path.exists(model_path):
        print(f"Error: User {user_id} not found in database.")
        raise HTTPException(status_code=404, detail=f"User {user_id} not enrolled")

    with open(model_path, 'rb') as f:
        enrolled_faces = pickle.load(f)

    content = await file.read()
    
    probes = process_image_with_flip(content)
    
    if not probes:
        print(f"Result: No Face Detected for User {user_id}")
        return {"status": "failed", "message": "No face detected", "user": user_id, "votes": 0}

    max_votes = 0
    
    for kp_login, des_login in probes:
        votes = 0
        for i, sample in enumerate(enrolled_faces):
            score = compute_inliers(des_login, kp_login, sample['des'], sample['kps'])
            
            if score >= INLIER_THRESHOLD_PER_FRAME:
                votes += 1
        
        if votes > max_votes:
            max_votes = votes

    is_match = max_votes >= REQUIRED_CONSENSUS_VOTES
    
    result_str = "Success" if is_match else "Failed"
    print(f"User {user_id}: {result_str} (Votes: {max_votes}/{len(enrolled_faces)})")

    return {
        "status": "success",
        "match": is_match,
        "votes": max_votes,
        "user": user_id
    }

@app.post("/scan-ktp")
async def scan_ktp(file: UploadFile = File(...)):
    if ocr is None:
        return {"status": "error", "message": "OCR Engine not loaded"}

    content = await file.read()
    try:
        result = ocr.scan(content)
        if result["success"]:
            return {"status": "success", "data": result}
        else:
            return {"status": "failed", "message": result["message"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/verify-identity")
async def verify_identity(user_id: str = Form(...), file: UploadFile = File(...)):
    model_path = os.path.join(MODEL_DIR, f"user_{user_id}.pkl")
    if not os.path.exists(model_path):
        return {"status": "failed", "message": "User not enrolled"}

    with open(model_path, 'rb') as f:
        enrolled_faces = pickle.load(f)

    content = await file.read()
    
    probes = process_image_with_flip(content)
    
    if not probes:
        return {"status": "failed", "message": "No face in KTP", "user": user_id}

    best_score = 0
    IDENTITY_INLIER_THRESHOLD = 10 
    
    for kp_login, des_login in probes:
        for sample in enrolled_faces:
            score = compute_inliers(des_login, kp_login, sample['des'], sample['kps'])
            if score > best_score: best_score = score

    is_match = best_score >= IDENTITY_INLIER_THRESHOLD

    return {
        "status": "success",
        "match": is_match,
        "score": best_score,
        "user": user_id
    }