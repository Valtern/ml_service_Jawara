from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import cv2 as cv
import numpy as np
import pickle
import os
import dlib

app = FastAPI()

ORB_FEATURES = 5000       
MATCH_RATIO = 0.75        
RANSAC_THRESH = 5.0      


INLIER_THRESHOLD_PER_FRAME = 10 
REQUIRED_CONSENSUS_VOTES = 2

MODEL_DIR = "models"

detector = dlib.get_frontal_face_detector()
orb = cv.ORB_create(nfeatures=ORB_FEATURES)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def get_sharpness(image):
    return cv.Laplacian(image, cv.CV_64F).var()

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
    if img is None: return None, None, 0.0

    # 1. Strong Enhancement
    img = adjust_gamma(img, gamma=1.5)

    # 2. Detection
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
    
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    face_img = clahe.apply(face_img)
    
    sharpness = get_sharpness(face_img)

    kp, des = orb.detectAndCompute(face_img, None)
    if des is None: return None, None, 0.0
            
    return kp, des, sharpness

@app.post("/enroll")
async def enroll_face(user_id: str = Form(...), files: list[UploadFile] = File(...)):
    candidates = []
    print(f"Enrolling User {user_id}...")

    for file in files:
        content = await file.read()
        kp, des, score = process_image(content)
        
        if des is not None and kp is not None:
            kp_coords = [p.pt for p in kp]
            candidates.append({'des': des, 'kps': kp_coords, 'score': score})
    
    # Sort by sharpness, keep top 25 (More samples = better chance of 2 votes)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best_faces = candidates[:25]
    
    print(f"Captured {len(candidates)}. Keeping {len(best_faces)}.")

    if len(best_faces) < 5:
        return {"status": "failed", "message": "Low quality. Please stay still."}

    final_data = [{'des': f['des'], 'kps': f['kps']} for f in best_faces]

    save_path = os.path.join(MODEL_DIR, f"user_{user_id}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(final_data, f)
        
    return {"status": "success", "model_path": save_path}

def compute_inliers(des_login, kp_login, stored_des, stored_kps):
    if stored_des is None or len(stored_des) < 2: return 0
    try:
        matches = bf.knnMatch(des_login, stored_des, k=2)
    except: return 0

    good_matches = []
    for m, n in matches:
        if m.distance < MATCH_RATIO * n.distance:
            good_matches.append(m)
    
    # Relaxed minimum to attempt geometry
    if len(good_matches) < 6: return 0

    src_pts = np.float32([ kp_login[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ stored_kps[m.trainIdx] for m in good_matches ]).reshape(-1,1,2)
    
    try:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, RANSAC_THRESH)
        if mask is not None:
            # Determinant check (0.5 - 2.0 range)
            det = np.linalg.det(M[0:2, 0:2])
            if det < 0.5 or det > 2.0: return 0 
            return sum(mask.ravel().tolist())
    except: pass
    
    return 0

@app.post("/verify")
async def verify_face(user_id: str = Form(...), file: UploadFile = File(...)):
    model_path = os.path.join(MODEL_DIR, f"user_{user_id}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="User not enrolled")

    with open(model_path, 'rb') as f:
        enrolled_faces = pickle.load(f)

    content = await file.read()
    kp_login, des_login, _ = process_image(content)
    
    if des_login is None:
        return {"status": "failed", "message": "No face detected"}

    votes = 0
    total_score_sum = 0
    
    print(f"--- Verifying User {user_id} ---")
    
    for i, sample in enumerate(enrolled_faces):
        score = compute_inliers(des_login, kp_login, sample['des'], sample['kps'])
        
        # Log all non-zero scores for debugging
        if score > 5:
            print(f"  Sample {i}: Score {score}")

        if score >= INLIER_THRESHOLD_PER_FRAME:
            votes += 1
            total_score_sum += score

    print(f"Votes: {votes} (Required: {REQUIRED_CONSENSUS_VOTES}) | Total Score: {total_score_sum}")


    is_match = (votes >= REQUIRED_CONSENSUS_VOTES) or (total_score_sum > 35 and votes >= 1)

    return {
        "status": "success",
        "match": is_match,
        "score": votes
    }