from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import cv2
from sklearn.cluster import KMeans
import numpy as np 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
CASCADE_PATH = "haarcascade_frontalface_default.xml" 

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

@app.post("/upload-selfie")
async def upload_selfie(file: UploadFile = File(...)):
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    image = cv2.imread(temp_file_path)
    
    if image is None:
        return JSONResponse({"error": "Failed to load image."}, status_code=400)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    if len(faces) == 0:
        return JSONResponse({"error": "No faces detected"}, status_code=404)
    
    (x, y, w, h) = faces[0]
    
    face_crop = image[y:y+h, x:x+w]
    
    cropped_face_path = os.path.join(UPLOAD_DIR, f"cropped_{file.filename}")
    cv2.imwrite(cropped_face_path, face_crop)
    
    # Now that we have the cropped face, we can determine the color clusters
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    clt = KMeans(n_clusters=3)
    clt.fit(face_rgb.reshape(-1, 3))
    
    dominant_colors = clt.cluster_centers_
    dominant_colors = dominant_colors.astype(int)
    color_list = []
    
    for color in dominant_colors:
        r, g, b = color
        color_list.append({
            "r": int(r),
            "g": int(g),
            "b": int(b)
        })
    
    # Determine cluster with maximum count
    labels = clt.labels_
    (label_counts, _) = np.histogram(labels, bins=np.arange(0, clt.n_clusters + 1))
    dominant_cluster_idx = np.argmax(label_counts)
    dominant_color = clt.cluster_centers_[dominant_cluster_idx]
    dominant_color = dominant_color.astype(int).tolist()
    r, g, b = dominant_color
    
    # Calculate brightness (luminance)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Classify skin tone on brightness
    if brightness > 180:
        skin_tone = "Fair"
    elif brightness > 130:
        skin_tone = "Light-Medium"
    elif brightness > 80:
        skin_tone = "Medium"
    else:
        skin_tone = "Deep"
        
    # Classify undertone
    if r > g and r > b:
        undertone = "Warm"
    elif b > r and b > g:
        undertone = "Cool"
    else:
        undertone = "Neutral"
    
    return JSONResponse({
        "filename": file.filename,
        "face_detected": True,
        "face_coordinates": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        },
        "cropped_face_saved_as": f"cropped_{file.filename}",
        "dominant_colors": color_list,
        "dominant_color": dominant_color,
        "brightness": brightness,
        "skin_tone": skin_tone,
        "undertone": undertone,
        "message": "Face detected and cropped successfully!"
    })
