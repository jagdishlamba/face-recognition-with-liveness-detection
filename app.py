from flask import Flask, render_template, request, redirect, flash, Response
import os
import cv2
import numpy as np
import pickle
import faiss
import insightface
import csv
from datetime import datetime
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Paths
EMBEDDINGS_FILE = "known_faces.pkl"
TRAIN_IMAGES_FOLDER = "Train"
UPLOAD_FOLDER = "uploads"
FORM_DETAILS_CSV = "form_details.csv"
DETECTION_DETAILS_CSV = "detection_details.csv"

os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Face Recognition Model
recognition_model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognition_model.prepare(ctx_id=0, det_size=(640, 480))

# Dlib detector and predictor for liveness detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib's website

# Liveness detection thresholds
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 3

# Load or initialize known embeddings
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

known_faces = load_embeddings()

# Prepare Faiss index
def prepare_faiss_index(known_faces):
    embeddings = []
    ids = []
    for person_id, emb_list in known_faces.items():
        for emb in emb_list:
            embeddings.append(emb)
            ids.append(person_id)
    embeddings = np.array(embeddings, dtype='float32') if embeddings else np.empty((0, 512), dtype='float32')
    index = faiss.IndexFlatL2(embeddings.shape[1]) if embeddings.size > 0 else faiss.IndexFlatL2(512)
    index.add(embeddings)
    return index, ids

faiss_index, person_ids = prepare_faiss_index(known_faces)

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def recognize_face(face_embedding, threshold=1.0):
    face_embedding = np.array([face_embedding], dtype='float32')
    distances, indices = faiss_index.search(face_embedding, 1)
    if distances[0][0] < threshold:
        return person_ids[indices[0][0]]
    return "Unknown"

# Initialize CSV files
def initialize_csv_files():
    if not os.path.exists(FORM_DETAILS_CSV):
        with open(FORM_DETAILS_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Name", "Age", "Faculty", "Branch", "Mobile", "Address", "Image Path"])
    
    if not os.path.exists(DETECTION_DETAILS_CSV):
        with open(DETECTION_DETAILS_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Identity", "Source", "Liveness", "Image Path"])

initialize_csv_files()

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        name = request.form.get("name")
        age = request.form.get("age")
        faculty = request.form.get("faculty")
        branch = request.form.get("branch")
        mobile = request.form.get("mobile")
        address = request.form.get("address")
        image_file = request.files.get("image")

        if not all([name, age, faculty, branch, mobile, address, image_file]):
            flash("All fields are required.")
            return redirect("/")

        person_folder = os.path.join(TRAIN_IMAGES_FOLDER, name)
        os.makedirs(person_folder, exist_ok=True)
        image_path = os.path.join(person_folder, image_file.filename)
        image_file.save(image_path)

        img = cv2.imread(image_path)
        faces = recognition_model.get(img)
        if faces:
            face_embedding = normalize_embedding(faces[0].embedding)
            known_faces[name] = known_faces.get(name, []) + [face_embedding]

            with open(EMBEDDINGS_FILE, "wb") as f:
                pickle.dump(known_faces, f)

            global faiss_index, person_ids
            faiss_index, person_ids = prepare_faiss_index(known_faces)

            with open(FORM_DETAILS_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), name, age, faculty, branch, mobile, address, image_path])

            flash(f"Model trained successfully for {name}.")
        else:
            flash("No face detected in the uploaded image.")
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
    return redirect("/")

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        source = request.form.get("source")
        image_file = request.files.get("image")

        if source == "webcam":
            return redirect("/webcam_feed_liveness")
        
        if source == "image" and not image_file:
            flash("No image provided.")
            return redirect("/")

        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        img = cv2.imread(image_path)
        faces = recognition_model.get(img)
        if not faces:
            flash("No face detected in the uploaded image.")
            return redirect("/")
        
        results = []
        for face in faces:
            face_embedding = normalize_embedding(face.embedding)
            identity = recognize_face(face_embedding)
            results.append(identity)

            with open(DETECTION_DETAILS_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), identity, "Image", "N/A", image_path])

        flash(f"Recognition results: {', '.join(results)}")
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
    return redirect("/")

@app.route("/webcam_feed_liveness")
def webcam_feed_liveness():
    return Response(gen_frames_with_liveness(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_frames_with_liveness():
    cap = cv2.VideoCapture(0)
    blink_count = 0
    liveness_confirmed = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            right_eye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_count += 1
            else:
                if blink_count >= CONSECUTIVE_FRAMES:
                    liveness_confirmed = True
                blink_count = 0

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "Live" if liveness_confirmed else "Not Live"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if liveness_confirmed else (0, 0, 255), 2)

        if liveness_confirmed:
            faces = recognition_model.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                face_embedding = normalize_embedding(face.embedding)
                identity = recognize_face(face_embedding)

                with open(DETECTION_DETAILS_CSV, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), identity, "Webcam", "Live", "Webcam"])

                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True)
