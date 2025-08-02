from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for, flash
from uuid import uuid4
import os
import base64
import logging
import hashlib
import cv2
import numpy as np
import sqlite3
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.preprocessing.image import img_to_array
from model_utils import predict_expression_from_image, predict_expression_from_video, emotion_model, class_labels
from chatbot import get_chatbot_response

# Setup
app = Flask(__name__)
app.secret_key = str(uuid4())
logging.basicConfig(level=logging.DEBUG)

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.template_filter('hash')
def hash_filter(value):
    return hashlib.md5(value.encode()).hexdigest()[:8]

# ----------------- AUTH -----------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']  # NOT name
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_pw))
            conn.commit()
        except sqlite3.IntegrityError:
            flash('Email already exists!')
            return redirect('/signup')
        conn.close()
        flash('Signup successful. Please login.')
        return redirect('/login')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash("Logged in successfully!")
            return redirect(url_for('index'))
        else:
            flash("Invalid email or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('login'))

# ----------------- MAIN APP -----------------

@app.route('/', methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if "conversations" not in session:
        session["conversations"] = []

    conversations = session["conversations"]

    if request.method == "POST":
        try:
            # Image or video upload
            if "file" in request.files:
                file = request.files["file"]
                if file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)

                    if filename.endswith(".mp4"):
                        expression = predict_expression_from_video(filepath)
                    else:
                        expression = predict_expression_from_image(filepath)
                        img = cv2.imread(filepath)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        framed_filename = f"framed_{filename}"
                        framed_filepath = os.path.join(UPLOAD_FOLDER, framed_filename)
                        cv2.imwrite(framed_filepath, img)
                        filename = framed_filename

                    new_convo = {
                        "media": filename,
                        "expression": expression,
                        "messages": [{"sender": "bot", "text": f"Hello! You look {expression}. How can I help you?"}]
                    }
                    conversations.append(new_convo)
                    session["conversations"] = conversations
                    return redirect(url_for("index"))
                else:
                    flash("Invalid file type.")
                    return redirect(url_for("index"))

            # Captured Image
            elif "captured_image" in request.form:
                image_data = request.form["captured_image"].split(",")[1]
                filename = f"capture_{uuid4().hex}.png"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(image_data))

                expression = predict_expression_from_image(filepath)
                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                framed_filename = f"framed_{filename}"
                framed_filepath = os.path.join(UPLOAD_FOLDER, framed_filename)
                cv2.imwrite(framed_filepath, img)
                filename = framed_filename

                new_convo = {
                    "media": filename,
                    "expression": expression,
                    "messages": [{"sender": "bot", "text": f"Hi! You seem {expression}. What’s up?"}]
                }
                conversations.append(new_convo)
                session["conversations"] = conversations
                return redirect(url_for("index"))

            # Chat message
            elif "message" in request.form:
                message = request.form["message"]
                if conversations:
                    last = conversations[-1]
                    last["messages"].append({"sender": "user", "text": message})
                    reply = get_chatbot_response(message)
                    last["messages"].append({"sender": "bot", "text": reply})
                    session["conversations"] = conversations
                return redirect(url_for("index"))

        except Exception as e:
            logging.error(f"Error during POST handling: {e}")
            flash("An error occurred. Please try again.")
            return redirect(url_for("index"))

    return render_template("index.html", conversations=conversations, username=session.get("username"))

@app.route('/delete_chats', methods=['POST'])
def delete_chats():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    session["conversations"] = []
    flash("All chats have been deleted.")
    return redirect(url_for('index'))

@app.route('/live-expression', methods=['POST'])
def live_expression():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        expression = "No face detected"
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi, verbose=0)[0]
            expression = class_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"

        return {"expression": expression, "image": img_data_url}
    except Exception as e:
        logging.error(f"Error in live-expression: {e}")
        return {"expression": "Error", "image": ""}

@app.route("/uploads/<filename>")
def uploaded_file_route(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)