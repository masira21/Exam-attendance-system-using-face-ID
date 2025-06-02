from flask import Flask, render_template, redirect,send_from_directory, url_for, jsonify, request
from flask_cors import CORS
from bson import ObjectId
from datetime import datetime,time as dt_time
import cv2
import numpy as np
import face_recognition
import time
import os
from database import db,students_collection,exams_collection,attendance_collection # ✅ Import database function
# Import Blueprints
from routes.attendance import attendance_bp
from routes.dashboard import dashboard_routes
from routes.exam import exam_bp
from routes.student_routes import student_bp  # ✅ Handles all /students routes

# ✅ Flask App Configuration
app = Flask(__name__, static_folder="static", static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 
students_collection = db["students"]  # ✅ Correctly use DB from get_database()
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend communication

print("✅ Connected to MongoDB successfully!") 

# ✅ Register Blueprints
app.register_blueprint(dashboard_routes)
app.register_blueprint(exam_bp, url_prefix="/exam")
app.register_blueprint(student_bp, url_prefix="/students")  # ✅ Student routes handled in blueprint

# ✅ Home Route (Redirects to Dashboard)
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

# ✅ Dashboard Route
@app.route('/dashboard')
def dashboard():
    print("🖥️ Rendering dashboard.html")
    return render_template('dashboard.html')

@app.route('/static/<path:filename>')  
def serve_static(filename):  
    return send_from_directory('static', filename)

# ✅ Capture Face Encoding (For Frontend)
@app.route("/capture_face", methods=["POST"])
def capture_face():
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return jsonify({"message": "❌ Could not open webcam!"}), 500

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({"message": "❌ Failed to capture image!"}), 500

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_locations = face_recognition.face_locations(frame)

        if not face_locations:
            return jsonify({"message": "⚠️ No face detected! Try again."}), 400

        face_encodings = face_recognition.face_encodings(frame, face_locations)

        if len(face_encodings) == 0:
            return jsonify({"message": "⚠️ No face encoding found!"}), 400

        return jsonify({
            "success":True,
            "message": "✅ Face captured!",
            "face_encoding": face_encodings[0].tolist()
        })

    except Exception as e:
        return jsonify({"message": f"❌ Server error: {str(e)}"}), 500


@app.route('/register_student', methods=['POST'])
def register_student():
    try:
        data = request.get_json()
        print("📥 Received data:", data)

        # Validate input fields
        required_fields = ["name", "student_id", "course", "year", "face_encoding"]
        if not all(key in data for key in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        if not data["face_encoding"]:
            return jsonify({"error": "Face encoding is missing"}), 400

        # Check for duplicate student_id
        if students_collection.find_one({"student_id": data["student_id"]}):
            return jsonify({"error": "Student ID already registered!"}), 409

        # Store student in MongoDB
        student = {
            "name": data["name"],
            "student_id": data["student_id"],
            "course": data["course"],
            "year": data["year"],
            "face_encoding": data["face_encoding"].tolist() if isinstance(data["face_encoding"], np.ndarray) else data["face_encoding"]
        }
        students_collection.insert_one(student)

        print("✅ Student registered:", student)
        return jsonify({"message": "Student registered successfully!"}), 200

    except Exception as e:
        print("❌ Error processing request:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500

    
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()

    if not data or 'face_encoding' not in data:
        print("❌ No face encoding received in request.")
        return jsonify({"message": "❌ No face encoding provided!"}), 400

    try:
        student_face_encoding = np.array(data['face_encoding'], dtype=np.float64)
        print("🔍 Received face encoding (first 5 values):", student_face_encoding[:5])
    except Exception as e:
        print("❌ Error converting face encoding:", str(e))
        return jsonify({"message": "❌ Invalid face encoding format!"}), 400

    today_date = datetime.now().date()
    students = students_collection.find({})
    print(f"📦 Total students in DB: {students_collection.count_documents({})}")

    best_match = None
    best_match_distance = float("inf")

    for student in students:
        if "face_encoding" not in student or not student["face_encoding"]:
            continue

        stored_encoding = np.array(student["face_encoding"], dtype=np.float64)
        distance = face_recognition.face_distance([stored_encoding], student_face_encoding)[0]

        print(f"🧪 Comparing with {student.get('name', 'Unknown')}, Distance: {distance:.4f}")

        if distance < best_match_distance:
            best_match = student
            best_match_distance = distance

    print(f"✅ Best Match Distance: {best_match_distance}")

    if best_match and best_match_distance < 0.5:
        student_course = best_match.get("course", "unknown")
        student_year = best_match.get("year", "unknown")
        student_id = best_match.get("student_id") or str(best_match["_id"])
        best_match['_id'] = str(best_match['_id'])

        exam = exams_collection.find_one({
            "course_id": student_course,
            "exam_date": {
                "$gte": datetime.combine(today_date, dt_time.min),
                "$lt": datetime.combine(today_date, dt_time.max)
            }
        })

        if not exam:
            print(f"⚠️ No exam scheduled today for {student_course}")
            return jsonify({"message": f"⚠️ No exam scheduled today for {student_course} ({student_year} Year)!"}), 400

        exam_name = exam["exam_name"]
        exam_date = exam["exam_date"]
        timestamp = datetime.now()

        # 🛑 Check if already marked present
        already_present = attendance_collection.find_one({
            "student_id": student_id,
            "exam_name": exam_name,
            "exam_date": exam_date.strftime("%Y-%m-%d"),
            "status": "Present"
        })

        if already_present:
            return jsonify({
                "status": "info",
                "message": f"⚠️ {best_match['name']} already marked present for {exam_name}!"
            }), 200

        attendance_record = {
            "student_id": student_id,
            "name": best_match["name"],
            "course": student_course,
            "year": student_year,
            "exam_name": exam_name,
            "exam_date": exam_date.strftime("%Y-%m-%d"),
            "status": "Present",
            "timestamp": timestamp,
            "date": timestamp.strftime("%Y-%m-%d")
        }

        # ✅ Update student record
        students_collection.update_one(
            {"student_id": student_id},
            {"$push": {"attendance": attendance_record}}
        )

        # ✅ Insert into global attendance collection
        attendance_collection.insert_one(attendance_record)

        print(f"🎉 Attendance marked for {best_match['name']} in {exam_name} exam.")

        # ✅ Step: Mark absent students for same course & exam
        all_students = students_collection.find({"course": student_course})
        for student in all_students:
            sid = student.get("student_id")
            if not sid or sid == student_id:
                continue

            already_marked = attendance_collection.find_one({
                "student_id": sid,
                "exam_name": exam_name,
                "exam_date": exam_date.strftime("%Y-%m-%d")
            })

            if not already_marked:
                absent_record = {
                    "student_id": sid,
                    "name": student["name"],
                    "course": student["course"],
                    "year": student.get("year", "unknown"),
                    "exam_name": exam_name,
                    "exam_date": exam_date.strftime("%Y-%m-%d"),
                    "status": "Absent",
                    "timestamp": timestamp,
                    "date": timestamp.strftime("%Y-%m-%d")
                }
                attendance_collection.insert_one(absent_record)
                print(f"❌ Marked Absent: {student['name']}")

        # ✅ Final response
        student_info = {
            "_id": best_match['_id'],
            "student_id": student_id,
            "name": best_match["name"],
            "course": student_course,
            "year": student_year
        }

        return jsonify({
            "status": "success",
            "message": f"✅ Attendance marked for {best_match['name']} ({student_year} Year) in {exam_name} exam!",
            "student": student_info,
            "attendance_record": {
                k: (str(v) if isinstance(v, ObjectId) else v)
                for k, v in attendance_record.items()
            }
        })

    else:
        print("❌ No matching student found or match confidence too low.")
        return jsonify({"message": "❌ Face not recognized or confidence too low!"}), 404


# ✅ Get All Students
@app.route("/students", methods=["GET"])
def get_students():
    students = list(students_collection.find({}, {"_id": 0}))  # Exclude _id
    return jsonify(students)

@app.route('/attendance/summary', methods=['GET'])
def get_attendance_summary():
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Find all attendance entries with today's date
    today_records = list(attendance_collection.find({"date": today_str}))

    total_students = len(today_records)
    present_today = sum(1 for r in today_records if r.get("status", "").lower() == "present")
    absent_today = sum(1 for r in today_records if r.get("status", "").lower() == "absent")

    return jsonify({
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today
    })

@attendance_bp.route("/api/today-attendance-summary", methods=['GET'])
def today_attendance_summary():
    """Fetches attendance summary for today's date, optionally filtered by course and year."""
    course = request.args.get("course")
    year = request.args.get("year")  # ✅ Get year from query params

    today_str = datetime.now().strftime("%Y-%m-%d")
    query = {"date": today_str}

    if course:
        query["course"] = course
    if year:  # ✅ Add year filter to the query if present
        query["year"] = year

    records = list(attendance_collection.find(query, {'_id': 0}))
    print("📅 Today’s Attendance Fetched from DB:", records)
    return jsonify(records), 200

@app.route('/attendance_report')
def attendance_report_page():
    return render_template('attendance_report.html')

@app.route('/get_summary_data', methods=['GET'])
def get_summary_data_alias():
    return get_attendance_summary()

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

