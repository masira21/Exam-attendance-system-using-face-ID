import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_attendance"]
students_collection = db["students"]

# Function to find the best matching student
def find_matching_student(face_encoding):
    students = list(students_collection.find())  # Fetch all students from DB
    best_match = None
    best_distance = 0.6  # Lower means more accurate match

    for student in students:
        stored_encoding = np.array(student["face_encoding"])
        distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
        
        print(f"🔍 Checking {student['name']} - Distance: {distance}")  # Debugging

        if distance < best_distance:  # Closer distance = better match
            best_distance = distance
            best_match = student

    if best_match:
        print(f"✅ Match Found: {best_match['name']} (Distance: {best_distance})")
        return best_match
    else:
        print("❌ No Match Found")
        return None

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("🔍 Looking for a face... Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error accessing webcam.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matched_student = find_matching_student(face_encoding)  # Use the improved function

        if matched_student:
            print(f"✅ Student Identified: {matched_student['name']} (ID: {matched_student['student_id']})")
            
            # Mark attendance
            attendance_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Present"
            }
            students_collection.update_one(
                {"student_id": matched_student["student_id"]},
                {"$push": {"attendance": attendance_entry}}
            )
            
            print("📝 Attendance marked successfully!")
            video_capture.release()
            cv2.destroyAllWindows()
            exit()

    # Show webcam feed
    cv2.imshow("Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
