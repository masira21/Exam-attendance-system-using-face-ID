import cv2
import numpy as np
import face_recognition
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["exam_attendance"]
students_collection = db["students"]

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error accessing webcam")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        print(f"✅ Face detected!")

        # Extract face encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        print(f"🔢 Encoding Vector (first 5 values): {face_encoding[:5]}")

        # Student details (For testing, you can replace with input values)
        student_id = "S001"
        name = "Masira Furniturewala"
        course = "BCA"

        # Check if student exists
        student_data = students_collection.find_one({"student_id": student_id})

        if student_data:
            # Update existing student's face encoding
            students_collection.update_one(
                {"student_id": student_id},
                {"$set": {"face_encoding": face_encoding.tolist()}}
            )
            print(f"✅ Updated face encoding for {name}")
        else:
            # Insert new student data
            new_student = {
                "student_id": student_id,
                "name": name,
                "course": course,
                "face_encoding": face_encoding.tolist(),
                "attendance": []
            }
            students_collection.insert_one(new_student)
            print(f"✅ New student added: {name}")

        break  # Exit loop after capturing face encoding

    # Show camera feed
    cv2.imshow("Face Capture", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
