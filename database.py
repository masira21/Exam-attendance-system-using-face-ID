from pymongo import MongoClient
from config import config

# Connect to MongoDB
client = MongoClient(config.MONGO_URI)
db = client["exam_attendance"]

# Define collections
students_collection = db["students"]
attendance_collection = db["attendance"]
exams_collection =db["exams"]

#create a global mongodb client
client = MongoClient("mongodb://localhost:27017/")
db = client["exam_attendance"]

def get_database():
    return db #always return the existing connection

print("Connected to MongoDB successfully!")
