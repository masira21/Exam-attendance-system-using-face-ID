import os

class Config:
    SECRET_KEY = os.getenv("masira", "masira")  # Change this
    MONGO_URI = "mongodb://localhost:27017/exam_attendance"

config = Config()
