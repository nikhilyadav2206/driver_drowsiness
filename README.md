# 🚗 Driver Drowsiness Detection Using Machine Learning

A real-time driver drowsiness detection system built using **OpenCV**, **MediaPipe**, and **Python**.  
The system monitors the driver’s eyes using facial landmarks and triggers an alert if drowsiness is detected.

---

## 🧠 Project Overview
Driver fatigue is a major cause of road accidents.  
This project uses **EAR (Eye Aspect Ratio)** to detect prolonged eye closure and determines whether a driver is drowsy.

The system:
- Tracks eye landmarks  
- Computes EAR  
- Detects drowsiness  
- Gives a **voice alert** using pyttsx3  

---

## ⚙️ Features
- Real-time detection  
- Uses MediaPipe Face Mesh (468 facial landmarks)  
- EAR-based drowsiness logic  
- Works offline  
- Lightweight (runs on CPU)  
- Easy to modify and extend  

---
Driver-Drowsiness-Detection/
│
├── realtime.py # Main program
├── utils.py # EAR calculation helper
├── record_sample.py # Optional script for data collection
├── requirements.txt # Dependencies
├── images/ # Diagrams & screenshots (optional)
└── README.md # Documentation


---

## 💻 Installation

### 1. Clone this repo
```bash
git clone https://github.com/nikhilyadav2206/driver-drowsiness-detection.git
cd driver-drowsiness-detection
//creating a virtual enironment would be more suitable 
2. Install dependencies
pip install -r requirements.txt

3. Run the project
python realtime.py
//creating a virtual enironment would be more suitable


🧮 How It Works

Webcam captures video

MediaPipe detects face and eye landmarks

EAR (Eye Aspect Ratio) is computed

If EAR < threshold (0.21) for multiple frames → Drowsy

System triggers visual + voice alert


## 🗂️ Project Structure
