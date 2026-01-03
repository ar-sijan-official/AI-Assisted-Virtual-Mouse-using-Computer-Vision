#  AI-Assisted Virtual Mouse using Computer Vision


A real-time **vision-controlled virtual mouse** that enables touch-free interaction with a computer using **hand gestures captured via a webcam**.  
The system leverages **ML-based hand landmark detection** combined with classical computer vision and geometric mapping.

---

## 📌 Description

This project implements an **AI-assisted virtual mouse** that allows users to control the system cursor without physical input devices.  
Using a standard webcam, the system tracks hand movements and interprets gestures to perform cursor movement and mouse clicks in real time.

The hand-tracking component is powered by a **pre-trained deep learning model (MediaPipe Hands)**, while cursor control and gesture logic are implemented using deterministic rules.

---

## 🚀 Features

- Real-time hand landmark detection
- Smooth cursor movement using index finger
- Gesture-based mouse click (finger distance)
- Touch-free human–computer interaction
- FPS display for performance monitoring
- No external hardware required

---

## 🧠 How It Works

1. Captures live video using OpenCV  
2. Detects and tracks 21 hand landmarks using MediaPipe (ML-based)  
3. Interprets finger positions to recognize gestures  
4. Maps hand coordinates to screen space  
5. Controls mouse movement and clicks using AutoPy  

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – real-time video processing
- **MediaPipe** – ML-based hand landmark detection
- **NumPy** – numerical operations
- **AutoPy** – system-level mouse control
- **Math / Time** – gesture distance & smoothing logic

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ar-sijan-official/AI-Assisted-Virtual-Mouse-using-Computer-Vision.git
cd AI-Assisted-Virtual-Mouse-using-Computer-Vision
