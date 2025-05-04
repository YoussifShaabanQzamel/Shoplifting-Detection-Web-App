# 🚨 Shoplifting Detection Web App

This project is an advanced **real-time shoplifting detection system** that uses a custom 3D ConvNet and **YOLO** (You Only Look Once) for precise detection. It is designed to detect theft in retail environments with high accuracy and dynamic real-time video analysis. The system integrates **optical flow tracking**, **sliding window detection**, and **dynamic bounding boxes** for accurate suspect tracking. The user interface is built with **Django** to provide a seamless and interactive web experience for security teams.

---

## 🚀 Live Demo

🔗 [Watch the demo](https://drive.google.com/file/d/1ZZJBUl4qT5qln9z5PlSmDeCaNhoo3Roc/view?usp=sharing)

---

## 🧠 Key Features

- ✅ **Real-time Shoplifting Detection**: High-accuracy detection with **98%+ accuracy** and **0.99 AUC**.
- ✅ **Dynamic Bounding Boxes**: Visual representation of suspects with confidence scores.
- ✅ **Optical Flow Tracking**: Monitors suspects' movements across video frames for continuous tracking.
- ✅ **Sliding Window Detection**: Ensures precise detection by analyzing frames at different scales.
- ✅ **Django-Based UI**: Interactive interface for security teams to review and track detected incidents.

---

## 🛠️ Tech Stack

| Component        | Tools/Frameworks              |
|------------------|-------------------------------|
| Language         | Python                        |
| Deep Learning    | TensorFlow, Keras             |
| Computer Vision  | OpenCV                        |
| Web Framework    | Django                        |
| Frontend         | HTML, CSS, Bootstrap, JavaScript |
| Backend          | Django                        |
| Data             | Custom Shoplifting Dataset    |

---

## Getting Started
### 1. Clone the repository
```
git clone https://github.com/YoussifShaabanQzamel/Shoplifting-Detection-Web-App.git
cd Shoplifting-Detection-Web-App
```

### 2. Create a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt

```
### 4. Run the Django app
```
python manage.py runserver
```

## 🌐 Web App Functionality

- **Upload video**: Security personnel upload video footage for shoplifting detection.
- **Real-time detection**: The backend processes the video, detecting shoplifting events in real-time.
- **Review results**: Interactive UI displays frames with bounding boxes around detected individuals.
- **Download results**: Export video frames with overlayed detection results.

---

## 📌 Notes

- The system uses **YOLO** for fast and accurate object detection and a custom **3D ConvNet** for deeper feature extraction.
- The web app was designed using **Django** for a dynamic and responsive interface.
- Ensure that video files are in the correct format and resolution for optimal detection performance.

---
