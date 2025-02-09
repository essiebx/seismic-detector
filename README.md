# Seismic Detector App 🌍⚡

This is a **Seismic Detector App** that predicts earthquake occurrences based on sensor data. It utilizes a Convolutional Neural Network (CNN) model to analyze proximity-based seismic signals.

This project is inspired by the **[NASA Space Apps Challenge 2024](https://www.spaceappschallenge.org/nasa-space-apps-2024/challenges/)**, where I participated in developing innovative GIS solutions for space and Earth-related challenges.  
## 🚀 Live Demo
You can access the live demo of this app here:  
👉 [Seismic Detector App](https://essiebx-seismic-detector-app-y8yl7w.streamlit.app/)

## 🛠 Features
- 📡 **Real-time Seismic Data Analysis**
- 🔍 **Proximity-Based Earthquake Detection**
- 📊 **Interactive Visualizations**
- 🧠 **Deep Learning Model Powered by CNN**

## 🏗️ How to Use
1. Upload seismic sensor data.
2. The model will process the input and predict earthquake occurrence.
3. Visualizations will help interpret the results.

## 🖥️ Installation & Setup
To run the app locally:
```bash
git clone https://github.com/essiebx/seismic-detector.git
cd seismic-detector
pip install -r requirements.txt
streamlit run app.py

## Model Details
Model: CNN-based classifier
Framework: TensorFlow/Keras
File: quake_detector_model(proximity)-CNN.h5

## Folder Structure
seismic-detector/
│── app.py          # Main Streamlit app
│── model/          # Pre-trained model files
│── requirements.txt # Dependencies
│── README.md       # Project documentation
