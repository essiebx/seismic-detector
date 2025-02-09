import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from obspy import read
from scipy.signal import detrend

# Load pre-trained CNN model
MODEL_PATH = "quake_detector_model(proximity)-CNN.h5"
model = load_model(MODEL_PATH)

# --- App Title & Description ---
st.set_page_config(page_title="Seismic Event Detector", layout="wide")

st.title("🌍 Seismic Event Detector")

st.markdown(
    """
    ### 📡 Detect and Analyze Seismic Events in Real-Time  
    This application uses **Deep Learning** to analyze seismic data and **detect potential seismic events**.  
    Simply **upload your seismic data file** (`CSV` or `MiniSEED`), and the system will:
    
    ✅ **Visualize the seismic waveform** 📈  
    ✅ **Process the data using advanced signal processing** ⚙️  
    ✅ **Predict seismic activity using a trained CNN model** 🤖  
    ✅ **Provide insights on potential earthquake events** 🌍  

    ---
    
    ### ⚙️ Technologies Used:
    - **TensorFlow/Keras** - Deep learning model for seismic event prediction  
    - **Streamlit** - Interactive user interface  
    - **Matplotlib & Pandas** - Data visualization and processing  
    - **SciPy & ObsPy** - Seismic signal processing  

    **Upload your file below to get started! 🚀**
    """
)

# --- Instructions for Uploading Data ---
st.subheader("📂 Upload Your Seismic Data File")
st.markdown(
    """
    **Accepted File Types:**  
    - **CSV File** (`.csv`) - Must contain columns **'time_rel(sec)'** and **'amplitude'**  
    - **MiniSEED File** (`.mseed`) - Standard seismic format  
    - Ensure data is properly formatted before uploading.  
    """
)

uploaded_file = st.file_uploader("Choose a file", type=["csv", "mseed"])

# --- Process Uploaded File ---
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    time_series = None

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.write("✅ **CSV File Loaded Successfully:**", df.head())

            # Validate required columns
            if 'time_rel(sec)' in df.columns and 'amplitude' in df.columns:
                time_series = df[['time_rel(sec)', 'amplitude']].values
            else:
                st.error("❌ CSV file must contain 'time_rel(sec)' and 'amplitude' columns.")
                st.stop()

        elif file_type == "mseed":
            st.write("✅ **MiniSEED File Loaded Successfully.**")
            stream = read(uploaded_file)
            trace = stream[0]
            time_series = np.column_stack((np.arange(len(trace.data)), detrend(trace.data)))

        # --- Plot Seismic Waveform ---
        if time_series is not None:
            st.subheader("📊 Seismic Waveform")
            fig, ax = plt.subplots()
            ax.plot(time_series[:, 0], time_series[:, 1], color="blue")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Seismic Signal")
            st.pyplot(fig)

            # --- Prepare Data for Model Prediction ---
            time_steps = 100  # Define time step window
            sequences = []
            for i in range(len(time_series) - time_steps):
                sequences.append(time_series[i:i + time_steps])
            X_test = np.array(sequences)

            # --- Make Predictions ---
            st.subheader("📈 Seismic Event Predictions")
            predictions = model.predict(X_test)
            st.line_chart(predictions.flatten())

            # Display prediction results in table
            st.write("🔍 **Predicted Seismic Activity Probabilities:**")
            st.dataframe(pd.DataFrame(predictions, columns=["Seismic Event Probability"]))

    except Exception as e:
        st.error(f"⚠️ Failed to process the uploaded file. Ensure it contains valid seismic data.\n\n**Error:** {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Developed with ❤️ by esthernaisimoi </p>", unsafe_allow_html=True)