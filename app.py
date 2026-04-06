import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import rasterio

# Title
st.title("🌾 AI Crop Stress Detection System")

# Description
st.markdown("""
### 🌱 Smart Crop Health Monitoring

This AI system analyzes multispectral drone imagery to detect crop stress.
It uses vegetation indices and machine learning to classify crop health.

**Upload a .tif image to begin analysis.**
""")

# Upload file
uploaded_file = st.file_uploader("Upload Multispectral Image (.tif)")

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.info("Processing image...")

    # Load model
    model = joblib.load("crop_health_model.pkl")

    # Read image
    with rasterio.open(image_path) as src:
        green = src.read(2).astype(float)
        red = src.read(3).astype(float)
        red_edge = src.read(4).astype(float)
        nir = src.read(5).astype(float)

    # Compute indices
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndre = (nir - red_edge) / (nir + red_edge + 1e-10)
    gndvi = (nir - green) / (nir + green + 1e-10)
    savi = ((nir - red) / (nir + red + 0.5)) * 1.5

    # Prepare features
    X = np.column_stack((
        ndvi.flatten(),
        ndre.flatten(),
        gndvi.flatten(),
        savi.flatten()
    ))

    # Predict
    predictions = model.predict(X)
    prediction_map = predictions.reshape(ndvi.shape)

    # Legend
    st.subheader("🧾 Class Legend")
    st.write("""
    0 → 🔴 Very Stressed  
    1 → 🟠 Stressed  
    2 → 🟡 Moderate  
    3 → 🟢 Healthy  
    """)

    # Show Map
    st.subheader("🗺️ Crop Stress Map")

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(prediction_map, cmap="jet")
    plt.colorbar(im, ax=ax)
    ax.set_title("Crop Stress Map")

    st.pyplot(fig)

    # Show Percentages
    st.subheader("📊 Crop Health Summary")

    labels = {
        0: "Very Stressed",
        1: "Stressed",
        2: "Moderate",
        3: "Healthy"
    }

    total_pixels = prediction_map.size
    results = {}

    for i in range(4):
        percent = np.sum(prediction_map == i) / total_pixels * 100
        results[labels[i]] = percent

    # Display metrics
    for label, value in results.items():
        st.metric(label, f"{value:.2f}%")

    # Smart Insights
    st.subheader("📢 Insights")

    if results["Very Stressed"] > 30:
        st.error("⚠️ High crop stress detected! Immediate attention required.")

    elif results["Stressed"] > 40:
        st.warning("⚠️ Moderate stress detected. Monitor crop health.")

    else:
        st.success("✅ Crop health is generally good.")

    st.success("✅ Analysis Complete")