import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import tempfile
import os

st.title("📊 Country Data Animation Generator")

uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx", "xls", "xlsm", "xlt", "xml", "xlsb"])
country_name = st.text_input("Enter Country Name")
generate_btn = st.button("Generate Animation")

if uploaded_file and country_name and generate_btn:
    # Determine file type and read accordingly
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls", "xlsm", "xlt", "xlsb"]:
            df = pd.read_excel(uploaded_file)
        elif file_extension == "xml":
            df = pd.read_xml(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            st.stop()
            
        st.success(f"✅ Successfully loaded {file_extension.upper()} file")
        
        # Check if required column exists
        if "Country Name" not in df.columns:
            st.error("The file doesn't contain a 'Country Name' column. Please upload a file with the correct format.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Display loading message
    with st.spinner(f"Processing data for {country_name}..."):
        if country_name not in df["Country Name"].values:
            st.error(f"Country '{country_name}' not found in the data file.")
            st.stop()
        
        # Process the country data
        country_row = df[df["Country Name"] == country_name].iloc[0]
        years = df.columns[4:]
        values = country_row[4:].astype(float).values

        data = pd.DataFrame({
            "Time": years.astype(int),
            "Value": values
        })

        x = data["Time"].values
        y = data["Value"].values

        norm = plt.Normalize(min(y), max(y))
        cmap = cm.plasma

        points_per_segment = 5
        x_smooth = np.linspace(x.min(), x.max(), len(x) * points_per_segment)
        y_smooth = np.interp(x_smooth, x, y)

        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], lw=2, color=cmap(0.5))
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y) - 1, max(y) + 1)
        ax.set_title(f"Smooth Animated Line Chart: {country_name}", fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(x_smooth[:i], y_smooth[:i])
            return line,

        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=len(x_smooth), interval=1, blit=True
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            anim.save(tmpfile.name, writer="ffmpeg", fps=30)
            st.success("✅ Animation Generated!")
            with open(tmpfile.name, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)  # Show the video on the Streamlit app
                st.download_button("Download Video", f, file_name="smooth_line_chart.mp4")
            os.remove(tmpfile.name)