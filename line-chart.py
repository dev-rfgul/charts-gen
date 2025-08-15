import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import tempfile
import os

st.title("📊 Country Data Animation Generator")

uploaded_file = st.file_uploader("Upload CSV", type=["csv","xlx","xlsx"])
country_name = st.text_input("Enter Country Name")
generate_btn = st.button("Generate Animation")

if uploaded_file and country_name and generate_btn:
    df = pd.read_csv(uploaded_file)

    if country_name not in df["Country Name"].values:
        st.error(f"Country '{country_name}' not found in CSV.")
    else:
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