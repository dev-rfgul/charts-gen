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
chart_type = st.selectbox("Select Chart Type", ["Line", "Bar","Scatter","Pie"])
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
        
        if chart_type == "Line":
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
            
        elif chart_type == "Bar":
            ax.set_xlim(min(x)-1, max(x)+1)
            ax.set_ylim(0, max(y) * 1.1)
            ax.set_title(f"Animated Bar Chart: {country_name}", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.5, axis='y')
            
            bar_container = ax.bar([], [], color=cmap(0.5), alpha=0.8)
            
            def init():
                for bar in bar_container:
                    bar.set_height(0)
                return bar_container
                
            def animate(i):
                # Add one bar at a time
                n_bars = min(i+1, len(x))
                for j, bar in enumerate(bar_container[:n_bars]):
                    bar.set_x(x[j] - 0.4)
                    bar.set_width(0.8)
                    bar.set_height(y[j])
                return bar_container
                
            # Create initial bars (will be updated in animation)
            bar_container = ax.bar(x, [0] * len(x), color=cmap(0.5), alpha=0.8)
            
            anim = FuncAnimation(
                fig, animate, init_func=init,
                frames=len(x)+5, interval=200, blit=True
            )
            
        elif chart_type == "Scatter":
            ax.set_xlim(min(x), max(x))
            ax.set_ylim(min(y) - 1, max(y) + 1)
            ax.set_title(f"Animated Scatter Plot: {country_name}", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            scatter = ax.scatter([], [], s=100, c=[], cmap=cmap, alpha=0.8)
            
            def init():
                scatter.set_offsets(np.empty((0, 2)))
                return scatter,
                
            def animate(i):
                data = np.column_stack((x[:i+1], y[:i+1]))
                scatter.set_offsets(data)
                scatter.set_array(np.linspace(0, 1, i+1))
                return scatter,
                
            anim = FuncAnimation(
                fig, animate, init_func=init,
                frames=len(x), interval=200, blit=True
            )
            
        elif chart_type == "Pie":
            # Pie charts don't animate well with FuncAnimation
            # Instead, create a static pie chart
            ax.clear()
            # Use periods/decades as labels instead of specific years
            if len(x) > 6:
                # Group by decades or periods for more meaningful pie chart
                periods = [f"{x[i]}-{x[i+len(x)//5]}" for i in range(0, len(x), len(x)//5)]
                period_values = [sum(y[i:i+len(x)//5]) for i in range(0, len(x), len(x)//5)]
                
                # Ensure positive values for pie chart
                period_values = [max(0, val) for val in period_values]
                total = sum(period_values)
                if total <= 0:
                    period_values = [1] * len(periods)  # Fallback if all values are negative
                
                explode = [0.1] * len(periods)  # Explode all slices
                ax.pie(period_values, labels=periods, explode=explode, autopct='%1.1f%%', 
                      shadow=True, startangle=90, colors=[cmap(i/len(periods)) for i in range(len(periods))])
            else:
                # If few data points, use them directly
                explode = [0.1] * len(x)
                ax.pie(np.maximum(y, 0), labels=[str(year) for year in x], explode=explode, 
                      autopct='%1.1f%%', shadow=True, startangle=90, 
                      colors=[cmap(i/len(x)) for i in range(len(x))])
                
            ax.set_title(f"Pie Chart: {country_name} Data Distribution", fontsize=14)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # No animation for pie chart
            anim = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4" if anim else ".png") as tmpfile:
            if anim:  # For animated charts (Line, Bar, Scatter)
                anim.save(tmpfile.name, writer="ffmpeg", fps=30)
                st.success(f"✅ {chart_type} Animation Generated!")
                with open(tmpfile.name, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes)  # Show the video on the Streamlit app
                    st.download_button("Download Video", f, file_name=f"{chart_type.lower()}_chart_{country_name}.mp4")
            else:  # For static charts (Pie)
                plt.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                st.success(f"✅ {chart_type} Chart Generated!")
                with open(tmpfile.name, "rb") as f:
                    image_bytes = f.read()
                    st.image(image_bytes, caption=f"{chart_type} Chart for {country_name}")
                    st.download_button("Download Image", f, file_name=f"{chart_type.lower()}_chart_{country_name}.png")
            os.remove(tmpfile.name)
