import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
import tempfile
import os
import re
import subprocess
import sys

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Check for video writers
def check_video_writers():
    """Check available video writers and return the best one"""
    writers = {}
    
    # Check for FFmpeg
    try:
        subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        writers['ffmpeg'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        writers['ffmpeg'] = False
    
    # Check for imageio-ffmpeg (fallback)
    try:
        import imageio_ffmpeg
        writers['imageio-ffmpeg'] = True
    except ImportError:
        writers['imageio-ffmpeg'] = False
    
    # Pillow writer (GIF fallback - always available)
    writers['pillow'] = True
    
    return writers

# Global variable to store available writers
AVAILABLE_WRITERS = check_video_writers()

st.title("📊 Time Series Data Animation Generator")
st.write("Upload your data file and create animated charts showing how values change over time!")

# Display system status
with st.expander("🔧 System Status", expanded=False):
    if AVAILABLE_WRITERS['ffmpeg']:
        st.success("✅ FFmpeg available - MP4 videos supported")
    elif AVAILABLE_WRITERS['imageio-ffmpeg']:
        st.info("⚡ ImageIO-FFmpeg available - MP4 videos supported")
    else:
        st.warning("⚠️ No MP4 support - animations will be saved as GIF files")
    
    st.write("Available formats:")
    formats = []
    if AVAILABLE_WRITERS['ffmpeg'] or AVAILABLE_WRITERS['imageio-ffmpeg']:
        formats.append("MP4")
    formats.append("GIF")
    st.write(f"• {', '.join(formats)}")

uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx", "xls", "xlsm", "xlt", "xml", "xlsb"])

def load_data(uploaded_file):
    """Load data from various file formats"""
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
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def detect_structure(df):
    """Detect the structure of the uploaded data"""
    structure = {
        'entity_column': None,
        'time_columns': [],
        'metadata_columns': [],
        'entities': []
    }
    
    # Find entity column (usually first column with country/entity names)
    possible_entity_cols = []
    for col in df.columns[:5]:  # Check first 5 columns
        if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.7:
            possible_entity_cols.append(col)
    
    if possible_entity_cols:
        structure['entity_column'] = possible_entity_cols[0]
    else:
        structure['entity_column'] = df.columns[0]  # Fallback to first column
    
    # Get unique entities
    structure['entities'] = sorted(df[structure['entity_column']].dropna().unique().tolist())
    
    # Detect time columns (years, dates, or numeric columns that could represent time)
    for col in df.columns:
        col_str = str(col)
        
        # Skip entity column
        if col == structure['entity_column']:
            continue
        
        # Check if it's a year (4 digits between 1900-2100)
        if col_str.isdigit() and len(col_str) == 4:
            year = int(col_str)
            if 1900 <= year <= 2100:
                structure['time_columns'].append(col)
        
        # Check if it looks like a date
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', col_str) or re.match(r'^\d{2}/\d{2}/\d{4}$', col_str):
            structure['time_columns'].append(col)
        
        # Check if column name suggests time
        elif any(word in col_str.lower() for word in ['year', 'date', 'time', 'period', 'quarter', 'month']):
            structure['time_columns'].append(col)
        
        # Check if it's a numeric column that could be time-based
        elif col_str.replace('.', '').replace('-', '').isdigit():
            try:
                num_val = float(col_str)
                if 1900 <= num_val <= 2100:  # Reasonable year range
                    structure['time_columns'].append(col)
            except:
                pass
        
        # If not a time column, it might be metadata
        else:
            structure['metadata_columns'].append(col)
    
    # Sort time columns
    structure['time_columns'] = sorted(structure['time_columns'], 
                                    key=lambda x: float(str(x)) if str(x).replace('.', '').replace('-', '').isdigit() else 0)
    
    return structure

def prepare_time_series_data(df, entity, entity_col, time_cols):
    """Prepare time series data for a specific entity"""
    # Get the row for the specific entity
    entity_data = df[df[entity_col] == entity]
    
    if entity_data.empty:
        return None
    
    entity_row = entity_data.iloc[0]
    
    # Extract time series data
    time_values = []
    data_values = []
    
    for time_col in time_cols:
        try:
            value = pd.to_numeric(entity_row[time_col], errors='coerce')
            if not pd.isna(value):
                # Convert time column to numeric if possible
                if str(time_col).isdigit():
                    time_val = int(str(time_col))
                else:
                    try:
                        time_val = float(str(time_col))
                    except:
                        time_val = len(time_values)  # Use index as fallback
                
                time_values.append(time_val)
                data_values.append(value)
        except:
            continue
    
    if not time_values:
        return None
    
    return pd.DataFrame({
        'Time': time_values,
        'Value': data_values
    }).sort_values('Time')

def create_animated_line_chart(time_series_data, entity_name, chart_title="Time Series Animation"):
    """Create animated line chart for time series data"""
    x = time_series_data['Time'].values
    y = time_series_data['Value'].values
    
    # Create smooth interpolation for better animation
    points_per_segment = 5
    x_smooth = np.linspace(x.min(), x.max(), len(x) * points_per_segment)
    y_smooth = np.interp(x_smooth, x, y)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create line and point objects
    line, = ax.plot([], [], lw=3, color='#2E86AB', alpha=0.8)
    point, = ax.plot([], [], 'o', color='#F18F01', markersize=8, zorder=5)
    
    # Set axis limits with padding
    x_padding = (x.max() - x.min()) * 0.05 if len(x) > 1 else 1
    y_padding = (y.max() - y.min()) * 0.1 if y.max() != y.min() else abs(y.max()) * 0.1 if y.max() != 0 else 1
    
    ax.set_xlim(x.min() - x_padding, x.max() + x_padding)
    ax.set_ylim(y.min() - y_padding, y.max() + y_padding)
    
    # Styling
    ax.set_title(f"{chart_title}\n{entity_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Value", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value annotation
    value_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Capture axis/text handles so we can fade them in
    title_text = ax.title
    xlabel_text = ax.xaxis.label
    ylabel_text = ax.yaxis.label
    x_tick_labels = ax.get_xticklabels()
    y_tick_labels = ax.get_yticklabels()
    gridlines = ax.yaxis.get_gridlines() + ax.xaxis.get_gridlines()

    # Initialize axis artists to invisible
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_alpha(0.0)
    title_text.set_alpha(0.0)
    xlabel_text.set_alpha(0.0)
    ylabel_text.set_alpha(0.0)
    for tl in x_tick_labels + y_tick_labels:
        tl.set_alpha(0.0)
    for g in gridlines:
        g.set_alpha(0.0)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        value_text.set_text('')
        return line, point, value_text

    def animate(i):
        total = max(1, len(x_smooth))
        progress = (i + 1) / total

        if i < len(x_smooth):
            # Update line and point
            line.set_data(x_smooth[:i+1], y_smooth[:i+1])
            current_x = x_smooth[i]
            current_y = y_smooth[i]
            point.set_data([current_x], [current_y])
            value_text.set_text(f'Time: {current_x:.1f}\nValue: {current_y:.2f}')

        # Fade axis elements in proportionally to progress
        alpha = min(1.0, progress * 1.2)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_alpha(alpha)
        title_text.set_alpha(alpha)
        xlabel_text.set_alpha(alpha)
        ylabel_text.set_alpha(alpha)
        for tl in x_tick_labels + y_tick_labels:
            tl.set_alpha(alpha)
        for g in gridlines:
            g.set_alpha(alpha * 0.3)

        return line, point, value_text

    # Disable blitting so axis artist updates are rendered properly
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_smooth), 
                        interval=100, blit=False, repeat=True)
    
    return anim, fig

def create_animated_bar_race(df, entity_col, time_cols, top_n=10):
    """Create animated bar race showing top entities over time"""
    
    # Prepare data for animation
    frames_data = []
    
    for time_col in time_cols[:20]:  # Limit to first 20 time periods for performance
        # Get data for this time period
        time_data = df[[entity_col, time_col]].copy()
        time_data.columns = ['Entity', 'Value']
        time_data['Time'] = str(time_col)
        time_data['Value'] = pd.to_numeric(time_data['Value'], errors='coerce')
        time_data = time_data.dropna()
        
        # Get top N entities for this time period
        time_data = time_data.nlargest(top_n, 'Value')
        frames_data.append(time_data)
    
    if not frames_data:
        return None, None
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate_bars(frame_idx):
        ax.clear()

        # compute progress for fade-in (0..1)
        total = max(1, len(frames_data))
        progress = (frame_idx + 1) / total

        if frame_idx < len(frames_data):
            frame_data = frames_data[frame_idx]
            time_period = frame_data['Time'].iloc[0]
            
            # Sort by value for this frame
            frame_data = frame_data.sort_values('Value', ascending=True)
            
            # Create colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(frame_data)))
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(frame_data)), frame_data['Value'], color=colors)
            
            # Set labels
            ax.set_yticks(range(len(frame_data)))
            ax.set_yticklabels(frame_data['Entity'])
            ax.set_xlabel('Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {top_n} Entities - Time Period: {time_period}', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, frame_data['Value'])):
                ax.text(value + max(frame_data['Value']) * 0.01, i, f'{value:.1f}', 
                       va='center', fontweight='bold')
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3)
            
            # Set consistent x-axis limits
            all_max = max([df['Value'].max() for df in frames_data if not df.empty])
            ax.set_xlim(0, all_max * 1.1)

            # Fade in left spine, title, and tick labels as progress increases
            alpha = min(1.0, progress * 1.5)
            ax.spines['left'].set_alpha(alpha)
            ax.title.set_alpha(alpha)
            ax.xaxis.label.set_alpha(alpha)
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_alpha(alpha)
        
        plt.tight_layout()
    
    anim = FuncAnimation(fig, animate_bars, frames=len(frames_data), 
                        interval=800, repeat=True, blit=False)
    
    return anim, fig

# Main App Logic
if uploaded_file:
    # Load the data
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
        
        # Detect data structure
        structure = detect_structure(df)
        
        # Show data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
        
        # Show detected structure
        st.subheader("🔍 Detected Data Structure")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Entity Column:** {structure['entity_column']}")
            st.write(f"**Number of Entities:** {len(structure['entities'])}")
        
        with col2:
            st.write(f"**Time Columns:** {len(structure['time_columns'])}")
            if structure['time_columns']:
                time_range = f"{structure['time_columns'][0]} - {structure['time_columns'][-1]}"
                st.write(f"**Time Range:** {time_range}")
        
        with col3:
            st.write(f"**Metadata Columns:** {len(structure['metadata_columns'])}")
        
        if not structure['time_columns']:
            st.error("❌ No time columns detected! Please make sure your data has year/date columns.")
        elif not structure['entities']:
            st.error("❌ No entities detected! Please check your data format.")
        else:
            # Configuration options
            st.subheader("⚙️ Chart Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Individual Entity Time Series", "Bar Race (Top Entities)"],
                    help="Choose the type of animation you want to create"
                )
            
            with col2:
                if chart_type == "Individual Entity Time Series":
                    selected_entity = st.selectbox(
                        f"Select {structure['entity_column']}:",
                        structure['entities'],
                        help="Choose which entity to animate"
                    )
                else:
                    top_n = st.slider(
                        "Number of top entities to show:",
                        min_value=5, max_value=20, value=10,
                        help="How many top entities to display in the bar race"
                    )
            
            # Generate chart button
            if st.button("🎬 Generate Animation", type="primary"):
                
                with st.spinner("Creating your animated chart..."):
                    
                    try:
                        if chart_type == "Individual Entity Time Series":
                            # Create time series for selected entity
                            time_series_data = prepare_time_series_data(
                                df, selected_entity, structure['entity_column'], structure['time_columns']
                            )
                            
                            if time_series_data is None or len(time_series_data) < 2:
                                st.error("❌ Not enough data points for the selected entity.")
                            else:
                                anim, fig = create_animated_line_chart(
                                    time_series_data, selected_entity, "Time Series Animation"
                                )
                                
                                filename = f"timeseries_{selected_entity.replace(' ', '_')}.mp4"
                        
                        else:  # Bar Race
                            anim, fig = create_animated_bar_race(
                                df, structure['entity_column'], structure['time_columns'], top_n
                            )
                            
                            if anim is None:
                                st.error("❌ Unable to create bar race. Please check your data.")
                                st.stop()
                            
                            filename = f"bar_race_top_{top_n}.mp4"
                        
                        # Save and display animation
                        success = False
                        video_bytes = None
                        file_extension = None
                        
                        # Try different writers in order of preference
                        if AVAILABLE_WRITERS['ffmpeg']:
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                                    anim.save(tmpfile.name, writer="ffmpeg", fps=20, bitrate=2000)
                                    with open(tmpfile.name, "rb") as f:
                                        video_bytes = f.read()
                                    os.unlink(tmpfile.name)
                                    file_extension = ".mp4"
                                    success = True
                            except Exception as e:
                                st.warning(f"FFmpeg failed: {str(e)}. Trying alternative...")
                        
                        # Fallback to imageio-ffmpeg
                        if not success and AVAILABLE_WRITERS['imageio-ffmpeg']:
                            try:
                                import imageio
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                                    anim.save(tmpfile.name, writer="ffmpeg", fps=20)
                                    with open(tmpfile.name, "rb") as f:
                                        video_bytes = f.read()
                                    os.unlink(tmpfile.name)
                                    file_extension = ".mp4"
                                    success = True
                            except Exception as e:
                                st.warning(f"ImageIO-FFmpeg failed: {str(e)}. Trying GIF...")
                        
                        # Final fallback to GIF using Pillow
                        if not success:
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
                                    writer = PillowWriter(fps=10)
                                    anim.save(tmpfile.name, writer=writer)
                                    with open(tmpfile.name, "rb") as f:
                                        video_bytes = f.read()
                                    os.unlink(tmpfile.name)
                                    file_extension = ".gif"
                                    filename = filename.replace('.mp4', '.gif')
                                    success = True
                                    st.info("� Generated as GIF since MP4 is not available in this environment")
                            except Exception as e:
                                st.error(f"All video generation methods failed: {str(e)}")
                                success = False
                        
                        if success and video_bytes:
                            st.success("✅ Animation Generated Successfully!")
                            
                            # Display video/gif
                            if file_extension == ".mp4":
                                st.video(video_bytes)
                                mime_type = "video/mp4"
                            else:  # .gif
                                st.image(video_bytes)
                                mime_type = "image/gif"
                            
                            # Download button
                            st.download_button(
                                "📥 Download Animation",
                                video_bytes,
                                file_name=filename,
                                mime=mime_type
                            )
                        else:
                            st.error("❌ Failed to generate animation. Please try with different data or contact support.")
                            plt.close(fig)
                    
                    except Exception as e:
                        st.error(f"❌ Error generating animation: {str(e)}")
                        st.write("Please check your data format and try again.")

else:
    st.info("👆 Upload your data file to get started!")
    
    st.subheader("📄 Expected Data Format")
    st.write("""
    Your data should have:
    - **First column**: Entity names (Countries, Products, Students, etc.)
    - **Year columns**: 1970, 1971, 1972, etc. (or any time periods)
    - **Values**: Numeric data for each entity and time period
    """)
    
    # Show example
    example_data = {
        'Country Name': ['Afghanistan', 'Albania', 'Algeria'],
        'Region': ['Asia', 'Europe', 'Africa'], 
        'Image URL': ['url1', 'url2', 'url3'],
        '1970': [26.25, 59.375, 30],
        '1971': [26.25, 59.375, 30],
        '1972': [26.25, 59.375, 30],
        '1973': [26.25, 59.375, 30]
    }
    
    st.subheader("📊 Example Data Structure")
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df)
    
    st.write("""
    **Supported Chart Types:**
    - 📈 **Individual Entity Time Series**: Shows how one entity's values change over time
    - 🏁 **Bar Race**: Animated ranking showing top entities competing over time
    """)