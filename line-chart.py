import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import tempfile
import os
from datetime import datetime
import re

st.title("📊 Dynamic Data Animation Generator")
st.write("Upload any CSV/Excel file and create animated charts from your data!")

uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx", "xls", "xlsm", "xlt", "xml", "xlsb"])

def detect_data_structure(df):
    """Automatically detect the structure of the uploaded data"""
    structure_info = {
        'entity_column': None,
        'time_columns': [],
        'value_columns': [],
        'data_type': 'unknown',
        'possible_entities': [],
        'time_format': None
    }
    
    # Look for common entity column names
    entity_patterns = [
        r'.*country.*', r'.*name.*', r'.*student.*', r'.*employee.*',
        r'.*product.*', r'.*item.*', r'.*entity.*', r'.*id.*'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in entity_patterns:
            if re.match(pattern, col_lower):
                structure_info['entity_column'] = col
                break
        if structure_info['entity_column']:
            break
    
    # If no entity column found, use first column
    if not structure_info['entity_column']:
        structure_info['entity_column'] = df.columns[0]
    
    # Get possible entities
    if structure_info['entity_column'] in df.columns:
        structure_info['possible_entities'] = df[structure_info['entity_column']].unique().tolist()
    
    # Detect time/year columns and value columns
    for col in df.columns:
        if col == structure_info['entity_column']:
            continue
            
        # Check if column name looks like a year or date
        if col.isdigit() and len(col) == 4 and 1900 <= int(col) <= 2100:
            structure_info['time_columns'].append(col)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', str(col)):  # Date format YYYY-MM-DD
            structure_info['time_columns'].append(col)
        elif re.match(r'.*year.*|.*date.*|.*time.*|.*period.*', col.lower()):
            structure_info['time_columns'].append(col)
        else:
            # Check if column contains numeric data
            try:
                pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():  # If conversion was successful for some values
                    structure_info['value_columns'].append(col)
            except:
                pass
    
    # Determine data type based on entity column name
    entity_col_lower = structure_info['entity_column'].lower()
    if 'country' in entity_col_lower:
        structure_info['data_type'] = 'country'
    elif any(word in entity_col_lower for word in ['student', 'pupil', 'learner']):
        structure_info['data_type'] = 'student'
    elif any(word in entity_col_lower for word in ['employee', 'worker', 'staff']):
        structure_info['data_type'] = 'employee'
    elif any(word in entity_col_lower for word in ['product', 'item', 'goods']):
        structure_info['data_type'] = 'product'
    elif any(word in entity_col_lower for word in ['sales', 'revenue', 'income']):
        structure_info['data_type'] = 'sales'
    else:
        structure_info['data_type'] = 'general'
    
    return structure_info

def create_time_series_data(df, entity_name, entity_column, time_columns):
    """Create time series data from wide format"""
    entity_row = df[df[entity_column] == entity_name].iloc[0]
    
    time_data = []
    value_data = []
    
    for time_col in time_columns:
        try:
            value = pd.to_numeric(entity_row[time_col], errors='coerce')
            if not pd.isna(value):
                # Try to convert time column to integer year
                if time_col.isdigit():
                    time_data.append(int(time_col))
                else:
                    time_data.append(len(time_data))  # Use index if can't parse
                value_data.append(value)
        except:
            continue
    
    return pd.DataFrame({
        "Time": time_data,
        "Value": value_data
    })

def create_value_comparison_data(df, entity_column, value_columns, selected_entity=None):
    """Create data for comparing multiple values for an entity"""
    if selected_entity:
        entity_data = df[df[entity_column] == selected_entity].iloc[0]
        categories = []
        values = []
        
        for col in value_columns:
            try:
                value = pd.to_numeric(entity_data[col], errors='coerce')
                if not pd.isna(value):
                    categories.append(col)
                    values.append(value)
            except:
                continue
        
        return pd.DataFrame({
            "Category": categories,
            "Value": values
        })
    return None

if uploaded_file:
    # Load file
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
        
        # Detect data structure
        structure = detect_data_structure(df)
        
        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())
        
        # Display detected structure
        st.subheader("🔍 Detected Data Structure")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Data Type:** {structure['data_type'].title()}")
            st.write(f"**Entity Column:** {structure['entity_column']}")
            st.write(f"**Number of Entities:** {len(structure['possible_entities'])}")
        
        with col2:
            st.write(f"**Time Columns:** {len(structure['time_columns'])}")
            st.write(f"**Value Columns:** {len(structure['value_columns'])}")
        
        # Allow user to select entity and chart type
        st.subheader("⚙️ Configuration")
        
        selected_entity = st.selectbox(
            f"Select {structure['data_type'].title()}:",
            structure['possible_entities']
        )
        
        chart_type = st.radio(
            "Chart Type:",
            ["Time Series Animation", "Value Comparison Animation"]
        )
        
        # Additional options based on chart type
        if chart_type == "Time Series Animation" and len(structure['time_columns']) > 0:
            st.write("Time series data detected - will animate values over time")
        elif chart_type == "Value Comparison Animation" and len(structure['value_columns']) > 0:
            selected_values = st.multiselect(
                "Select values to compare:",
                structure['value_columns'],
                default=structure['value_columns'][:5]  # Default to first 5
            )
        
        # Generate button
        generate_btn = st.button("🎬 Generate Animation")
        
        if generate_btn and selected_entity:
            with st.spinner(f"Generating animation for {selected_entity}..."):
                
                if chart_type == "Time Series Animation" and structure['time_columns']:
                    # Create time series animation
                    data = create_time_series_data(df, selected_entity, structure['entity_column'], structure['time_columns'])
                    
                    if len(data) > 1:
                        x = data["Time"].values
                        y = data["Value"].values
                        
                        # Create smooth interpolation
                        points_per_segment = 5
                        x_smooth = np.linspace(x.min(), x.max(), len(x) * points_per_segment)
                        y_smooth = np.interp(x_smooth, x, y)
                        
                        # Create animation
                        fig, ax = plt.subplots(figsize=(10, 6))
                        line, = ax.plot([], [], lw=3, color='#1f77b4')
                        
                        ax.set_xlim(min(x), max(x))
                        ax.set_ylim(min(y) * 0.9, max(y) * 1.1)
                        ax.set_title(f"{structure['data_type'].title()} Data Over Time: {selected_entity}", fontsize=16, pad=20)
                        ax.set_xlabel("Time", fontsize=12)
                        ax.set_ylabel("Value", fontsize=12)
                        ax.grid(True, linestyle="--", alpha=0.3)
                        
                        def init():
                            line.set_data([], [])
                            return line,
                        
                        def animate(i):
                            line.set_data(x_smooth[:i], y_smooth[:i])
                            return line,
                        
                        anim = FuncAnimation(
                            fig, animate, init_func=init,
                            frames=len(x_smooth), interval=50, blit=True
                        )
                        
                elif chart_type == "Value Comparison Animation" and structure['value_columns']:
                    # Create value comparison animation
                    if 'selected_values' in locals():
                        comparison_data = create_value_comparison_data(df, structure['entity_column'], selected_values, selected_entity)
                    else:
                        comparison_data = create_value_comparison_data(df, structure['entity_column'], structure['value_columns'], selected_entity)
                    
                    if comparison_data is not None and len(comparison_data) > 0:
                        categories = comparison_data["Category"].values
                        values = comparison_data["Value"].values
                        
                        # Create bar chart animation
                        fig, ax = plt.subplots(figsize=(12, 8))
                        bars = ax.bar(range(len(categories)), [0] * len(categories), color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
                        
                        ax.set_xlim(-0.5, len(categories) - 0.5)
                        ax.set_ylim(0, max(values) * 1.1)
                        ax.set_title(f"{structure['data_type'].title()} Value Comparison: {selected_entity}", fontsize=16, pad=20)
                        ax.set_xlabel("Categories", fontsize=12)
                        ax.set_ylabel("Values", fontsize=12)
                        ax.set_xticks(range(len(categories)))
                        ax.set_xticklabels(categories, rotation=45, ha='right')
                        plt.tight_layout()
                        
                        def animate_bars(frame):
                            for i, bar in enumerate(bars):
                                if frame > i * 5:  # Stagger the animation
                                    current_height = min(values[i], values[i] * (frame - i * 5) / 20)
                                    bar.set_height(current_height)
                            return bars
                        
                        anim = FuncAnimation(
                            fig, animate_bars,
                            frames=len(categories) * 5 + 20, interval=100, blit=False
                        )
                
                # Save animation
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                    try:
                        anim.save(tmpfile.name, writer="ffmpeg", fps=30)
                        st.success("✅ Animation Generated Successfully!")
                        
                        with open(tmpfile.name, "rb") as f:
                            video_bytes = f.read()
                            st.video(video_bytes)
                            
                            # Create filename based on data type and entity
                            filename = f"{structure['data_type']}_{selected_entity}_{chart_type.lower().replace(' ', '_')}.mp4"
                            st.download_button(
                                "📥 Download Animation", 
                                video_bytes, 
                                file_name=filename,
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Error generating video: {str(e)}")
                    finally:
                        if os.path.exists(tmpfile.name):
                            os.remove(tmpfile.name)
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please make sure your file has the correct format and contains numeric data.")

else:
    st.info("👆 Please upload a CSV or Excel file to get started!")
    
    # Show example of expected data formats
    st.subheader("📄 Supported Data Formats")
    
    tab1, tab2 = st.tabs(["Time Series Format", "Value Comparison Format"])
    
    with tab1:
        st.write("**Example: Country data over years**")
        example_time = pd.DataFrame({
            'Country Name': ['USA', 'Canada', 'Mexico'],
            'Indicator': ['Population', 'Population', 'Population'],
            '2020': [331, 38, 128],
            '2021': [332, 38.2, 129],
            '2022': [333, 38.4, 130]
        })
        st.dataframe(example_time)
    
    with tab2:
        st.write("**Example: Student performance data**")
        example_values = pd.DataFrame({
            'Student Name': ['Alice', 'Bob', 'Charlie'],
            'Math': [85, 92, 78],
            'Science': [90, 88, 82],
            'English': [88, 85, 90],
            'History': [92, 80, 85]
        })
        st.dataframe(example_values)