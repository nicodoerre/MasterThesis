import pandas as pd
import numpy as np
import random

axis_labels = {
    "x_labels": ["Time", "Frequency", "Categories", "Days", "Months", "Years", "Age", "Distance","Time (seconds)", 
                    "Distance (m)", "Temperature (°C)", "Pressure (Pa)", "Speed (m/s)","Energy (Joules)", "Angle (Degrees)", 
                    "Power (Watts)", "Voltage (V)", "Frequency (Hz)","Amplitude", "Mass (kg)", "Wavelength (nm)", "Resistance (Ω)", "Force (N)",
                    "Probability (%)", "Population Density", "Market Share (%)", "Cost ($)", "Altitude (m)",
                    "Signal Strength (dB)", "Time of Day", "Frame Number", "Sample Index", "Iteration Count",
                    "Noise Level", "Error Rate (%)", "Learning Rate", "Reaction Time (ms)", "Pixel Intensity",
                    "Brightness", "Contrast", "Saturation", "Hue", "Luminance", "Chroma", "Volume (m³)"],
    "y_labels": ["Value", "Count", "Percentage", "Temperature", "Sales", "Growth", "Intensity", "Score","Velocity (m/s)", 
                    "Acceleration (m/s²)", "Growth Rate (%)", "Temperature Change (Δ°C)", "Stress (Pa)",
                    "Voltage Difference (ΔV)", "Power Output (W)", "Signal-to-Noise Ratio (SNR)", "Correlation Coefficient", "Stock Price ($)",
                    "Efficiency (%)", "Entropy", "Memory Usage (MB)", "Processing Time (ms)", "Brightness Level",
                    "Water Level (m)", "Magnitude", "Radiation Dose (mSv)", "Conversion Rate (%)", "Battery Life (%)",
                    "Error Magnitude", "Satisfaction Score", "Transmission Speed (Mbps)", "Pressure Gradient", "Bitrate (kbps)",
                    "Temperature Fluctuation", "Confidence Score", "Logarithmic Scale Value", "Success Rate (%)", "Information Density"]
}
titles = [
    "Data Analysis", "Monthly Trends", "Distribution Overview", 
    "Sales Growth", "Temperature Variation", "Frequency Distribution",
    "Performance Metrics", "Yearly Comparison", "User Activity",
    "Trends Over Time", "Performance Analysis", "Growth Comparison", 
    "Energy Consumption Patterns", "Market Trends",
    "Signal Strength vs. Distance", "Effect of Temperature on Performance", 
    "Probability Distributions", "Learning Curve Progression", "Error Reduction Over Iterations",
    "Computational Speed Analysis", "Real-Time Data Fluctuations", "Sensor Accuracy Comparison", 
    "Statistical Variability", "Experimental Results Overview",
    "Power vs. Efficiency Trade-off", "Noise Impact on Signal Processing", 
    "Economic Growth Models", "Voltage vs. Current Characteristics", "Seasonal Variations",
    "Relationship Between Mass and Force", "Accuracy vs. Computational Cost", 
    "Changes in Pressure Over Time", "Scaling Laws in Physics", "Impact of External Factors on Performance",
    "Frequency Response of a System", "Machine Learning Model Evaluation", "Network Latency vs. Bandwidth",
    "Predictive Analysis of Trends", "Stability Analysis of a System"
]

color_palettes = {
    "cool_palette": ["#1f77b4", "#aec7e8", "#ffbb78", "#ff7f0e", "#2ca02c", "#98df8a"],
    "warm_palette": ["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94"],
    "earth_tones": ["#8B4513", "#A0522D", "#CD853F", "#D2B48C", "#DEB887", "#F4A460"],
    "bright_vibrant": ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#2ca02c"],
    "sunset": ["#ff8c00", "#ffa07a", "#fa8072", "#e9967a", "#ff6347", "#ff4500"],
    "pastel_palette": ["#bbd6f2", "#b0e57c", "#fcd581", "#fdc0c0", "#a6dcef", "#ffdbac"],
    "grayscale": ["#333333", "#666666", "#999999", "#cccccc", "#e0e0e0", "#f2f2f2"]
}

single_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#ff69b4", "#b22222", "#ff4500", "#ff8c00", "#ffd700", "#00ff00",
    "#40e0d0", "#4682b4", "#000080", "#6a5acd", "#ff1493", "#c71585", "#da70d6", "#dda0dd",
    "#7b68ee", "#483d8b", "#2f4f4f", "#556b2f", "#8b4513", "#d2691e", "#cd5c5c", "#ffdab9",
    "#ffb6c1", "#db7093", "#ff6347", "#ff8c69", "#ff1493", "#8b0000", "#b03060", "#ff1493",
    "#9acd32", "#8fbc8f", "#00fa9a", "#adff2f", "#7fff00", "#7cfc00", "#32cd32", "#98fb98",
    "#7fffd4", "#66cdaa", "#5f9ea0", "#4682b4", "#1e90ff", "#4169e1", "#6a5acd", "#8a2be2",
    "#9400d3", "#9932cc", "#8b008b", "#800080", "#dda0dd", "#d2b48c", "#cd853f", "#a52a2a",
    "#deb887", "#bc8f8f", "#8b4513", "#ffdead", "#ffebcd", "#ffefdb", "#f0fff0", "#ffefdb",
    "#708090", "#778899", "#2f4f4f", "#d2b48c", "#bc8f8f", "#ff4500", "#ff8c69", "#ff8c00",
    "#ffd700", "#daa520", "#cd853f", "#a0522d", "#8b4513", "#8b0000", "#a9a9a9", "#696969"
]
background_colors = [
    "#f5f5f5", "#f0f0f0", "#eaeaea", "#e0e0e0", "#dcdcdc", "#d3d3d3", "#c0c0c0",
    "#faf0e6", "#fff5ee", "#fdf5e6", "#f0fff0", "#e6e6fa", "#f0f8ff", "#f8f8ff", "#fffacd",
    "#fafafa", "#f7f7f7", "#f0f8ff", "#f5f5dc", "#fdf5e6", "#ffefd5", "#fff8dc", "#fffacd",
    "#fafad2", "#e0eee0", "#e0ffff", "#e6e6fa", "#f5fffa", "#ffebcd", "#ffefdb", "#f0f5f5",
    "#f5f5f5", "#f8f8ff", "#fff8e7", "#f7f0f5", "#faf0f8", "#f8f5f0", "#faf0e6", "#f3f2f1",
    "#edeef0", "#ebebeb", "#dcdcdc", "#d3d3d3", "#c8c8c8", "#cccccc", "#bdbdbd", "#a9a9a9",
    "#8f8f8f", "#dcd0ff", "#d8bfd8", "#d3d3ff", "#f0fff0", "#fdf5f7", "#fffaf0", "#ffe4e1",
    "#ffe4c4", "#fff8e7", "#f7e9e9", "#e6e6fa", "#f4f1de", "#ece6d6", "#e6e6e6", "#faf4d3"
]

text_size_ranges = {
    "small": (8,12),
    "medium": (12,16),
    "large": (16,24)
}

legend_labels = ["Dataset 1", "Dataset 2", "Experiment A", "Control Group", "Prediction", "Actual", "Model 1", "Model 2", "Model 3", 
                "Baseline", "Optimized", "Random", "Test Data", "Training Data", "Validation Data", "Results", "Analysis", "Comparison", 
                "Evaluation", "Forecast", "Projection", "Simulation", "Scenario", "Outcome", "Trend", "Pattern", "Insight", "Observation", 
                "Conclusion", "Recommendation", "Implication", "Impact", "Assessment", "Review", "Summary", "Overview", "Introduction", "Background", 
                "Methodology", "Results", "Discussion", "Conclusion", "References", "Appendix", "Acknowledgements", "Abstract", "Keywords", "Table of Contents", 
                "List of Figures", "List of Tables", "Glossary", "Abbreviations", "Index", "References", "Appendix", "Acknowledgements", "Abstract", "Keywords", 
                "Table of Contents", "List of Figures", "List of Tables", "Glossary", "Abbreviations", "Index"]

annotation_texts = ["Peak", "Low Point", "Outlier", "Average", "Trendline", "Threshold", "Critical Point", "Optimal Value", "Warning", "Error", "Success",
                    "Target", "Goal", "Limit", "Boundary", "Breakpoint", "Milestone", "Record", "High", "Low", "Start", "End", "Focus", "Change", "Impact",
                    "Analysis", "Conclusion", "Recommendation", "Insight", "Observation", "Pattern", "Trend", "Forecast", "Projection", "Simulation", "Scenario",
                    "Outcome", "Implication", "Assessment", "Review", "Summary", "Overview", "Introduction", "Background", "Methodology", "Results", "Discussion",
                    "Conclusion", "References", "Appendix", "Acknowledgements", "Abstract", "Keywords", "Table of Contents", "List of Figures", "List of Tables", "Glossary",
                    "Abbreviations", "Index", "References", "Appendix", "Acknowledgements", "Abstract", "Keywords", "Table of Contents", "List of Figures", "List of Tables", "Glossary",
                    "Abbreviations", "Index"]

labels1 = random.sample(legend_labels, 4)
df1 = pd.DataFrame({
    labels1[0]: np.random.normal(size=100),
    labels1[1]: np.random.uniform(size=100),
    labels1[2]: np.random.normal(loc=1, scale=2, size=100),
    labels1[3]: np.random.exponential(size=100)
})

labels2 = random.sample(legend_labels, 3)
df2 = pd.DataFrame({
    labels2[0]: np.random.randint(1, 10, 100),
    labels2[1]: np.random.randint(10, 20, 100),
    labels2[2]: np.random.randint(20, 30, 100),
})

labels3 = random.sample(legend_labels, 3)
df3 = pd.DataFrame({
    labels3[0]: np.random.normal(170, 10, 100),
    labels3[1]: np.random.normal(65, 15, 100),
    labels3[2]: np.random.randint(20, 50, 100),
})

dataframes = {
    'df1': df1,
    'df2': df2,
    'df3': df3
}

line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^", "v", "<", ">", "x","."]


grid_options = [True, False]

font_families = [
    "Arial", "Times New Roman", "Courier New", "Comic Sans MS", "Georgia", 
     "Verdana", "Trebuchet MS", "Palatino Linotype"
]
font_weights = ["normal", "bold", "light", "heavy"]
font_styles = ["normal", "italic", "oblique"]

legend_positions = [
    (1.2, 0.5),  # Right side, centered
    (-0.4, 0.5), # Left side, centered
    (1.2, 1.0),  # Top-right
    (-0.4, 1.0), # Top-left
    (1.2, -0.1), # Bottom-right
    (-0.4, -0.1) # Bottom-left
]