# Torque Measurement Visualization and Analysis Application

Welcome to the **Torque Measurement Visualization and Analysis Application**! This application is designed to assist in visualizing, analyzing, and evaluating torque measurements from your in-house automatic torque measurement machine. It provides a user-friendly interface to process CSV data files, perform advanced analyses, and determine PASS or FAILED results based on multiple criteria.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Using the Interface](#using-the-interface)
- [Analysis Criteria](#analysis-criteria)
  - [1. Sudden Spike Detection](#1-sudden-spike-detection)
  - [2. HPF_RMS Threshold Comparison](#2-hpf_rms-threshold-comparison)
  - [3. Machine Learning Prediction](#3-machine-learning-prediction)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Simultaneous Visualization**: Display all graphs for up to two measurement files side by side for easy comparison.
- **Adjustable Parameters**:
  - High-pass filter cutoff frequency.
  - RMS window size for moving RMS calculation.
  - HPF_RMS threshold value.
  - Y-axis scale adjustment.
- **Automatic File Detection**: The application automatically detects new CSV files added to the specified directory without needing to refresh the page.
- **PASS or FAILED Analysis**:
  - Detect sudden spikes in filtered data.
  - Compare HPF_RMS values against a user-defined threshold.
  - Placeholder for machine learning model predictions.
- **Data Download**: Easily download processed CSV and PDF reports.
- **User-Friendly Interface**: Intuitive layout with helpful tooltips and documentation.

---

## Getting Started

### Prerequisites

- **Python 3.7 or higher**
- **Git** (optional, for cloning the repository)
-  specify your data directory.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/plypieng/InteractiveTorqueProject.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd InteractiveTorqueProject
   ```

3. **Create a Virtual Environment** (recommended):

   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On Unix or MacOS:

     ```bash
     source venv/bin/activate
     ```

5. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Configure the Data Directory**:

   - Ensure that your CSV data files are located in the `W:/` drive.
   - If you need to change the data directory, update the `ALLOWED_DIRECTORY` variable in `app.py`:

     ```python
     ALLOWED_DIRECTORY = 'path/to/your/data/directory'
     ```

---

## Usage

### Running the Application

1. **Start the Application**:

   ```bash
   python app.py
   ```

2. **Access the Application**:

   - Open a web browser and navigate to `http://localhost:8050`.

### Using the Interface

1. **Select Measurement Files**:

   - Choose up to two CSV files from the dropdown menus to visualize and compare their data.

2. **Adjust Parameters**:

   - **High-Pass Filter Cutoff Frequency**: Enter the desired cutoff frequency in Hz.
   - **RMS Window Size**: Specify the window size for calculating the moving RMS.
   - **HPF_RMS Threshold**: Set the threshold value for HPF_RMS analysis.
   - **Y-axis Scale**: Use the slider to adjust the Y-axis range of the plots.

3. **View Graphs and Analysis**:

   - The application displays the normal data plot, filtered data plot, and FFT plot for each selected file.
   - Extracted features and analysis results are shown alongside the graphs.

4. **Interpret Analysis Results**:

   - **Overall Result**: Displays PASS or FAILED based on the analysis criteria.
   - **Detailed Messages**: Provides information on sudden spike detection, HPF_RMS threshold comparison, and machine learning predictions.

5. **Download Data**:

   - Use the "Download CSV" and "Download PDF" buttons to download the processed data and reports.

6. **Automatic File Updates**:

   - The application automatically updates the file selection options when new CSV files are added to the data directory.

---

## Analysis Criteria

### 1. Sudden Spike Detection

- **Purpose**: Identify sudden spikes in the filtered data, which may indicate dust particles or surface defects on the ball screw.
- **Method**:
  - Calculate the absolute difference between consecutive points in the high-pass filtered data.
  - Detect spikes where the difference exceeds a predefined threshold.
- **Result**:
  - If spikes are detected, the analysis recommends re-measurement.

### 2. HPF_RMS Threshold Comparison

- **Purpose**: Assess the roughness of the ball screw surface based on the HPF_RMS value.
- **Method**:
  - Calculate the moving RMS of the filtered data using the specified window size.
  - Compare the maximum HPF_RMS value against the user-defined threshold.
- **Result**:
  - If the HPF_RMS exceeds the threshold, it may indicate excessive surface roughness, leading to increased heat generation.

### 3. Machine Learning Prediction

- **Purpose**: Utilize advanced machine learning models to predict the condition of the ball screw.
- **Method**:
  - **Placeholder**: Currently, this is a placeholder function.
  - **Future Integration**: Plans to incorporate models like LSTM autoencoders or supervised learning models trained on the CSV data.
- **Result**:
  - Provides a PASS or FAILED prediction based on the model's output.

---

## Project Structure

```
torque-measurement-app/
├── app.py                # Main Dash application
├── utils.py              # Utility functions
├── plots.py              # Plot creation functions
├── pdf_generator.py      # PDF report generation
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

## Future Work

- **Machine Learning Integration**:

  - Train and integrate machine learning models for more accurate predictions.
  - Models may include LSTM autoencoders, supervised classifiers, etc.

- **Enhanced Reporting**:

  - Improve PDF report generation to include analysis results and recommendations.

- **User Interface Improvements**:

  - Add notifications or alerts for new files detected.
  - Provide more customization options for plots and analysis parameters.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as per the terms of the license.

---
