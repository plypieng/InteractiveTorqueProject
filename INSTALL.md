# Installation Guide for Factory Workers

## Interactive Torque Analysis Application

This guide will help you install and run the Interactive Torque Analysis application on your Windows computer.

---

## System Requirements

- **Operating System**: Windows 10 or Windows 11
- **Disk Space**: At least 500 MB free space
- **Memory**: At least 4 GB RAM

> [!NOTE]
> The application uses Edge WebView2, which is built into Windows 10/11. If you're using an older version of Windows, you may need to install the Edge WebView2 Runtime (free from Microsoft).

---

## Installation Steps

### 1. Download the Application

Download the `InteractiveTorqueApp.zip` file from your IT administrator or shared network location.

### 2. Extract the Files

1. Right-click on `InteractiveTorqueApp.zip`
2. Select **Extract All...**
3. Choose a location (e.g., `C:\Program Files\InteractiveTorqueApp` or your Desktop)
4. Click **Extract**

### 3. Run the Application

1. Navigate to the extracted folder `InteractiveTorqueApp`
2. Double-click on **InteractiveTorqueApp.exe**
3. The application window will open automatically

> [!IMPORTANT]
> **First Launch**: The first time you run the application, Windows may show a security warning. Click **More info**, then click **Run anyway**. This is normal for applications not downloaded from the Microsoft Store.

---

## Using the Application

### First Launch

When you first launch the application, it will create a data folder in your user directory:
```
C:\Users\<YourUsername>\AppData\Roaming\InteractiveTorqueApp\
```

This folder contains:
- **Database**: All your measurement data
- **Data files**: Your uploaded measurement files
- **Logs**: Application logs for troubleshooting

### Main Features

1. **File Upload**: Upload torque measurement CSV files
2. **Data Visualization**: View normal, filtered, and FFT plots
3. **Analysis**: Automatic spike detection and RMS analysis
4. **Model Training**: Train machine learning models on your data
5. **Database Review**: Review historical measurements

---

## Troubleshooting

### Application Won't Start

- **Check Windows Version**: Make sure you have Windows 10 or 11
- **Install WebView2**: Download and install [Edge WebView2 Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section)
- **Check Log File**: Look in `C:\Users\<YourUsername>\AppData\Roaming\InteractiveTorqueApp\app.log` for error messages

### Application Crashes or Freezes

1. Close the application
2. Delete the log file: `C:\Users\<YourUsername>\AppData\Roaming\InteractiveTorqueApp\app.log`
3. Restart the application

### Data Not Showing Up

- Make sure your measurement files are in CSV format
- Check that the data directory path is correct in Settings

---

## Getting Help

If you encounter problems:

1. Check the log file in `C:\Users\<YourUsername>\AppData\Roaming\InteractiveTorqueApp\app.log`
2. Contact your IT administrator or supervisor
3. Provide the log file when reporting issues

---

## Uninstallation

To remove the application:

1. Delete the application folder (e.g., `C:\Program Files\InteractiveTorqueApp`)
2. (Optional) Delete your data folder: `C:\Users\<YourUsername>\AppData\Roaming\InteractiveTorqueApp`

> [!WARNING]
> Deleting the data folder will remove all your measurement data and trained models. Make sure to back up any important data first.
