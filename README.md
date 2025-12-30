# Interactive Torque Project

This is a Dash-based desktop application for visualizing and analyzing torque measurement data.

## Two Deployment Modes

### 1. Standalone Desktop Application (Recommended for Factory Workers)

A Windows desktop application that runs directly without any setup. See [INSTALL.md](INSTALL.md) for installation instructions.

**Building the Standalone Application:**

1. **Activate virtual environment**
   ```powershell
   .\venv\Scripts\Activate.ps1   # PowerShell
   # OR
   call venv\Scripts\activate      # CMD
   ```

2. **Run the build script**
   ```powershell
   .\build.bat
   ```

3. **Find the executable**
   The standalone application will be in `dist\InteractiveTorqueApp\`

4. **Distribute to factory workers**
   - Zip the `dist\InteractiveTorqueApp\` folder
   - Share the zip file with factory workers
   - Workers can extract and run `InteractiveTorqueApp.exe`

### 2. Development Mode (For Developers)

For development and testing, run the application directly with Python.

## Development Setup

### Prerequisites

- Python 3.9+ installed and available on your PATH

### Quick Setup (Windows)

1. **Clone the repository**
   ```powershell
   git clone <repo-url>
   cd InteractiveTorqueProject
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # PowerShell
   # OR
   call venv\Scripts\activate      # CMD
   ```

3. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Copy and configure environment file**
   ```powershell
   copy .env.sample .env
   ```
   Edit `.env` to set the correct paths (e.g., `ALLOWED_DIRECTORY`, `DATABASE_URL`).

5. **Run the application in development mode**
   ```powershell
   python -m app.main
   ```
   Open your browser at http://localhost:8050

## Additional Scripts

- **setup.bat**: Automates steps 2–4 (env setup and dependency install)
- **build.bat**: Builds the standalone desktop application

## Notes

- Use `.env.sample` as a template. Do not commit your actual `.env`.
- For production, consider using PostgreSQL or another robust database.
- The standalone application uses PyWebView to create a native desktop window.
