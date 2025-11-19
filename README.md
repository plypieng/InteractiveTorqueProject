# Interactive Torque Project

This is a Dash-based web application for visualizing and analyzing torque measurement data.

## Prerequisites

- Python 3.9+ installed and available on your PATH
- Redis server running locally (used by Celery)

## Quick Setup (Windows)

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

   Edit `.env` to set the correct paths and credentials (e.g., `ALLOWED_DIRECTORY`, `DATABASE_URL`).

5. **Start Redis server**

   Ensure Redis is running (e.g., `redis-server`).

6. **(Optional) Start Celery worker**

   ```powershell
   celery -A app.tasks worker --loglevel=info
   ```

7. **Run the Dash app**

   ```powershell
   python -m app.main
   ```

   Open your browser at http://localhost:8050

## Additional Scripts

- **setup.bat**: Automates steps 2–4 (env setup and dependency install)

## Notes

- Use `.env.sample` as a template. Do not commit your actual `.env`.
- For production, consider using PostgreSQL or another robust database.
