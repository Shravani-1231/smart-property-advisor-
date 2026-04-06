"""
Smart Property Advisor - Main Entry Point
Run this file to start the application
"""

import os
import sys
import subprocess
import argparse


def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'plotly']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")


def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'src', 'notebooks']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Directories created")


def generate_data():
    """Generate initial dataset"""
    from src.data_generator import generate_and_save_dataset
    if not os.path.exists("data/property_data.csv"):
        print("Generating property data...")
        generate_and_save_dataset(n_samples=5000)
        print("✓ Data generated")
    else:
        print("✓ Data already exists")


def train_model():
    """Train the ML model"""
    from src.model_trainer import train_and_save_model
    if not os.path.exists("models/property_price_model.pkl"):
        print("Training ML model...")
        train_and_save_model()
        print("✓ Model trained and saved")
    else:
        print("✓ Model already exists")


def run_streamlit():
    """Run the Streamlit application"""
    print("\n" + "="*60)
    print("Starting Smart Property Advisor...")
    print("="*60 + "\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def run_api():
    """Run the FastAPI backend"""
    print("\n" + "="*60)
    print("Starting API Server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])


def main():
    parser = argparse.ArgumentParser(description='Smart Property Advisor')
    parser.add_argument('--setup', action='store_true', help='Setup the project (install deps, generate data, train model)')
    parser.add_argument('--api', action='store_true', help='Run only the API server')
    parser.add_argument('--app', action='store_true', help='Run only the Streamlit app')
    
    args = parser.parse_args()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    
    # Setup mode
    if args.setup:
        print("Setting up Smart Property Advisor...")
        check_dependencies()
        setup_directories()
        generate_data()
        train_model()
        print("\n✅ Setup complete! Run 'python run.py' to start the app.")
        return
    
    # API only mode
    if args.api:
        setup_directories()
        generate_data()
        train_model()
        run_api()
        return
    
    # App only mode (default)
    setup_directories()
    generate_data()
    train_model()
    run_streamlit()


if __name__ == "__main__":
    main()