import subprocess, sys, os
from pathlib import Path

def run(cmd):
    print(f"\n▶️ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("🔍 Checking environment...")
    run("pip install -r requirements.txt")
    
    print("\n📥 Step 1: Preparing dataset...")
    run("python prepare_data.py")
    
    if not (Path("data/train").exists() and Path("data/test").exists()):
        print("❌ Dataset not found. Aborting.")
        sys.exit(1)
        
    print("\n🧠 Step 2: Training 3 CNN models (takes ~10-15 mins)...")
    run("python train_models.py")
    
    if not Path("models").exists() or len(list(Path("models").glob("*.pth"))) < 3:
        print("❌ Training failed. Check logs.")
        sys.exit(1)
        
    print("\n🌐 Step 3: Starting inference server...")
    print("👉 Open http://localhost:5000 in your browser")
    run("python server.py")

if __name__ == "__main__":
    main()