import os, sys, shutil, random
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

def prepare_local_dataset():
    # Check if already prepared
    if (TRAIN_DIR / "mask").exists() and (TEST_DIR / "mask").exists():
        print("✅ Dataset already prepared. Skipping.")
        return True

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check for raw input folders
    with_mask_src = DATA_DIR / "with_mask"
    without_mask_src = DATA_DIR / "without_mask"

    if not (with_mask_src.exists() and without_mask_src.exists()):
        print("❌ Required folders not found!")
        print("\n📁 Please structure your data like this:")
        print(f"  data/")
        print(f"    ├── with_mask/    [images of people wearing masks]")
        print(f"    └── without_mask/ [images of people without masks]")
        print("\nThen run this script again.")
        return False

    print("📂 Found local dataset. Preparing train/test split (80/20)...")
    random.seed(42)  # Reproducible split

    # Create target directories
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        (split_dir / "mask").mkdir(parents=True, exist_ok=True)
        (split_dir / "no_mask").mkdir(parents=True, exist_ok=True)

    # Process each class
    for src_folder, dest_class in [(with_mask_src, "mask"), (without_mask_src, "no_mask")]:
        # Support common image extensions
        images = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.jpeg")) + list(src_folder.glob("*.png"))
        if not images:
            print(f"⚠️  No images found in {src_folder.name}/")
            continue
            
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)

        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy2(img, TRAIN_DIR / dest_class / img.name)
        for img in test_imgs:
            shutil.copy2(img, TEST_DIR / dest_class / img.name)

        print(f"  ✅ {src_folder.name}: {len(train_imgs)} train | {len(test_imgs)} test")

    # Verify output
    train_mask = len(list((TRAIN_DIR / "mask").glob("*")))
    test_mask = len(list((TEST_DIR / "mask").glob("*")))
    
    if train_mask == 0 or test_mask == 0:
        print("❌ Split failed. Check your input images.")
        return False

    print("\n✅ Dataset prepared successfully!")
    print(f"   📁 Output: {DATA_DIR}/train/{{mask, no_mask}} | {DATA_DIR}/test/{{mask, no_mask}}")
    print("   🔜 You can now run: python run_all.py")
    return True

if __name__ == "__main__":
    success = prepare_local_dataset()
    sys.exit(0 if success else 1)