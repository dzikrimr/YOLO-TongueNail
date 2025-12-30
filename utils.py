from pathlib import Path

def check_dataset_structure(base_dir: str):
    """
    Pastikan dataset punya struktur: train/valid/test + images/labels
    """
    splits = ['train', 'valid', 'test']
    subfolders = ['images', 'labels']
    missing = []

    for split in splits:
        for sub in subfolders:
            path = Path(base_dir) / split / sub
            if not path.exists():
                missing.append(str(path))
    if missing:
        print("⚠️ Missing folders:", missing)
        return False
    print("✅ Dataset structure looks good!")
    return True

def list_images_labels(path: str):
    """
    List semua gambar dan labels di folder tertentu
    """
    img_files = sorted(Path(path).glob("*.jpg")) + sorted(Path(path).glob("*.png"))
    label_files = sorted(Path(path).glob("*.txt"))
    return img_files, label_files
