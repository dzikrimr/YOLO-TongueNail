import argparse
from pathlib import Path
import yaml
from utils import check_dataset_structure
from ultralytics import YOLO

def create_yaml(data_dir: str, classes: list, yaml_path: str):
    """
    Generate dataset.yaml untuk YOLO training
    """
    data = {
        'train': str((Path(data_dir) / 'train' / 'images').resolve()),
        'val': str((Path(data_dir) / 'valid' / 'images').resolve()),
        'test': str((Path(data_dir) / 'test' / 'images').resolve()),
        'nc': len(classes),
        'names': classes
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    print(f"✅ YAML file created at {yaml_path}")


def train_yolo(model_name: str, data_yaml: str, epochs: int = 50, batch_size: int = 16,
               imgsz: int = 640, project_name: str = "runs/train"):
    """
    Train YOLO model menggunakan dataset yang sudah di YAML
    """
    print(f"🚀 Starting YOLO training for {Path(data_yaml).stem}...")
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=project_name,
        name=Path(data_yaml).stem,
        exist_ok=True
    )
    print(f"✅ Training finished for {Path(data_yaml).stem}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO for lidah or kuku dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path ke folder dataset (misal: data/lidah atau data/kuku)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO pretrained model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Cek struktur dataset
    if not check_dataset_structure(args.data_dir):
        raise ValueError("Dataset structure salah. Pastikan ada train/valid/test + images/labels")

    # YAML
    yaml_file = f"{args.data_dir}.yaml"
    classes = [Path(args.data_dir).name, "none"] 
    create_yaml(args.data_dir, classes, yaml_file)

    # Train
    train_yolo(
        model_name=args.model,
        data_yaml=yaml_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz
    )
