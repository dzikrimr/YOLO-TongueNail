# download_nail_dataset.py
from roboflow import Roboflow
rf = Roboflow(api_key="0Ta4755XUrYm6iX1PwgB")
project = rf.workspace("sss-1tgcb").project("nail-segmentation-odcgv-fhaai")
version = project.version(1)
dataset = version.download("yolov8")
print("✅ Dataset downloaded at:", dataset.location)
