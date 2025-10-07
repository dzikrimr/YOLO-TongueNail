import onnx
from onnx_tf.backend import prepare

# Load model ONNX
onnx_model = onnx.load("runs/train/kuku/weights/best.onnx")

# Convert ONNX → TensorFlow
tf_rep = prepare(onnx_model)

# Export ke SavedModel
tf_rep.export_graph("runs/train/kuku/weights/tf_model")
print("✅ ONNX → TensorFlow SavedModel done!")

# Convert SavedModel → TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("runs/train/kuku/weights/tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # opsional, untuk optimasi size & speed
tflite_model = converter.convert()

# Simpan TFLite
with open("runs/train/kuku/weights/best.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted to TFLite successfully!")
