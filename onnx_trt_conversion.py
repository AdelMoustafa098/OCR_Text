import tf2onnx
import onnx
import tensorrt as trt
from config import *
import tensorflow as tf

# Load the saved Keras model
saved_model_path = "crnn_ocr_model_cross_entropy.h5"
output_onnx_path = "crnn_ocr_model.onnx"

# Convert the Keras model to a TensorFlow SavedModel

model = tf.keras.models.load_model(saved_model_path)

# Convert to ONNX
spec = (tf.TensorSpec((None, IMAGE_HEIGHT, IMAGE_WIDTH, 1), tf.float32),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save(model_proto, output_onnx_path)

print(f"Model has been converted to ONNX format and saved as {output_onnx_path}")


onnx_model_path = "crnn_ocr_model.onnx"
trt_engine_path = "crnn_ocr_model.trt"

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create builder, network, and parser
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX model
with open(onnx_model_path, "rb") as f:
    if not parser.parse(f.read()):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# Create builder configuration
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# Build serialized network
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build the TensorRT engine")

# Save serialized engine to file
with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"Model has been converted to TensorRT format and saved as {trt_engine_path}")