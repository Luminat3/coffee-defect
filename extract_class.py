import onnx

# Load the ONNX model
model_path = "models/best640.onnx"  # Update the path if needed
model = onnx.load(model_path)

# Extract model metadata
model_metadata = model.metadata_props

# Get model labels if available
labels = None
for prop in model_metadata:
    if prop.key.lower() in ["labels", "classes", "names"]:
        labels = prop.value
        break

print(labels)