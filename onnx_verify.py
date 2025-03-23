import onnx

model = onnx.load("mobilefacenet.onnx")  # Try loading the model
onnx.checker.check_model(model)  # Validate the model
print("✅ ONNX model is valid")
