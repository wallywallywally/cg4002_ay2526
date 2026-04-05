import torch
from model_definition import CNN1DClassifier

model = CNN1DClassifier(input_size=30, num_classes=9)
dummy_input = torch.randn(1, 30, 25)
torch.onnx.export(model, dummy_input, "model.onnx")