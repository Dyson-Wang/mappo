# 层分割flask服务

import io
import time
import json
import base64
import io
import pyRAPL
from flask import Flask, request, jsonify
from PIL import Image
import pynvml

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet152, ResNet152_Weights
import torchvision.transforms as transforms

from resnet152_define import ResNet152Split

# ------------------- Flask 初始化 -------------------
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet152Split()
# model.load_state_dict(torch.load("./resnet152-394f9c45.pth", map_location=device))

model = model.to(device)
model.eval()

pyRAPL.setup()

# 类别名称
with open("imagenet_classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------- /head 接口 -------------------
@app.route("/head", methods=["POST"])
def head_inference():
    if "file" not in request.files or "split" not in request.form:
        return jsonify({"error": "Missing file or split param"}), 400

    file = request.files["file"]
    try:
        split = int(request.form["split"])
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            meter = pyRAPL.Measurement("head_inference")
            meter.begin()
            start = time.perf_counter()
            middle_output = model(input_tensor, mode="head", split=split)
            elapsed = time.perf_counter() - start
            meter.end()

        buffer = io.BytesIO()
        torch.save(middle_output.cpu(), buffer)
        compressed_bytes = buffer.getvalue()
        compressed_base64 = base64.b64encode(compressed_bytes).decode("utf-8")

        size_bytes = len(compressed_bytes)

        energy_joules = meter.result.pkg[0] / 1e6  # 微焦耳 → 焦耳
        power_watt = energy_joules / elapsed if elapsed > 0 else 0

        # 返回 tensor 的嵌套 list 表示
        return jsonify({
            "features": compressed_base64,
            "shape": list(middle_output.shape),
            "inference_time_seconds": f"{elapsed:.4f}",
            "device": str(device),
            "compressed_size_bytes": size_bytes,
            "energy": f"{energy_joules:.4f}",
            "power": f"{power_watt:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- /tail 接口 -------------------
@app.route("/tail", methods=["POST"])
def tail_inference():
    try:
        data = request.get_json()
        split = int(data.get("split", 0))
        compressed_bytes = base64.b64decode(data["features"])
        buffer = io.BytesIO(compressed_bytes)
        features = torch.load(buffer).to(device)

        if features.dim() != 4:
            return jsonify({"error": "Input features must be a 4D tensor"}), 400

        # GPU 能耗初始化
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_usage_start = pynvml.nvmlDeviceGetPowerUsage(handle)  # 毫瓦

        with torch.no_grad():
            start = time.perf_counter()
            output = model(features, mode="tail", split=split)
            elapsed = time.perf_counter() - start

        power_usage_end = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()

        # GPU 功率估算（平均功耗）
        avg_gpu_power_watts = ((power_usage_start + power_usage_end) / 2) / 1000  # 转为瓦

        top5_prob, top5_idx = torch.topk(output[0], 5)
        return jsonify({
            "top5": [
                {"class": classes[i], "probability": f"{p:.4f}"}
                for p, i in zip(top5_prob, top5_idx)
            ],
            "inference_time_seconds": f"{elapsed:.4f}",
            "device": str(device),
            "energy":  round(avg_gpu_power_watts * elapsed, 4),
            "power": f"{avg_gpu_power_watts:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route("/full", methods=["POST"])
def full_inference():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            start = time.time()
            output = model(input_tensor, mode="full")
            elapsed = time.time() - start

        top5_prob, top5_idx = torch.topk(output[0], 5)
        return jsonify({
            "top5": [
                {"class": classes[i], "probability": f"{p:.4f}"}
                for p, i in zip(top5_prob, top5_idx)
            ],
            "inference_time_seconds": f"{elapsed:.4f}",
            "device": str(device)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- 启动 -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
