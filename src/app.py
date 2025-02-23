from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import torchvision.transforms as T

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

# Modell laden (z. B. ResNet18 mit 2 Outputs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Lade die gelernten Gewichte
model.load_state_dict(torch.load("resnet18_carparts.pth", map_location=device))
model.eval()  # Setzt das Modell in Inference-Mode wg Dropout etc. 

# Transform definieren (genau wie im val/test)
inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# Endpoint: /predict
@app.route("/predict", methods=["POST"])
def predict():
    # 1) Empfange das Bild aus der Request
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # 2) Bild öffnen und transformieren
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = inference_transform(image).unsqueeze(0)  # (1, C, H, W) # wg Batch Dimension

    # 3) Inference auf GPU oder CPU
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)  # shape: [1, 2] -> [perspective_score_hood, perspective_score_backdoor_left]
    
    hood_score  = float(outputs[0, 0].item())
    backdoor_score = float(outputs[0, 1].item())

    # 4) JSON-Response
    result = {
        "perspective_score_hood": hood_score,
        "perspective_score_backdoor_left": backdoor_score
    }
    return jsonify(result)

# -----------------------------------------------------------------------------
# Starter (wenn du app.py direkt aufrufst)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Starte Flask-App auf 0.0.0.0 (damit von Docker aus erreichbar)
    app.run(host="0.0.0.0", port=5000, debug=False)
