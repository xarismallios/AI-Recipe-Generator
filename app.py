import os
import json
import requests
import openai
import torch
from flask import Flask, request, render_template, redirect
from PIL import Image
from torchvision import models, transforms

# Set up Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load OpenAI API key
openai.api_key = "XXX"

# Function to download ImageNet labels if not already present
def download_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    if response.status_code == 200:
        with open("imagenet_labels.txt", "w") as f:
            json.dump(response.json(), f)
    else:
        raise Exception("Failed to download ImageNet labels")

# Load ImageNet labels
if not os.path.exists("imagenet_labels.txt"):
    download_imagenet_labels()

with open("imagenet_labels.txt") as f:
    labels = json.load(f)

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def identify_items(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
    _, indices = torch.sort(out, descending=True)
    
    detected_items = []
    for idx in indices[0][:5]:
        index = idx.item()
        if 0 <= index < len(labels):
            detected_items.append(labels[index])
        else:
            detected_items.append("Unknown")

    return detected_items

def get_recipe(ingredients):
    prompt = f"I have the following ingredients: {', '.join(ingredients)}. Can you suggest a recipe?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Save uploaded file
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Identify ingredients
            ingredients = identify_items(file_path)
            
            # Get recipe
            recipe = get_recipe(ingredients)

            return render_template("result.html", ingredients=ingredients, recipe=recipe)

    return render_template("upload.html")

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
