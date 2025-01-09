import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the pre-trained DeepLabV3 model
def load_model():
    model = deeplabv3_resnet101(pretrained=True).eval()
    return model

# Perform segmentation using DeepLabV3
def segment_image(model, image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Resize segmentation mask back to original image size
    mask = cv2.resize(output_predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

# Add text to the selected region or background
def add_text(image, text, position, box_width, box_height, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate optimal font scale to fill the bounding box
    font_scale = 1.0
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    while text_size[0] < box_width and text_size[1] < box_height:
        font_scale += 0.1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    font_scale -= 0.1

    text_x = position[0] + (box_width - text_size[0]) // 2
    text_y = position[1] + (box_height + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded."
    file = request.files['image']
    if file.filename == '':
        return "No selected file."

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    # Get text and bounding box data from the form
    text = request.form['text']
    x1, y1 = int(request.form['x1']), int(request.form['y1'])
    x2, y2 = int(request.form['x2']), int(request.form['y2'])

    return process_image(filepath, text, x1, y1, x2, y2)

def process_image(image_path, text, x1, y1, x2, y2):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image."

    # Load the model
    model = load_model()

    # Perform segmentation
    mask = segment_image(model, image)

    # Extract the foreground (mask != 15 means foreground)
    foreground = image.copy()
    foreground[mask != 15] = 0

    # Extract the background (mask == 15 means background)
    background = image.copy()
    background[mask == 15] = 0

    # Ensure valid bounding box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Add text to the background in the bounding box area
    box_width, box_height = x2 - x1, y2 - y1
    add_text(background, text, (x1, y1), box_width, box_height, (0, 255, 255), 2)

    # Merge the foreground back into the background
    combined_image = background.copy()
    combined_image[mask == 15] = image[mask == 15]

    # Save the final image
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, combined_image)

    return send_file(result_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

