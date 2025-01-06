import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(device)
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    """Preprocess the image for DeepLabV3."""
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return input_tensor.to(device), np.array(image)

# Run segmentation
def run_segmentation(image_path):
    """Perform segmentation on the input image."""
    input_tensor, original_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # Get output from the model
    segmentation_map = torch.argmax(output, dim=0).cpu().numpy()  # Convert to NumPy
    return segmentation_map, original_image

# Separate foreground and background
def separate_foreground_background(original_image, segmentation_map):
    """Separate the foreground and background from the segmentation map."""
    # Create foreground mask (segmentation_map > 0)
    foreground_mask = (segmentation_map > 0).astype(np.uint8)
    
    # Create the background mask
    background_mask = (segmentation_map == 0).astype(np.uint8)
    
    # Apply masks to the original image
    foreground = original_image * foreground_mask[..., None]  # Retain only foreground
    background = original_image * background_mask[..., None]  # Retain only background
    
    return foreground, background

# Add text to background
def add_text_to_image(image, text, position=(50, 50), font_size=30, color=(255, 255, 255)):
    """Add text to the given image."""
    # Convert the background to a PIL image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Load a font (default font if no custom font is provided)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text to the image
    draw.text(position, text, fill=color, font=font)
    
    return np.array(pil_image)

# Visualize results
def visualize_results(original_image, segmentation_map, foreground, background_with_text):
    """Visualize the original image, segmentation map, foreground, and background with text."""
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Segmentation Map")
    plt.imshow(segmentation_map, cmap="jet")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Foreground")
    plt.imshow(foreground)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Background with Text")
    plt.imshow(background_with_text)
    plt.axis("off")
    plt.show()

# Example usage
image_path = "/home/gpandit/Pictures/Screenshot from 2024-11-18 16-34-46.png"  # Replace with your image path
segmentation_map, original_image = run_segmentation(image_path)
foreground, background = separate_foreground_background(original_image, segmentation_map)

# Add text to the background
background_with_text = add_text_to_image(background, "Yoga", position=(50, 50), font_size=50, color=(255, 255, 255))

# Visualize results
visualize_results(original_image, segmentation_map, foreground, background_with_text)

