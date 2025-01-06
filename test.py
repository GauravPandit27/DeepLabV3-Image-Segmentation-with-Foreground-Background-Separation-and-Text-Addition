import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(device)
model.eval()

# Preprocessing function
def preprocess_image(image_path):
    """Preprocess the image for DeepLabV3."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.0, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return input_tensor.to(device), np.array(image)

# Run segmentation
def run_segmentation(image_path):
    """Perform segmentation on the input image."""
    input_tensor, original_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]  # Get output from the model

    output = output.cpu()  # Ensure tensor is on CPU
    segmentation_map = torch.argmax(output, dim=0).numpy()  # Convert to NumPy
    return segmentation_map, original_image

# Separate foreground and background
def separate_foreground_background(original_image, segmentation_map):
    """Separate foreground and background from the image."""
    mask = segmentation_map != 0  # Non-zero regions are the foreground
    foreground = original_image * mask[:, :, None]
    background = original_image * (~mask[:, :, None])
    return foreground, background

# Global variables for rectangle
rect_start = None
rect_end = None
img_original = None
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Yoga"
drawing_box = False  # Flag to prevent drawing multiple bounding boxes
foreground_cropped = None

# Mouse callback function to draw rectangle
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, img_original, drawing_box, foreground_cropped
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing_box:  # Allow drawing only if no box is being drawn
            rect_start = (x, y)  # Set the starting point of the rectangle
            drawing_box = True  # Set flag to prevent further drawing

    elif event == cv2.EVENT_MOUSEMOVE and rect_start:
        rect_end = (x, y)  # Set the ending point of the rectangle as the mouse moves
        img_copy = img_original.copy()  # Create a copy to show updates
        cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)  # Draw the rectangle
        cv2.imshow('Select Object', img_copy)  # Show updated image
        
    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)  # Finalize the rectangle on mouse release
        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end
            # Draw rectangle on final image
            cv2.rectangle(img_original, rect_start, rect_end, (0, 255, 0), 2)
            
            # Calculate text size based on bounding box
            box_width = x2 - x1
            box_height = y2 - y1
            font_scale = min(box_width, box_height) // 30  # You can adjust this scaling factor
            if font_scale < 1:
                font_scale = 1  # Prevent font size from being too small
            
            # Calculate the text size dynamically
            text_size = cv2.getTextSize(text, font, font_scale, 12)[0]  # Increased thickness for bolder text
            text_width, text_height = text_size
            text_position = (x1 + (box_width - text_width) // 2, y1 + (box_height + text_height) // 2)  # Center text inside box
            
            # Add bold, white shadowed text inside the bounding box (3D effect)
            shadow_offset = 3  # Shadow offset for 3D effect
            shadow_color = (50, 50, 50)  # Shadow color (dark gray)
            cv2.putText(img_original, text, (text_position[0] + shadow_offset, text_position[1] + shadow_offset),
                        font, font_scale, shadow_color, 12, cv2.LINE_AA)  # Add shadow
            
            cv2.putText(img_original, text, text_position, font, font_scale, (255, 255, 255), 12, cv2.LINE_AA)  # White color text
            cv2.imshow('Select Object', img_original)  # Show the image with rectangle and text

            # Crop the foreground based on the rectangle coordinates
            foreground_cropped = img_original[y1:y2, x1:x2]

            # Check if the foreground is not empty
            if foreground_cropped.size == 0:
                print("Error: Cropped foreground is empty. Check your rectangle dimensions.")
                return  # Exit the function if the foreground is empty
            
            # Close the window after the rectangle is drawn and text is added
            cv2.destroyAllWindows()

# Main script
image_path = "/home/gpandit/Pictures/Screenshot from 2024-11-18 16-34-46.png"  # Replace with your image path
segmentation_map, original_image = run_segmentation(image_path)

# Ensure the segmentation map shape matches the original image
if segmentation_map.shape != original_image.shape[:2]:
    print("Error: Segmentation map shape does not match the original image shape.")
    exit(1)

foreground, background = separate_foreground_background(original_image, segmentation_map)

# Use the original image (background) for further processing
img_original = background  # Work with the original background (no resizing)

# Set up OpenCV window and mouse callback
cv2.imshow('Select Object', img_original)
cv2.setMouseCallback('Select Object', draw_rectangle)

# Wait for the 'Esc' key to exit or 'Enter' to paste the foreground
while True:
    key = cv2.waitKey(1) & 0xFF  # Wait for key press (non-blocking)
    
    # If 'Esc' key is pressed, exit
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break
    
    # If 'Enter' key is pressed, paste the cropped foreground back
    if key == 13 and foreground_cropped is not None:  # 13 is the ASCII code for the 'Enter' key
        # Paste the cropped foreground back at its original position
        x1, y1 = rect_start
        x2, y2 = rect_end

        # Ensure the cropped foreground matches the bounding box size
        foreground_resized = cv2.resize(foreground_cropped, (x2 - x1, y2 - y1))
        
        # Paste the resized foreground back into the original image
        img_original[y1:y2, x1:x2] = foreground_resized  # Place the cropped foreground back into the original image
        cv2.imshow('Select Object', img_original)  # Show the updated image

cv2.destroyAllWindows()  # Close the window after 'Esc' is pressed

