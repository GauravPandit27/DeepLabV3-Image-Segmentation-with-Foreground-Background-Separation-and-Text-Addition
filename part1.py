import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101

# Load the pre-trained DeepLabV3 model
def load_model():
    model = deeplabv3_resnet101(pretrained=True).eval()
    return model

# Perform segmentation using DeepLabV3
def segment_image(model, image):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Resize segmentation mask back to original image size
    mask = cv2.resize(output_predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

# Mouse callback for drawing bounding box
bbox_start = None
bbox_end = None
drawing = False
def draw_bbox(event, x, y, flags, param):
    global bbox_start, bbox_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        bbox_start = (x, y)
        bbox_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox_end = (x, y)

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

# Main function
def main():
    global bbox_start, bbox_end, drawing

    # Load the image
    image_path = input("Enter the path to the image: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Load the segmentation model
    print("Loading model...")
    model = load_model()

    # Perform segmentation
    print("Segmenting image...")
    mask = segment_image(model, image)

    # Extract the foreground
    foreground = image.copy()
    foreground[mask != 15] = 0

    # Extract the background
    background = image.copy()
    background[mask == 15] = 0

    # Display the segmented image for user interaction
    print("Draw a bounding box on the image.")
    display_image = background.copy()
    cv2.namedWindow("Segmented Image")
    cv2.setMouseCallback("Segmented Image", draw_bbox)

    while True:
        temp_image = display_image.copy()
        if bbox_start and bbox_end:
            cv2.rectangle(temp_image, bbox_start, bbox_end, (0, 255, 0), 2)
        cv2.imshow("Segmented Image", temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break
        elif key == 27:  # Escape key
            print("Bounding box selection canceled.")
            return

    cv2.destroyAllWindows()

    # Ensure valid bounding box
    x1, y1 = bbox_start
    x2, y2 = bbox_end
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Crop and manipulate the selected region
    box_width, box_height = x2 - x1, y2 - y1
    print("Enter text to overlay:")
    text = input("Text: ")

    # Add text to the background in the bounding box area
    add_text(background, text, (x1, y1), box_width, box_height, (0, 255, 255), 2)

    # Merge the foreground back into the background
    combined_image = background.copy()
    combined_image[mask == 15] = image[mask == 15]

    # Display the final image
    cv2.imshow("Final Image", combined_image)
    print("Press 's' to save or any other key to exit.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        save_path = input("Enter the path to save the image: ")
        cv2.imwrite(save_path, combined_image)
        print("Image saved.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

