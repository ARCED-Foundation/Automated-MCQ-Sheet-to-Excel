import pathlib  # Import pathlib module to handle paths
import cv2  # for reading images, draw bounding boxes
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv
import pandas as pd
import os

# Replace PosixPath with WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

# Define constants
BOX_COLORS = {
    "unchecked": (242, 48, 48),
    "checked": (0, 255, 0),
    "block": (242, 159, 5)
}
SHIFTED_BOX_COLORS = {
    "alpha": (0, 255, 255),  # Yellow
    "question": (255, 0, 0)  # Blue
}
BOX_PADDING = 2

# Load models
DETECTION_MODEL = YOLO("checkbox-detector\\models\\detector-model.pt")
CLASSIFICATION_MODEL = YOLO("checkbox-detector\\models\\classifier-model.pt")  # 0: block, 1: checked, 2: unchecked

# Load TrOCR models
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str")

def process_images_in_folder(folder_path):
    try:
        # Iterate through all files in the directory
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if the current file is an image
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                print(f"Processing image: {file_name}")
                upload_and_display_image(file_path)  # Call your function here

        print("Processing completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def rename_images_with_numbers(folder_path):
    try:
        # Get all image file names in the specified directory
        images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
        
        # Sort images alphabetically to maintain order before numbering
        images.sort()
        
        # Rename each image with a number
        for i, image_name in enumerate(images, start=1):
            old_path = os.path.join(folder_path, image_name)
            # Preserve the original file extension
            extension = os.path.splitext(image_name)[1]
            new_path = os.path.join(folder_path, f"{i}{extension}")
            os.rename(old_path, new_path)
            print(f"Renamed '{image_name}' to '{i}{extension}'")
            
        print("Renaming completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def detect_text_trocr(roi):
    """
    Detects text in the provided ROI using the TrOCR model.

    Args:
    - roi: Image region of interest as a PIL Image.

    Returns:
    - Detected text as a string.
    """
    image_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convert to PIL
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    detected_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return detected_text.strip()

def image_display(output_image):
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    plt.imshow(output_image_rgb)
    plt.axis('off')
    plt.show()

def upload_and_display_image(image_path):
    output_image = detect(image_path)
    
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    """ plt.imshow(output_image_rgb)
    plt.axis('off')
    plt.show()
     """

def draw_box(image, start, end, color, thickness):
    return cv2.rectangle(img=image, pt1=start, pt2=end, color=color, thickness=thickness)

def draw_label(image, text, start, color, font_scale, thickness):
    (text_w, text_h), _ = cv2.getTextSize(text=text, fontFace=0, fontScale=font_scale, thickness=thickness)
    image = cv2.rectangle(
        img=image,
        pt1=(start[0], start[1] - text_h - BOX_PADDING * 2),
        pt2=(start[0] + text_w + BOX_PADDING * 2, start[1]),
        color=color,
        thickness=-1
    )
    image = cv2.putText(
        img=image, 
        text=text, 
        org=(start[0] + BOX_PADDING, start[1] - BOX_PADDING), 
        fontFace=0, 
        fontScale=font_scale, 
        color=(255, 255, 255), 
        thickness=thickness
    )
    return image

def detect(image_path, shiftX=100, shiftY=0, qn_boxX=-250, qn_boxY=-90):
    image = cv2.imread(image_path)
    if image is None:
        return image
    
    file_name = os.path.basename(image_path)  # Extract the file name
    paper_number = os.path.splitext(file_name)[0]  # Use the file name without extension as "Paper Number"

    results = DETECTION_MODEL.predict(source=image, conf=0.2, iou=0.8)
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} bounding box(es).")

    if len(boxes) == 0:
        return image

    for box in boxes:
        detection_class_conf = round(box.conf.item(), 2)
        detection_class = list(BOX_COLORS)[int(box.cls)]
        if detection_class != "checked":
            continue

        start_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
        end_box = (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
        line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

        image = draw_box(image, start_box, end_box, BOX_COLORS[detection_class], line_thickness)

        # Movable label box for letters
        shifted_start_box = (start_box[0] + shiftX, start_box[1] + shiftY)
        shifted_end_box = (end_box[0] + shiftX, end_box[1] + shiftY)
        shifted_box = image[shifted_start_box[1]:shifted_end_box[1], shifted_start_box[0]:shifted_end_box[0], :]
        image = draw_box(image, shifted_start_box, shifted_end_box, SHIFTED_BOX_COLORS["alpha"], line_thickness)

        text_alpha = detect_text_trocr(shifted_box)
        print(f"Detected text in shifted box: {text_alpha}")
        image = draw_label(image, text_alpha, shifted_start_box, (0, 0, 255), line_thickness / 3, max(line_thickness - 1, 1))

        FIXED_X_QN = 110

        shifted_start_box_qn = (FIXED_X_QN, start_box[1] + qn_boxY)
        shifted_end_box_qn = (FIXED_X_QN + (end_box[0] - start_box[0]), end_box[1] + qn_boxY)
        shifted_box_Qn = image[shifted_start_box_qn[1]:shifted_end_box_qn[1], shifted_start_box_qn[0]:shifted_end_box_qn[0], :]
        image = draw_box(image, shifted_start_box_qn, shifted_end_box_qn, SHIFTED_BOX_COLORS["question"], line_thickness)

        text_qn = detect_text_trocr(shifted_box_Qn)
        print(f"Detected question number: {text_qn}")
        image = draw_label(image, text_qn, shifted_start_box_qn, (0, 255, 0), line_thickness / 3, max(line_thickness - 1, 1))
        
        first_char_text_qn = text_qn[0] if text_qn else ""
        first_char_text_alpha = text_alpha[0] if text_alpha else ""
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([paper_number, first_char_text_qn, first_char_text_alpha, 'checked'])

        image = draw_label(image, str(detection_class_conf), start_box, BOX_COLORS[detection_class], line_thickness / 3, max(line_thickness - 1, 1))

    return image

csv_file_path = "output.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Paper Number", "Text_Qn", "Text_Alpha", "Checked_Status"])

# image_path = r'checkbox-detector\images\testTwo.jpg'

rename_images_with_numbers('checkbox-detector\imgFolder')
process_images_in_folder('checkbox-detector\imgFolder')

df = pd.read_csv('output.csv')
df_sorted = df.sort_values(by='Text_Qn', ascending=True)
df_sorted.to_csv('sorted_file.csv', index=False)
