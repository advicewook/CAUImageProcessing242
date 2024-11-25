import cv2
import pandas as pd
import subprocess
import numpy as np
import os

# Path to Tesseract executable
pytesseract_tesseract_cmd = r'C:\Users\JoEonWook\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Function to run Tesseract and generate a .box file
def run_tesseract(image_path, output_base, config='makebox'):
    box_file = f"{output_base}.box"
    command = [
        pytesseract_tesseract_cmd,
        image_path,
        output_base,
        config
    ]
    
    print("Running Tesseract command...")
    print(f"Command: {' '.join(command)}")  # Print the full command for debugging
    subprocess.run(command, check=True, text=True, encoding='utf-8'), 
    
    print("Running Tesseract command...")
    print(f"Command: {' '.join(command)}")

    return box_file

# Function to parse .box files into a pandas DataFrame
def parse_boxes(boxes_path):
    data = []
    with open(boxes_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 6:
                data.append({
                    'char': parts[0],
                    'left': int(parts[1]),
                    'bottom': int(parts[2]),
                    'right': int(parts[3]),
                    'top': int(parts[4]),
                    'page_num': int(parts[5])
                })
    return pd.DataFrame(data)

# Function to extract character positions using Tesseract's .box output
def image_to_boxes_dict(image_path):
    output_base = "output"
    boxes_path = run_tesseract(image_path, output_base, config='makebox')
    return parse_boxes(boxes_path)

# Function to replace pytesseract.image_to_data and extract text data
def custom_image_to_data(image_path):
    command = [
        pytesseract_tesseract_cmd,
        image_path,
        'stdout',
        '--psm', '3',  # Page segmentation mode
        'tsv'  # Get data in TSV format
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    tsv_output = result.stdout

    # Parse TSV output into a pandas DataFrame
    rows = tsv_output.strip().split("\n")
    header = rows[0].split("\t")
    if len(rows) <= 1:
        print("Warning: No text detected by Tesseract.")
        return pd.DataFrame()  # Return empty DataFrame if no text detected
    
    data = [dict(zip(header, row.split("\t"))) for row in rows[1:] if len(row.split("\t")) == len(header)]
    df = pd.DataFrame(data)

    # Convert numeric columns to appropriate types
    numeric_columns = ['left', 'top', 'width', 'height', 'conf', 'page_num']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# Function to search for a word and get its bounding box from image data
def get_word_coordinates(data, word):
    coordinates = []
    if 'text' not in data.columns:
        print("Warning: No 'text' column found in data.")
        return coordinates
    
    n_boxes = len(data['text'])  # Number of detected elements
    for i in range(n_boxes):
        recognized_word = data['text'][i]
        if recognized_word and word.lower() in recognized_word.lower():
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            coordinates.append((x, y, w, h))
    return coordinates

# Convert to grayscale without using built-in functions
def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            grayscale_image[i, j] = (int(b) + int(g) + int(r)) // 3  # Simple average to get grayscale
    return grayscale_image

# Apply adaptive thresholding without using built-in functions
def apply_adaptive_threshold(image, block_size=15, C=1):
    height, width = image.shape
    adaptive_thresh_image = np.zeros((height, width), dtype=np.uint8)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        for j in range(half_block, width - half_block):
            local_region = image[i - half_block:i + half_block + 1, j - half_block:j + half_block + 1]
            local_thresh = np.mean(local_region) - C
            if image[i, j] > local_thresh:
                adaptive_thresh_image[i, j] = 255
            else:
                adaptive_thresh_image[i, j] = 0

    return adaptive_thresh_image

# Custom function to draw a rectangle manually
def draw_rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=1):
    x1, y1 = top_left
    x2, y2 = bottom_right

    if thickness <= 0:
        thickness = 1

    # Draw top and bottom lines
    for t in range(thickness):
        if y1 + t < image.shape[0]:
            image[y1 + t, x1:x2] = color  # Top line
        if y2 + t < image.shape[0]:
            image[y2 + t, x1:x2] = color  # Bottom line

    # Draw left and right lines
    for t in range(thickness):
        if x1 + t < image.shape[1]:
            image[y1:y2, x1 + t] = color  # Left line
        if x2 + t < image.shape[1]:
            image[y1:y2, x2 + t] = color  # Right line
            
# Function to highlight text on the image
def highlight_text(image, top_left, bottom_right, color=(0, 255, 255), alpha=0.5):
    overlay = image.copy()
    output = image.copy()
    
    # Draw the rectangle on the overlay image
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    
    # Apply the overlay on the output image with transparency
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ask the user for the image file name
image_file_name = input("Enter the image file name (e.g., 'Sample2.png'): ")
word_to_search = input("Enter the word to search: ")

# Construct the full image path
image_path = os.path.join(script_dir, image_file_name)

# Load the image using OpenCV
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found! Check the file name and ensure it is in the same directory as this script.")

grayscale_image = convert_to_grayscale(image)
adaptive_threshold_image = apply_adaptive_threshold(grayscale_image)

# Save the processed image
preprocessed_image_path = os.path.join(script_dir, 'preprocessed_image.jpg')
cv2.imwrite(preprocessed_image_path, adaptive_threshold_image)

# Use Tesseract to extract text data without additional internal preprocessing
data = custom_image_to_data(preprocessed_image_path)

# Search for a specific word and get its coordinates
# word_to_search = "spain" # Replace with the word you want to find
coordinates = get_word_coordinates(data, word_to_search)

# Define the highlight color in BGR format and alpha transparency
highlight_color = (0, 255, 255)  # Yellow in BGR format
alpha = 0.5  # Transparency level

# Highlight found words in the image
padding = 1  # Padding for word boxes
for (x, y, w, h) in coordinates:
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    draw_rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=1)
    
    image2 = highlight_text(image, top_left, bottom_right, color=highlight_color, alpha=alpha)


# Display the processed image with both character-level and word-level boxes
cv2.imshow('Highlighted Words', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Highlighted Words', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
