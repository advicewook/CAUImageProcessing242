import cv2
import pandas as pd
import subprocess
import numpy as np
import os

# Path to Tesseract executable
pytesseract_tesseract_cmd = r'C:\Users\homes\main\Recognizer_Files\Recognizer\tes\tesseract.exe'

def custom_image_to_data(image_path):
    command = [
        pytesseract_tesseract_cmd,
        image_path,
        'stdout',
        '--psm', '3',  # Page segmentation mode
        'tsv'  # Get data in TSV format
    ]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        tsv_output = result.stdout.decode('utf-8', errors='ignore')  # Decode with utf-8, ignoring errors
    except UnicodeDecodeError:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        tsv_output = result.stdout.decode('latin-1', errors='ignore')  # Fallback to latin-1 if utf-8 fails

    # Remove '\r' from each row
    rows = [row.replace('\r', '') for row in tsv_output.strip().split("\n")]
    if len(rows) <= 1:
        print("Warning: No text detected by Tesseract.")
        return pd.DataFrame()  # Return empty DataFrame if no text detected

    header = rows[0].split("\t")
    # print("Header:", header)  # Debugging: Print header to confirm column names
    data = [dict(zip(header, row.split("\t"))) for row in rows[1:] if len(row.split("\t")) == len(header)]

    df = pd.DataFrame(data)

    numeric_columns = ['left', 'top', 'width', 'height', 'conf', 'page_num']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Debugging: Print DataFrame columns
    # print("DataFrame columns:", df.columns)
    
    return df



def get_word_coordinates(data, word):
    coordinates = []
    # Debugging: Print DataFrame columns
    # print("Data columns:", data.columns)
    
    if 'text' not in data.columns:
        print("Warning: No 'text' column found in data.")
        return coordinates
    
    

    n_boxes = len(data['text'])  # Number of detected elements
    for i in range(n_boxes):
        recognized_word = data['text'][i]
        if recognized_word and word.lower() in recognized_word.lower():
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            coordinates.append((x, y, w, h))
    
    print("Number of words detected:", len(coordinates))        
    
    return coordinates


def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            grayscale_image[i, j] = (int(b) + int(g) + int(r)) // 3  # Simple average to get grayscale
    return grayscale_image

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

# def highlight_text(image, top_left, bottom_right, color=(0, 255, 255), alpha=0.5):
#     print("Highlighting text...")
#     overlay = image.copy()
#     output = image.copy()
    
#     cv2.rectangle(overlay, top_left, bottom_right, color, -1)
#     cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
#     return output

def highlight_text(image, top_left, bottom_right, color=(0, 255, 255), alpha=0.5):
    print("Highlighting text...")
    overlay = image.copy()
    output = image.copy()
    
    #draw_rectangle(overlay, top_left, bottom_right, color, -1)
    for y in range(top_left[1], bottom_right[1]):
        for x in range(top_left[0], bottom_right[0]):
            if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                overlay[y, x] = color  # Set the pixel color to highlight color
    
    # Add the overlay to the output image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if y >= top_left[1] and y < bottom_right[1] and x >= top_left[0] and x < bottom_right[0]:
                output[y, x] = (alpha * overlay[y, x] + (1 - alpha) * output[y, x]).astype(np.uint8)
    
    return output

def resize_image(image, scale_factor):
    
    # Get original image dimensions
    height, width = image.shape[:2]
    
    # Calculate new image dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Create an empty array for the resized image (set to new size based on the scale factor)
    
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    # Use bilinear interpolation to resize the image
    for y in range(new_height):
        for x in range(new_width):
            # Calculate corresponding coordinates in the original image
            original_x = x / scale_factor
            original_y = y / scale_factor

            # Separate the integer and fractional parts of the coordinates
            x1 = int(original_x)
            y1 = int(original_y)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            # Calculate the fractional parts
            dx = original_x - x1
            dy = original_y - y1

            # Compute the interpolated value using the four neighboring pixels
            
            f1 = (1 - dx) * image[y1, x1] + dx * image[y1, x2]
            f2 = (1 - dx) * image[y2, x1] + dx * image[y2, x2]
            resized_image[y, x] = int((1 - dy) * f1 + dy * f2)

    return resized_image

