import cv2
import pandas as pd
import subprocess
import numpy as np
import os

# Path to Tesseract executable
pytesseract_tesseract_cmd = r'C:\Users\JoEonWook\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def run_tesseract(image_path, output_base, config='makebox'):
    box_file = f"{output_base}.box"
    command = [
        pytesseract_tesseract_cmd,
        image_path,
        output_base,
        config
    ]
    subprocess.run(command, check=True)
    return box_file

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

def image_to_boxes_dict(image_path):
    output_base = "output"
    boxes_path = run_tesseract(image_path, output_base, config='makebox')
    return parse_boxes(boxes_path)

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


def convert_to_grayscale(image):
    height, width, _ = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            grayscale_image[i, j] = (int(b) + int(g) + int(r)) // 3  # Simple average to get grayscale
    return grayscale_image

def median_filter(image, kernel_size=3):
    half_k = kernel_size // 2
    padded_image = np.pad(image, half_k, mode='edge')
    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            local_region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.median(local_region)
    
    return filtered_image


def adaptive_threshold(image, block_size=15, C=1):
    padded_image = np.pad(image, block_size // 2, mode='reflect')
    thresholded = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_region = padded_image[i:i+block_size, j:j+block_size]
            local_threshold = np.mean(local_region) - C
            thresholded[i, j] = 255 if image[i, j] > local_threshold else 0
    return thresholded


def simple_threshold(image, threshold_value=128):
    height, width = image.shape
    thresholded_image = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            if image[i, j] > threshold_value:
                thresholded_image[i, j] = 255
            else:
                thresholded_image[i, j] = 0
                
    return thresholded_image

def resize_image(image, scale_percent):
    height, width = image.shape
    new_height = int(height * scale_percent / 100)
    new_width = int(width * scale_percent / 100)
    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            x = i / (new_height / height)
            y = j / (new_width / width)
            
            x0 = int(x)
            x1 = min(x0 + 1, height - 1)
            y0 = int(y)
            y1 = min(y0 + 1, width - 1)
            
            a = x - x0
            b = y - y0
            
            resized_image[i, j] = (1 - a) * (1 - b) * image[x0, y0] + a * (1 - b) * image[x1, y0] + (1 - a) * b * image[x0, y1] + a * b * image[x1, y1]
    
    return resized_image



#def preprocess_image(image_path, script_dir):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply thresholding
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Deskew the image
#     coords = np.column_stack(np.where(binary > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = binary.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
#     # Resize the image
#     resized = cv2.resize(deskewed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    

#     preprocessed_image_path = os.path.join(script_dir, 'preprocessed_image_pro.jpg')
#     cv2.imwrite(preprocessed_image_path, resized)   

#     return preprocessed_image_path

