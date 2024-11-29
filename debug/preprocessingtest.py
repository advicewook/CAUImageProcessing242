import cv2
import os
from text_highlighter import convert_to_grayscale, custom_image_to_data, get_word_coordinates,  subprocess, median_filter, simple_threshold, resize_image


def preprocess_image(image, apply_grayscale = False, apply_median_filter = False, apply_threshold = False, apply_resize = False):
    if apply_grayscale:
        image = convert_to_grayscale(image)
    if apply_median_filter:
        image = median_filter(image)
    if apply_threshold:
        image = simple_threshold(image)
    if apply_resize:
        image = resize_image(image, 150)
        
    return image

def print_result_combinations(combination):
    if combination[0]:
        print("Grayscale applied", end = "/")
    else:
        print("Grayscale not applied", end = "/")
    if combination[1]:
        print("Median filter applied", end = "/")
    else:
        print("Median filter not applied", end = "/")
    if combination[2]:
        print("Threshold applied", end = "/")
    else:
        print("Threshold not applied", end = "/")
    if combination[3]:
        print("Resize applied")
    else:
        print("Resize not applied")
        
# def test_C_adaptive_threshold(image, block_size=15, C=1):

# def test_simple_or_adaptive_threshold(image, block_size=15, C=1):
    
# def test_scale_resize(image, scale_percent):

def main():
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
    
    # Define the preprocessing combinations to test
    combinations = [
    (False, False, False, False),
    (True, False, False, False),
    (True, True, False, False),
    (True, False, False, True),
    (True, False, False, True),
    (True, True, True, False),
    (True, True, False, True),
    (True, False, True, True),
    (True, True, True, True)
]
    
    for index, combination in enumerate(combinations):
        print(f"Testing combination {index + 1}/{len(combinations)}: {combination}")
        processed_image = preprocess_image(image, combination)
        
        image_name, ext = os.path.splitext(image_file_name)
        
        # Save the processed image
        preprocessed_image_path = os.path.join(script_dir, image_name + '_preprocessed_image.jpg')
        cv2.imwrite(preprocessed_image_path, processed_image)
        
        try:
            data = custom_image_to_data(preprocessed_image_path)
        
        except subprocess.CalledProcessError as e:
            print(f"Error during Tesseract execution: {e}")
            return
        except Exception as e:
            print(f"Unexpected error: {e}")
            return
        
        print_result_combinations(combination)    
        coordinates = get_word_coordinates(data, word_to_search)



    
if __name__ == "__main__":
    main()