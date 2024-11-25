import cv2
import os
from text_highlighter import convert_to_grayscale, apply_adaptive_threshold, custom_image_to_data, get_word_coordinates, highlight_text, subprocess, draw_rectangle

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

    grayscale_image = convert_to_grayscale(image)
    adaptive_threshold_image = apply_adaptive_threshold(grayscale_image)

    # Save the processed image
    preprocessed_image_path = os.path.join(script_dir, 'preprocessed_image.jpg')
    cv2.imwrite(preprocessed_image_path, adaptive_threshold_image)


    try:
        # Use Tesseract to extract text data without additional internal preprocessing
        data = custom_image_to_data(preprocessed_image_path)
        
        # Print the detected text data
        # print("Detected text data:")
        # print(data)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during Tesseract execution: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Search for a specific word and get its coordinates
    coordinates = get_word_coordinates(data, word_to_search)

    # Define the highlight color in BGR format and alpha transparency
    highlight_color = (167, 198, 237)  
    alpha = 0.7  # Transparency level

    # Highlight found words in the image
    padding = 1  # Padding for word boxes
    for (x, y, w, h) in coordinates:
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        image = highlight_text(image, top_left, bottom_right, color=highlight_color, alpha=alpha)

        # draw_rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=1)

    # Display the processed image with highlighted text
    
    cv2.imshow('Highlighted Words', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    filename = image_file_name
    image_result, ext = os.path.splitext(filename)
    result_image_path = os.path.join(script_dir, image_result + '_' + word_to_search+ '_result'+ ext)
    cv2.imwrite(result_image_path, image)

    
    # cv2.imshow('Highlighted Words', image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

