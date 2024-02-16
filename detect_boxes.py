########################################################################################################
########################################################################################################
# OMR / OCR system for reading data from paper checkbox forms and converting to tidy data set
#
# Workflow
#
# The script accepts command-line arguments specifying the image path, mode of operation 
# (1 for detecting checkboxes, 2 for detecting checked boxes), 
# fixed dimensions for cropping, threshold for checkbox detection, and form name for naming output files
#
# The basic function is that a form will have some text marker/anchor, like "Ꭶ"
# The marker would be placed in the corners of a bounding box
# The script finds the corners, then crops the area of interest to this bounding box
# and performs checkbox detection within that zone
# 
# The text anchors can be quite small, but obviously if your camera/scanner is not so good
# you'll run the risk of things failing if the text anchors are too small.
#
# Based on the mode, it either (1) detects checkboxes and saves their positions, 
# or (2) detects whether checkboxes are checked and records the results in a CSV file.
#
# The script supports image preprocessing, such as cropping to a fixed size after detecting text markers,
# to standardize the input for checkbox detection.
#
# It uses OCR to extract text from images, which is essential for finding text markers and 
# extracting specific text based on patterns.
#
# Prerequisites
# Install Python packages: pip3 install opencv-python pytesseract numpy
# Install Tesseract-OCR: Follow the installation guide for your operating system 
# from the official Tesseract GitHub page.
# Download the cherokee training data from https://tesseract-ocr.github.io/tessdoc/tess3/Data-Files.html
# Install cherokee data in to tessdata folder
# on mac cd /usr/local/share/tessdata/    
#
#
# Usage Example:
# Mode 1: Detecting Checkboxes
#
# For detecting checkboxes, you would run the script in mode 1. 
# Assume you want the cropped image to be 800x600 pixels, and the form name is my_form.
#
# RUN THIS COMMAND : python Form_OCMR.py form_image.jpg 1 800 600 0 my_form
#
# This command does the following:
#
#         Reads form_image.jpg.
#         Detects and crops the image to a fixed size of 800x600 pixels based on "Ꭶ" markers.
#         Detects checkboxes in the cropped image.
#         Saves an annotated image showing detected checkboxes with bounding boxes.
#         Outputs positions of checkboxes in a JSON file named my_form_template.json.
#
#
# After you have created the template json file, you should open it in a text editor
#
# Mode 2: Identifying Checked Checkboxes
#
# To identify which checkboxes are checked, you'll run the script in mode 2. 
# Let's use the same dimensions and assume you've decided on a threshold value of 127.5 
# for checkbox detection (this value might need adjustment based on your images).
# 
# RUN THIS COMMAND : python Form_OCMR.py form_image_filled.jpg 2 800 600 127.5 my_form
#
# This command will:
#
# Read form_image_filled.jpg.
# Crop the image to 800x600 pixels based on "Ꭶ" markers.
# Use the checkbox positions defined in my_form_template.json to identify which checkboxes are checked 
# based on the specified threshold.
# Save an annotated image showing the checked status of each checkbox.
# Append the results (image name, extracted text, and checkbox statuses) to a CSV file 
# named my_form_filled_data.csv. If the file doesn't exist, it creates it and includes headers based on the template.
#
#
# Notes:
# The threshold parameter in mode 2 might need to be adjusted based on the specific characteristics 
# of your form image and how filled checkboxes appear in terms of intensity.
#
# Ensure that Tesseract-OCR is correctly installed and accessible by pytesseract. 
# You might need to set the path to the Tesseract executable in your script or environment variables 
# if it's not automatically detected.
#
# The effectiveness of detecting "Ꭶ" markers and checkboxes depends on the quality 
# and clarity of the scanned form images.
########################################################################################################
########################################################################################################
# Import modules
########################################################################################################
import cv2                # OpenCV library for image processing. pip3 install opencv-python-headless
import pytesseract        # Python wrapper for Google's Tesseract-OCR Engine
import numpy as np        # Fundamental package for scientific computing with Python
import json               # Module for parsing JSON data.
import sys                # Provides access to some variables used/maintained by the Python interpreter
import os                 # Methods for operating system dependent functionality.
import csv                # Module for reading/writing CSVs
import re                 # Module for working with regular expressions
from pytesseract import Output
from PIL import Image
import argparse


########################################################################################################
########################################################################################################
# Define Functions
########################################################################################################

def save_results_to_csv(data_filename, headers, checked_results):
    file_exists = os.path.isfile(data_filename)
    with open(data_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(checked_results)

def load_template(template_path):
    with open(template_path, 'r') as file:
        return json.load(file)

def analyze_checkbox_status(image, checkbox, threshold):
    x, y, w, h = checkbox['x'], checkbox['y'], checkbox['w'], checkbox['h']
    roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary_roi == 255)
    total_pixels = binary_roi.size
    return white_pixels / total_pixels < 0.5  # Example criterion: more than half of the pixels are black

def process_filled_form(image_path, template_path, threshold):
    image = cv2.imread(image_path)
    checkboxes = load_template(template_path)
    
    for checkbox in checkboxes:
        if analyze_checkbox_status(image, checkbox, threshold):
            # This checkbox is checked; annotate the image
            x, y, w, h = checkbox['x'], checkbox['y'], checkbox['w'], checkbox['h']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Use green to indicate checked
        else:
            # Not checked; optionally, annotate with a different color
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Use red to indicate unchecked

    cv2.imshow("Processed Form", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Optionally, save the annotated image and/or results to a CSV file



def remove_black_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return image



########################################################################################################
# detect_checkboxes 
#
#      Detects checkboxes in an image, draws bounding boxes around them, 
#      and saves their positions along with the image.
#
########################################################################################################

def detect_checkboxes(image, output_image_path, template_path, min_area, max_area, epsilon):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []
    min_area = min_area  # Minimum area of a checkbox
    max_area = max_area  # Maximum area of a checkbox

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        
        # Check if the approximated polygon could be a checkbox
        area = cv2.contourArea(approx)
        if len(approx) == 4 and min_area < area < max_area:  # Adjust this condition as needed
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            checkboxes.append({
                "x": x, 
                "y": y, 
                "w": w, 
                "h": h,
                "label": "label_here"
            })

    # Sort the checkboxes by their y-coordinate, and then by their x-coordinate
    checkboxes = sorted(checkboxes, key=lambda k: (k['y'], k['x']))

    for checkbox in checkboxes:
        label_position = (checkbox["x"], checkbox["y"]-10)
        cv2.putText(image, f"({checkbox['x']}, {checkbox['y']})", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(output_image_path, image)

    # Save the checkbox positions to a JSON file
    with open(template_path, 'w') as f:
        json.dump(checkboxes, f, indent=4)

########################################################################################################
# detect_checked_boxes 
#
#      Determines whether the detected checkboxes are checked, 
#      based on intensity thresholding, and annotates the image accordingly. 
#
########################################################################################################

def detect_checked_boxes(image, image_path, template_path, threshold=127.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract specific text surrounded by ± using OCR
    ocr_text = pytesseract.image_to_string(gray)
    print("Full OCR Text:", ocr_text)  # Print the entire extracted text for debugging
    matched_text = re.search("~(.*?)~", ocr_text)
    extracted_text = matched_text.group(1) if matched_text else "N/A"

    # Load checkbox template
    with open(template_path, 'r') as f:
        checkboxes = json.load(f)

    results = [os.path.basename(image_path), extracted_text]  # Start with the filename and the extracted text

    for box in checkboxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        
        roi = gray[y:y+h, x:x+w]
        mean_intensity = cv2.mean(roi)[0]

        if mean_intensity < threshold:
            check_status = "pos"
        else:
            check_status = "neg"

        results.append(check_status)

        label_text = f"{check_status} ({x}, {y})"
        label_text_2 = f"Mean:{mean_intensity:.2f}"


        if check_status == "pos":
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.putText(image, label_text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, label_text_2, (x-10, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    output_path = generate_annotated_filename(image_path)
    cv2.imwrite(output_path, image)

    return results


########################################################################################################
# generate_annotated_filename 
#
#      Generates a filename for saving annotated images by 
#      appending a suffix to the original filename
#
########################################################################################################

def generate_annotated_filename(original_path, suffix="_annotated"):
    base_name, ext = os.path.splitext(original_path)
    return f"{base_name}{suffix}{ext}"



########################################################################################################
########################################################################################################
# Main workflow
########################################################################################################
########################################################################################################



# Main section with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OMR/OCR system for reading data from paper checkbox forms.')

    # Optional arguments for all inputs including mode and image path with abbreviated versions
    parser.add_argument('-i', '--image_path', type=str, required=False, help='Path to the form image')
    parser.add_argument('-m', '--mode', type=int, required=True, choices=[1, 2], help='Mode of operation (1 for detecting checkboxes, 2 for detecting checked boxes)')
    parser.add_argument('-w', '--fixed_width', type=int, default=800, help='Fixed width for cropping')
    parser.add_argument('-ht', '--fixed_height', type=int, default=600, help='Fixed height for cropping')
    parser.add_argument('-t', '--threshold', type=float, default=127.5, help='Threshold for checkbox detection')
    parser.add_argument('-f', '--form_name', type=str, default='my_form', help='Form name for naming output files')
    parser.add_argument('-ma', '--min_area', type=int, default=1000, help='Minimum area of a checkbox')
    parser.add_argument('-xa', '--max_area', type=int, default=20000, help='Maximum area of a checkbox')
    parser.add_argument('-e', '--epsilon', type=float, default=0.02, help='Epsilon value for contour approximation')
    parser.add_argument('-id', '--input_dir', type=str, required=False, help='Directory path to the input images')
    parser.add_argument('-o', '--output_dir', type=str, required=False, help='Directory path for saving output images')
    parser.add_argument('-tm', '--template', type=str, required=False, help='Path to the form template json')

    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image_path)
    

    # Define filenames based on the input image name
    form_name = os.path.splitext(os.path.basename(args.image_path))[0]
    template_filename = f"{form_name}_template.json"
    data_filename = f"{form_name}_data.csv"

    if args.mode == 2 and not args.template:
        parser.error("The -tm/--template argument is required for mode 2.")

    # Process based on the mode
    if args.mode == 1:
        output_image_path = generate_annotated_filename(args.image_path)
        image_without_border = remove_black_border(image)
        detect_checkboxes(image_without_border, output_image_path, template_filename, args.min_area, args.max_area, args.epsilon)

    elif args.mode == 2:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        template = load_template(args.template)  # Load the template JSON once outside the loop

        all_checked_results = []  # Initialize a list to store results from all images

        for filename in os.listdir(args.input_dir):
            filepath = os.path.join(args.input_dir, filename)
            if os.path.isfile(filepath) and filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(filepath)
                if image is not None:
                    form_name = os.path.splitext(os.path.basename(filepath))[0]  # Use filepath, not args.image_path
                    image_without_border = remove_black_border(image)
                    checked_boxes = []

                    for checkbox in template:
                        status = "Checked" if analyze_checkbox_status(image_without_border, checkbox, args.threshold) else "Unchecked"
                        checked_boxes.append((checkbox['label'], status))

                    # Optionally annotate image here based on checked_boxes information

                    annotated_image_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + "_mode2" + os.path.splitext(filename)[1])
                    cv2.imwrite(annotated_image_path, image_without_border)

                    all_checked_results.extend(checked_boxes)  # Append results for this image to the overall list

        # Define CSV file path and headers
        csv_file_path = os.path.join(args.output_dir, "checked_boxes_results.csv")
        headers = ['Checkbox Label', 'Status']

        # Save all results to CSV
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for result in all_checked_results:
                writer.writerow(result)

    print(f"Batch processing complete. Processed images and results saved to {args.output_dir}.")