import os
import cv2
import json
import argparse
import numpy as np
from pyzbar.pyzbar import decode

def load_template(template_path):
    with open(template_path, 'r') as file:
        return json.load(file)

def read_qr_code(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Decode the QR code
    decoded_objects = decode(img)
    
    # Return the decoded data
    qr_data = []
    for obj in decoded_objects:
        qr_data.append(obj.data.decode("utf-8"))
    return qr_data
    
def assess_filled(image, checkboxes, threshold=127.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []
    for checkbox in checkboxes:
        contour = checkbox["contour"]
        contour = np.array(contour)  # Convert contour to NumPy array
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        mean_intensity = cv2.mean(roi)[0]
        print(f"Checkbox contour: {contour}")
        print(f"Bounding box (x, y, w, h): ({x}, {y}, {w}, {h})")
        print(f"Mean intensity: {mean_intensity}")
        filled = mean_intensity < threshold
        print(f"filled: {filled}")
        checkbox["intensity"] = mean_intensity  # Assign 'intensity' directly to the checkbox dictionary
        results.append({"bbox": [x, y, w, h], "filled": filled, "intensity": mean_intensity})
    return results

def draw_boxes(image, checkboxes, threshold_results):
    for checkbox, threshold_result in zip(checkboxes, threshold_results):
        contour = checkbox["contour"]

        # Get the coordinates of the first point of the contour
        first_point = contour[0]
        x, y = first_point

        # Draw the contour on the image
        contour_np = np.array(contour)
        color = (0, 255, 0) if threshold_result else (0, 0, 255)

        # Draw the contour on the image with the determined color
        cv2.drawContours(image, [contour_np], -1, color, 2)

        # Add text annotations
        if "intensity" not in checkbox:
            checkbox["intensity"] = 0  # Initialize intensity to zero if not already present
        intensity = checkbox["intensity"]
        position_text = f"Position: ({x}, {y})"
        intensity_text = f"{intensity:.2f}"
        label_text = f"{checkbox['label']+checkbox['subquestion']}"
        threshold_result_text = f"{threshold_result}"

        #cv2.putText(image, position_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black color

        #cv2.putText(image, threshold_result_text, (x+10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black color
        cv2.putText(image, intensity_text, (x+10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black color
        cv2.putText(image, label_text, (x+10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black color
                                       
def process_image(image_path, template, output_dir, threshold):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    checkboxes = []
    for item in template:
        checkboxes.append({
            "contour": item["contour"], 
            "label": item["question"],  # Use the "question" field as the label
            "subquestion": item["subquestion"],
            "question_type": item["question_type"]
        })

    threshold_results = [result["filled"] for result in assess_filled(image, checkboxes, threshold)]

    # Read QR code data
    qr_data = read_qr_code(image_path)

    results = []
    for checkbox, threshold_result in zip(checkboxes, threshold_results):
        results.append({
            "question": checkbox["label"], 
            "subquestion": checkbox["subquestion"],
            "question_type": checkbox["question_type"],
            "threshold_result": threshold_result,
            "intensity": checkbox.get("intensity", 0),
            "used_threshold": threshold,
            "qr_data": qr_data  # Add the QR code data to each result
        })

    draw_boxes(image, checkboxes, threshold_results)
    annotated_image_path = os.path.join(output_dir, f"annotated_{image_name}")
    cv2.imwrite(annotated_image_path, image)
    return image_name, results

def main():
    parser = argparse.ArgumentParser(description='Batch process images with a template.')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('-tm', '--template', type=str, required=True, help='Path to the template JSON file.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the folder to save results.')
    parser.add_argument('-t', '--threshold', type=float, default=127.5, help='Threshold for checkbox detection')
    args = parser.parse_args()

    # Load template
    template = load_template(args.template)

    # Dictionary to store results for all images
    all_results = {}

    # Ensure the output directory exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Process each image
    for image_name in os.listdir(args.input_folder):
        image_path = os.path.join(args.input_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {image_path}")
            filename, results = process_image(image_path, template, args.output_folder, args.threshold)
            all_results[filename] = results

    # Write all results to a single JSON file
    all_results_file_path = os.path.join(args.output_folder, "all_results.json")
    with open(all_results_file_path, 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()