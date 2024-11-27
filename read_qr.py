import cv2
from pyzbar.pyzbar import decode
import argparse

def read_qr_code(image_path):
    """
    Reads an image, detects QR codes, and extracts their contents.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None. Prints decoded QR code data to the console.
    """
    # Try to load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from path '{image_path}'. Please check the file path.")
        return
    
    # Attempt to decode QR codes
    decoded_objects = decode(img)
    if not decoded_objects:
        print("No QR code found in the image.")
        return
    
    # Print each QR code's decoded data
    for i, obj in enumerate(decoded_objects):
        try:
            decoded_data = obj.data.decode('utf-8')  # Decode data to UTF-8
            print(f"QR Code {i + 1}: {decoded_data}")
        except UnicodeDecodeError:
            print(f"QR Code {i + 1}: Unable to decode data.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect and decode QR codes from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file containing the QR code.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the provided image path
    read_qr_code(args.image_path)
