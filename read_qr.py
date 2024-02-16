#pip3 install 
# opencv-python pyzbar
# brew install zbar

import cv2
from pyzbar.pyzbar import decode

def read_qr_code(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Decode the QR code
    decoded_objects = decode(img)
    
    # Print the decoded data
    for obj in decoded_objects:
        print("Data:", obj.data.decode("utf-8"))

# Example usage
image_path = "QR_test2.png" # Replace with the path to your image file
read_qr_code(image_path)
