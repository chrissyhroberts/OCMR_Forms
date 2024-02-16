import cv2
import numpy as np
import argparse
import glob
import os

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, maxWidth, maxHeight):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def apply_padding(warped_image, padding):
    if padding > 0:
        h, w = warped_image.shape[:2]
        return warped_image[padding:h-padding, padding:w-padding]
    return warped_image

def process_images(input_dir, output_dir, width, height, epsilon, padding):
    for imagePath in glob.glob(input_dir + '/*.jpg') + glob.glob(input_dir + '/*.png'):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        filename = os.path.basename(imagePath)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                warped = four_point_transform(image, approx.reshape(4, 2), width, height)
                cropped_warped = apply_padding(warped, padding)
                cv2.imwrite(os.path.join(output_dir, filename), cropped_warped)
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", type=int, default=1926, help="Width of the warped image")
    parser.add_argument("-ht", "--height", type=int, default=1096, help="Height of the warped image")
    parser.add_argument("-e", "--epsilon", type=float, default=0.02, help="Epsilon for contour approximation")
    parser.add_argument("-p", "--padding", type=int, default=50, help="Padding to crop inside the edges after warp")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory containing images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for warped images")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_images(args.input_dir, args.output_dir, args.width, args.height, args.epsilon, args.padding)
