import cv2
import pytesseract
from PIL import Image
import streamlit as st
import pyperclip

def capture_image():
    # Open the default camera (you can change the index to use a different camera)
    camera = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not camera.isOpened():
        st.error("Failed to open the camera")
        return None

    # Read a frame from the camera
    _, frame = camera.read()

    # Release the camera
    camera.release()

    return frame

def extract_text(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply image preprocessing techniques (e.g., resizing, denoising, thresholding)

    # Resize the image
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(threshold)

    # Use Pytesseract to extract text from the image
    text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')

    return text

def main():
    st.title("Image Text Extraction")

    # Capture an image from the camera
    if st.button("Capture Image"):
        image = capture_image()

        if image is not None:
            # Display the captured image
            st.image(image, channels="BGR", caption="Captured Image")

            # Extract text from the image
            text = extract_text(image)

            # Display the extracted text
            st.header("Extracted Text")
            st.write(text)  # Display the extracted text

            # Copy the extracted text to the clipboard
            pyperclip.copy(text)
            st.success("Text copied to clipboard!")
        else:
            st.error("Failed to capture the image")

if __name__ == "__main__":
    main()


