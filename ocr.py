import streamlit as st
import cv2

def main():
    st.title("OpenCV + Streamlit Example")

    # Create a video capture object
    cap = cv2.VideoCapture(0)

    # Continuously read and display frames
    while True:
        ret, frame = cap.read()

        if ret:
            # Display the frame using Streamlit
            st.image(frame, channels="BGR")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

if __name__ == '__main__':
    main()
