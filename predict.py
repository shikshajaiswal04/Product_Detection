import torch
import os
import cv2
import numpy as np
from tkinter import filedialog, Tk, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import pytesseract 

capture_flag = False
current_frame = None

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for YOLOv8.')

# Load the model once globally
model = YOLO(r'best.pt')

# Function to detect objects in the image
def detect_objects(image):
    results = model.predict(source=image, conf=0.25, device=device)
    return results

# Function to analyze the quality of detected fruits
def analyze_fruit_quality(image, fruit_name):
    quality_status = ""
    
    if fruit_name == "Apple":
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for a good apple
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Create masks for red and green apples
        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # Check the counts of red and green pixels
        if cv2.countNonZero(mask_red) > 100 or cv2.countNonZero(mask_green) > 100:
            quality_status = "Good"
        else:
            quality_status = "Bad (discolored)"
    
    elif fruit_name == "Carrot":
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range for a fresh carrot (bright orange)
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])

        # Define color range for white spots or discoloration (indicating dehydration or spoilage)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])

        # Create masks for orange and white areas
        mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        # Check the counts of orange and white pixels
        orange_count = cv2.countNonZero(mask_orange)
        white_count = cv2.countNonZero(mask_white)

        # Freshness logic for carrots
        if orange_count > 200 and white_count < 100:
            quality_status = "Good"
        else:
            quality_status = "Bad (dehydrated or discolored)"
    
    return quality_status

# Function to extract price and expiry date using OCR
def extract_price_expiry(image):
    ocr_result = pytesseract.image_to_string(image)
    price = None
    expiry_date = None

    # Simple checks for price and expiry format (you can improve this regex)
    for line in ocr_result.split('\n'):
        line = line.strip()
        if "price" in line.lower():
            price = line  # Assuming the line contains price
        elif "expiry" in line.lower() or "exp" in line.lower():
            expiry_date = line  # Assuming the line contains expiry date

    return price, expiry_date

# Function to display results in the GUI
def display_results(results):
    detected_count = {}  # Dictionary to store detected class names and their counts
    quality_info = []  # List to store quality information
    ocr_info = []  # List to store OCR information

    for result in results:
        img = result.orig_img  # Original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # For displaying in Tkinter

        # Extract detected class names and draw boxes
        for box in result.boxes:
            cls = int(box.cls[0])  # Class index
            label = result.names[cls]

            # Analyze fruit quality if the detected object is an apple or banana
            fruit_quality = ""
            box_color = (255, 0, 0)  # Default to red

            if label in ["Apple", "Carrot"]:
                fruit_quality = analyze_fruit_quality(img, label)
                quality_info.append(f"{label}: {fruit_quality}")  # Append quality information

                # Set box color based on quality
                if fruit_quality == "Good":
                    box_color = (0, 255, 0)  # Green for good quality
                elif fruit_quality.startswith("Bad"):
                    box_color = (255, 0, 0)  # Red for bad quality

            # Count occurrences of each detected class
            if label in detected_count:
                detected_count[label] += 1
            else:
                detected_count[label] = 1

            # Draw bounding box (for visualization)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            img_rgb = cv2.rectangle(img_rgb, (x1, y1), (x2, y2), box_color, 2)  # Draw rectangle with appropriate color

            # Perform OCR on the original image to get price and expiry date
            price, expiry_date = extract_price_expiry(img)

            # Create the label with quality status if available
            label_text = f"{label} ({fruit_quality})" if fruit_quality else label

            # Add price and expiry date if available
            if price:
                label_text += f"\nPrice: {price}"
            if expiry_date:
                label_text += f"\nExpiry: {expiry_date}"

            # Draw label above the bounding box
            img_rgb = cv2.putText(img_rgb, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

        # Convert to PIL Image and display in the GUI
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)

        result_label.config(image=imgtk)
        result_label.image = imgtk

    # Update the detected products and their counts
    result_text_content = "Detected Products:\n"
    total_count = 0

    for item, count in detected_count.items():
        result_text_content += f"{item}: {count}\n"
        total_count += count

    result_text_content += f"Total Count: {total_count}\n\n"
    
    # Display quality information
    if quality_info:
        result_text_content += "Quality Information:\n" + "\n".join(quality_info) + "\n"

    # Display OCR information (price and expiry date)
    if ocr_info:
        result_text_content += "\nOCR Information:\n" + "\n".join(ocr_info)

    result_text.config(text=result_text_content)

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image = cv2.imread(file_path)
        results = detect_objects(image)
        display_results(results)

# Function to take a snapshot from the webcam
SAVE_PATH = "captured_image.jpg"
def take_snapshot(frame):
    # Prompt user to select file path to save the image
    cv2.imwrite(SAVE_PATH, frame)  # Save the captured image directly
    # Process the captured frame for object detection
    results = detect_objects(frame)
    display_results(results)
    cap.release()
    cv2.destroyAllWindows()

# Function to capture image from the phone camera
def capture_image():
    global cap
    url = "http://192.168.42.21:8080/video"
    cap = cv2.VideoCapture(url)  # Open the phone camera stream
    if not cap.isOpened():
        print("Could not open phone camera stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Show the current frame using OpenCV
        cv2.imshow('Phone Camera', frame)

        # Wait for 's' key to take a snapshot
        if cv2.waitKey(1) & 0xFF == ord('s'):
            take_snapshot(frame)

        # Exit if 'q' is pressed
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to capture from phone camera stream
def phone_camera():
    ip_stream_url = "http://192.168.42.21:8080/video"  # Update with your IP and port from the app
    cap = cv2.VideoCapture(ip_stream_url)

    if not cap.isOpened():
        print("Error opening video stream")
        return

    def process_frame():
        global capture_flag

        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (640, 480)) 
            

            # Perform detection on the current frame
            results = detect_objects(frame_resized)
            display_results(results)

            # Capture the image if the capture flag is set
            if capture_flag:
                save_image(frame)
                capture_flag = False  # Reset the flag after capturing

        # Call this function again after a short delay (to continue real-time processing)
        result_label.after(1, process_frame)

    process_frame()  # Start the real-time detect

def key_event(event):
    global capture_flag
    if event.keysym == 'c':  # If the "C" key is pressed
        capture_flag = True  # Set the capture flag to True

# Function to save the current frame to a file
def save_image(frame):
    directory = "captured_images"
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    file_name = os.path.join(directory, "captured_image.jpg")
    cv2.imwrite(file_name, frame)
    print(f"Image saved as {file_name}")

# Main function to set up the GUI
def main():
    global result_label, result_text

    root = Tk()
    root.title("Fruit Quality and Pricing Detection")

    # Set a fixed size for the window (e.g., 800x600)
    root.geometry("800x800")
    
    # Disable resizing the window
    root.resizable(width=False, height=False)

    # Create a frame to hold the image and result
    frame = Frame(root)
    frame.pack(expand=True, fill="both")

    # Create text label to display detection results at the top
    result_text = Label(frame, text="Detected Products will be displayed here.", wraplength=400, anchor="center", justify="left")
    result_text.pack(side="top", pady=10)  # Ensure it's always at the top

    # Create upload button
    upload_button = Button(frame, text="Upload Image", command=upload_image)
    upload_button.pack(pady=5)

    capture_button = Button(frame, text="capture Image", command=capture_image)
    capture_button.pack(pady=5)

    # Create capture button to start the webcam
    capture_button = Button(frame, text="Live Detecion", command=phone_camera)
    capture_button.pack(pady=5)

    # Create result label to display image
    result_label = Label(frame)
    result_label.pack(expand=True, fill="both")

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
