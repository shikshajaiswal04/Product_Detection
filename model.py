import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# Change this to your working directory
os.chdir(r'c:\Users\jiwan\OneDrive\Desktop\yolo')

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for YOLOv8.')

# 1. Train the YOLOv8 Model
def train_yolo_model():
    # Load the YOLOv8 model
    model = YOLO('yolov8s.pt')  # Load a pretrained YOLOv8 model (small version)

    # Train the model with your custom dataset (make sure 'data.yaml' is set up correctly)
    model.train(
        data='data.yaml',
        epochs=25,   # Number of epochs
        imgsz=224,   # Image size
        device=device,  # Use GPU if available
        plots=True  # Show plots for results
    )
    print("Training Complete.")


# 2. Predict on New Images
def predict_on_new_images():
    # Load the trained model from the best weights
    model = YOLO(r'runs\detect\train\weights\best.pt')

    # Predict on test images
    results = model.predict(source='train/images', conf=0.25, device=device)

    # Visualize results
    for result in results:
        img = result.orig_img  # Original image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract detected class names and draw boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{result.names[cls]}: {conf:.2f}"
            
            # Draw rectangle and label on the image
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle
            img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Draw label

        # Display the image with a title
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Detected Objects: {', '.join([result.names[int(cls)] for cls in result.boxes.cls])}")
        plt.axis('off')
        plt.show()



# Main function to run training and prediction
def main():
    print("Starting YOLOv8 Training...")
    train_yolo_model()  # Train the model
    print("Training completed. Now predicting on test images...")
    predict_on_new_images()  # Predict on new images

if __name__ == "__main__":
    main()
