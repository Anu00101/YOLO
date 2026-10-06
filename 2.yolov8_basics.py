from ultralytics import YOLO
#import numpy
import cv2

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  


# predict on an image
detection_output = model.predict(source=r"C:\Users\anubh\.vscode\YOLO\images\1.jpg", conf=0.25, save=True) 

# Display tensor array
#print(detection_output)

# Display numpy array
#print(detection_output[0].numpy())

annotated_img = detection_output[0].plot()
cv2.imshow("YOLOv8 Detection",annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()