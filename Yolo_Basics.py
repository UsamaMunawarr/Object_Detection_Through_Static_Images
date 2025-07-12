######################
## Libraries Import ##
######################
from ultralytics import YOLO # Import the YOLO class from the ultralytics package
import cv2 # Import OpenCV for image processing

######################################
### YOLOv8 Model Different Weights ###
######################################
model = YOLO(r'C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\01_Yolo_Basics\Object_Detection_Through_Static_Images\yolov8n.pt')  # Load the yolov8n means nano model weights for faster inference
# model = YOLO('../YOLO_Weights/yolov8l.pt')  # Load the yolov8l means large model weights for better accuracy
# model = YOLO('../YOLO_Weights/yolov8x.pt')  # Load the yolov8x means extra large model weights for maximum accuracy
# model = YOLO('../YOLO_Weights/yolov8m.pt')  # Load the yolov8m means medium model weights for balanced performance
# model = YOLO('../YOLO_Weights/yolov8s.pt')  # Load the yolov8s means small model weights for faster inference

##########################################################################
### Import image of bus and Detect Objects and then download the image ###
##########################################################################
results = model('https://ultralytics.com/images/bus.jpg')  # Run inference on an image
for result in results:
    result.show()  # Display the results  
# Download the image with detections
results[0].save('C:/Users/usama/OneDrive/Desktop/Compurt_Vision_Proj/01_Yolo_Basics/Object_Detection_Through_Static_Images/Outputs/detected_image_bus.png')


###############################################################################
### Import image of School boys and Detect Objects and then download the image ###
###############################################################################
results_1 = model('C:\\Users\\usama\\OneDrive\\Desktop\\Compurt_Vision_Proj\\01_Yolo_Basics\\Object_Detection_Through_Static_Images\\Images\\1.png', show=True)  # Run inference on a local image file
results_1[0].save('C:/Users/usama/OneDrive/Desktop/Compurt_Vision_Proj/01_Yolo_Basics/Object_Detection_Through_Static_Images/Outputs/detected_image_1.png')


###############################################################################
### Import image of Bike and Detect Objects and then download the image ###
###############################################################################
results_2 = model('C:/Users/usama/OneDrive/Desktop/Compurt_Vision_Proj/01_Yolo_Basics/Object_Detection_Through_Static_Images/Images/2.jpeg', show=True)  # Run inference on a local image file
for result in results_2:
    result.show()  # Display the results
results_2[0].save('C:/Users/usama/OneDrive/Desktop/Compurt_Vision_Proj/01_Yolo_Basics/Object_Detection_Through_Static_Images/Outputs/detected_image_2.png')
