'''
This program file is for:
	- generating YOLO-format labels
	- moving image and labels files into respective folder
	- plotting label on image, for testing purposes
	- training YOLO model
Uncomment the functions at the end to use their function
'''

##################################################################
# (1) For training custom dataset, creating a txt file from json file

import json
import os
import cv2
import matplotlib.pyplot as plt
WIDTH = 320
HEIGHT = 180
folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 8"

# Reading JSON files
def get_json(file_path):
    '''
    Returns the number of passengers and bounding boxes from a JSON file.
    '''
    file = open(file_path)
    data = json.load(file)

    passenger_count = 0
    bounding_boxes = []
    for capture in data['captures']:
        if 'annotations' in capture:
            for annotation in capture['annotations']:
                if 'id' in annotation and annotation['id'] == 'bounding box':
                    try:
                        # Get bounding box in 'xywh' format
                        boxes = annotation['values']
                        for box in boxes:
                            bounding_box = []
                            bounding_box.append((box['origin'][0] + box['dimension'][0] / 2) / WIDTH)
                            bounding_box.append((box['origin'][1] + box['dimension'][1] / 2) / HEIGHT)
                            bounding_box.append(box['dimension'][0] / WIDTH)
                            bounding_box.append(box['dimension'][1] / HEIGHT)
                            bounding_boxes.append(bounding_box)
                    except:
                        break
    return bounding_boxes

def create_label():
    # Loop through main folder for annotation from JSON file
    counter = 0
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name).replace("/", "//")

            # Create a new text file in the subfolder directory
            txt_file_path = os.path.join(subfolder_path, 'output.txt')

            # Loop through files
            for file_name in os.listdir(subfolder_path):
                if file_name == "step0.frame_data.json":
                    file_path = os.path.join(subfolder_path, file_name).replace("/", "//")
                    bounding_boxes = get_json(file_path)

                    # Write bounding box instances to the text file
                    with open(txt_file_path, 'w') as txt_file:
                        for box in bounding_boxes:
                            class_id = 0
                            line = f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}\n"
                            txt_file.write(line)

                counter += 1

##################################################################
# (2) For moving files, and renaming
import os
import shutil

def move_image():
    # Specify the main folder path
    main_folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 8"
    # Define the target folder name
    target_folder_name = "images"
    target_folder_path = os.path.join(main_folder_path, target_folder_name)

    # Loop through subfolders in the main folder
    for root, dirs, files in os.walk(main_folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)

            # Create the target folder if it doesn't exist
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)

            # Loop through files in the subfolder
            for file_name in os.listdir(subfolder_path):
                if file_name == "step0.camera.png":
                    try:
                        source_file_path = os.path.join(subfolder_path, file_name)
                        target_file_name = "im" + dir_name.split(".")[1] + ".png"
                        target_file_path = os.path.join(target_folder_path, target_file_name)

                        # Move the file to the target folder
                        shutil.copy(source_file_path, target_file_path)
                    except:
                        break
def move_label():
    # Specify the main folder path
    main_folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 8"
    # Define the target folder name
    target_folder_name = "labels"
    target_folder_path = os.path.join(main_folder_path, target_folder_name)

    # Loop through subfolders in the main folder
    for root, dirs, files in os.walk(main_folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)

            # Create the target folder if it doesn't exist
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)

            # Loop through files in the subfolder
            for file_name in os.listdir(subfolder_path):
                if file_name == "output.txt":
                    try:
                        source_file_path = os.path.join(subfolder_path, file_name)
                        target_file_name = "im" + dir_name.split(".")[1] + ".txt"
                        target_file_path = os.path.join(target_folder_path, target_file_name)

                        # Move the file to the target folder
                        shutil.copy(source_file_path, target_file_path)
                    except:
                        break

##################################################################
# (3) For plotting labels on image, for testing

import os
import cv2
import matplotlib.pyplot as plt

def plot_label():
    # Specify the root folder path
    root_folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 8/"

    # Define the subfolder names
    image_subfolder = "images"
    label_subfolder = "labels"

    # Get the list of image files
    image_folder_path = os.path.join(root_folder_path, image_subfolder)
    image_files = sorted(os.listdir(image_folder_path))

    # Get the list of label files
    label_folder_path = os.path.join(root_folder_path, label_subfolder)
    label_files = sorted(os.listdir(label_folder_path))

    # Iterate through the image and label files
    counter = 0
    # Iterate through the image and label files
    for image_file, label_file in zip(image_files, label_files):
        # Read the image
        image_path = os.path.join(image_folder_path, image_file)
        image = cv2.imread(image_path)

        # Read the label file
        label_path = os.path.join(label_folder_path, label_file)
        with open(label_path, 'r') as file:
            lines = file.readlines()
            boxes = [line.strip().split() for line in lines]

        # Draw bounding boxes on the image
        for box in boxes:
            class_label, center_x, center_y, width_x, height_y = map(float, box)
            img_height, img_width, _ = image.shape

            x = int((center_x - width_x / 2) * img_width)
            y = int((center_y - height_y / 2) * img_height)
            w = int(width_x * img_width)
            h = int(height_y * img_height)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original image and the image with bounding boxes side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Image with Bounding Boxes")
        ax[1].axis('off')

        plt.show()
        counter += 1
        if counter == 10:
            break

##################################################################
# (4) For training custom dataset
from ultralytics import YOLO

def train_model():
    # pretrained
    model = YOLO("yolov8n.pt")
    model.train(data = "my_dataset.yaml", epochs = 10)

##################################################################
# MAIN
#create_label()
#move_image()
#move_label()
#plot_label()
train_model()







