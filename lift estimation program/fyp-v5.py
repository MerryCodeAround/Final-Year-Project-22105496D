import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import random
import os
from ultralytics import YOLO
import json
import numpy as np

folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 8/"
image_file = ""
passenger_file = ""
floor_file = ""
data_file = ""

#################################################################################
# GUI components

# Create the main window
window = tk.Tk()
window.title("Estimation of Available Space in Elevators using Image Processing and Machine Learning")

# Create the Image frame
image_frame = tk.Frame(window)
image_frame.pack(pady=10)

# Create the panel for estimation and actual values
panel_frame = tk.Frame(window)
panel_frame.pack(pady=10)

# Create the labels and entries for estimation and actual values
estimation_label = tk.Label(panel_frame, text="Estimated number of passengers:")
estimation_number = tk.Entry(panel_frame)

actual_label = tk.Label(panel_frame, text="Actual number of passengers:")
actual_number = tk.Entry(panel_frame)

#################################################################################
# Change the image on Tkinter GUI

def change_image(image_path, passenger_path, floor_path, boxes):
    # List of 3 images, and their size
    images = []
    img_width = 300
    img_height = 300

    # Append image into list
    # top left = original
    image = Image.open(image_path)
    image.thumbnail((img_width, img_height))
    tk_image = ImageTk.PhotoImage(image)
    images.append(tk_image)

    # top right = original + bounding box
    image = Image.open(image_path)
    img_box = Image.new('RGBA', image.size, (255,255,255,0)) # blank image with transparent colour
    img_draw  = ImageDraw.Draw(img_box)
    for box in boxes: #draw bounding boxes on image
        for rect in box.xyxy:
            img_draw.rectangle([rect[0], rect[1], rect[2], rect[3]], fill = None, outline = "red", width = 2)
            # x = (rect[0] - rect[2] / 2) / 960 * 300
            # y = (rect[1] - rect[3] / 2)/ 540* 300
            # w = (rect[2])/ 960 * 300
            # h = (rect[3])/ 540* 300
            # img_draw.rectangle([x,y,w,h], fill = None, outline = "red", width = 5)
    out = Image.alpha_composite(image, img_box)
    #image = image.resize((300,300), Image.LANCZO)
    out.thumbnail((img_width, img_height))
    tk_image = ImageTk.PhotoImage(out)
    images.append(tk_image)

    # bottom left = passenger mask
    passenger = Image.open(passenger_path)
    passenger.thumbnail((img_width, img_height))
    tk_image = ImageTk.PhotoImage(passenger)
    images.append(tk_image)

    # buttom right = floor mask
    floor = Image.open(floor_path)
    floor.thumbnail((img_width, img_height))
    tk_image = ImageTk.PhotoImage(floor)
    images.append(tk_image)

    # Update the image labels
    labels = ["Original Image", "Bounding Box", "Passenger Mask", "Floor Mask"]
    for i, image_label in enumerate(image_labels):
        image_label.config(image=images[i])
        image_label.image = images[i]
        image_label.config(text=labels[i])

#################################################################################
# Randomize the file being opened

def randomize_image():
    # Get a list of all subfolders in the root folder
    subfolders = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

    # Select a random subfolder
    random_subfolder = random.choice(subfolders)
    subfolder_path = folder_path + random_subfolder + "/"

    # Get a list of image and json files in the selected subfolder
    global image_file, passenger_file, floor_file, data_file
    image_file = subfolder_path + "step0.camera.png"
    passenger_file = subfolder_path + "step0.camera.passenger.png"
    floor_file = subfolder_path + "step0.camera.floor.png"
    data_file = subfolder_path + "step0.frame_data.json"

#################################################################################
# Retrieving estimation and actual passenger data from file and neural network
# See "LiftEstimation5.py" for more

# pretrained model
model = YOLO("best.pt")

# estimate the image and return number of passenger and bounding box in xyxy
def estimate_passenger(path):
    # predict on the image
    results = model.predict(source = path, classes = [0], conf = 0.2)

    # Process results generator
    boxes = []
    for result in results:
        boxes.append(result.boxes) # Boxes object for bounding box outputs
        #print(boxes.xyxy)
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #result.show()  # display to screen

    return len(result.boxes.xyxy), boxes

# return number of passengers from json file
def count_passenger(path):
    file = open(path)
    data = json.load(file)
    instance_count = 0
    for capture in data['captures']:
        if 'annotations' in capture:
            for annotation in capture['annotations']:
                if 'filename' in annotation and annotation['filename'] == 'step0.camera.passenger.png':
                    try:
                        instances = annotation['instances']
                        instance_count = len(instances)
                        break
                    except:
                        break
    file.close()
    return instance_count

#################################################################################
# Randomize and then update image on GUI

def update_image():
    # randomizing image paths
    randomize_image()

    # estimate passenger number and their bounding box
    estimation, bounding_boxes = estimate_passenger(image_file)
    print(estimation)

    # get the actual passenger number from json file
    actual = count_passenger(data_file)

    # update the text entry fields
    estimation_number.delete(0, "end")
    estimation_number.insert(tk.END, str(estimation))
    actual_number.delete(0, "end")
    actual_number.insert(tk.END, str(actual))

    # changing the image on GUI accordingly
    change_image(image_file, passenger_file, floor_file, bounding_boxes)

#################################################################################
# Creating the GUI

# Create the label for the image
image_labels = []
for i in range(4):
    image_label = tk.Label(image_frame, text="Image", compound=tk.TOP, wraplength = 300)
    image_label.grid(row = i // 2, column = i % 2, padx = 10, pady = 10)
    image_labels.append(image_label)

# Create the button to change the image
change_button = tk.Button(image_frame, text="Change Image", command=update_image)
change_button.grid(row = 2, column = 0, columnspan = 2, pady=10)

# To randomize the initial image
update_image()

# Create the labels and entries for estimation and actual values
estimation_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
estimation_number.grid(row=0, column=1, padx=5, pady=5)

actual_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
actual_number.grid(row=1, column=1, padx=5, pady=5)

window.mainloop()

#################################################################################

