import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Constants
FOLDER_PATH = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/"
MODEL_NAME = "UNet1.model"
TRAIN_PATH = FOLDER_PATH + "lift test 5"
result_color = (0, 255, 0) #green
estimate_count = []
result_count = []
passenger_count = []

def estimate_lift_area(image):
    # Convert the image to the HSV color space
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper threshold for sharp green color
	lower_green = np.array([ 50, 160,   0]) 
	upper_green = np.array([200, 255, 100]) 
	
    # Create a mask using the green color threshold
	mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply a series of morphological operations to remove noise and fill in holes
	#kernel = np.ones((5, 5), np.uint8)
	#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
    # Bitwise AND operation to extract the green area
	estimate_area = cv2.bitwise_and(image, image, mask=mask)
	
    # Count the number of pixels with the specified color
	pixel_count = np.sum(mask / 255)
	
	return int(pixel_count), mask#estimate_area

def count_color_area(image, color):
    # Convert the image to the BGR color space
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Define the lower and upper threshold for the specified color
    lower_color = np.array(color)
    upper_color = np.array(color)

    # Create a mask using the color threshold
    mask = cv2.inRange(bgr_image, lower_color, upper_color)

    # Count the number of pixels with the specified color
    pixel_count = np.sum(mask / 255)

    return int(pixel_count)

def count_passenger(data):
	# Find the "instances" in "step0.camera.passenger.png"
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
	return instance_count

def print_results(estimate, result, passenger):
	# Display results
	counter2 = 0
	for counter2 in range(len(estimate)):
		print("Picture", counter2, ": Estimtate:", estimate[counter2], " Actual:", result[counter2], " Passenger:", passenger[counter2])
		counter2 += 1

def plot_result(estimate, result, passenger):
	# Plot estimation against result with linear regression
	m1, b1 = np.polyfit(passenger, estimate, 1)
	m2, b2 = np.polyfit(passenger, result, 1)
	regression_line1 = np.add(np.multiply(m1, passenger), b1) # y = m1 * passenger_count + b1
	regression_line2 = np.add(np.multiply(m2, passenger), b2) # y = m2 * passenger_count + b2
	plt.scatter(passenger, estimate, color = 'red')
	plt.scatter(passenger, result, color = 'blue')
	plt.plot(passenger, regression_line1, color='red', label='estimate')
	plt.plot(passenger, regression_line2, color='blue', label='result')
	plt.show()

# Iterate over each file in the folder
file_list = os.listdir(FOLDER_PATH)
counter1 = 0
for counter1 in range(10):
	# set file path
	image_path = "{}/sequence.{}/step0.camera.png".format(TRAIN_PATH, counter1)
	result_path = "{}/sequence.{}/step0.camera.floor.png".format(TRAIN_PATH, counter1)
	data_path = "{}/sequence.{}/step0.frame_data.json".format(TRAIN_PATH, counter1)

    # load file data 
	image = cv2.imread(image_path)
	result = cv2.imread(result_path)
	f = open(data_path)
	data = json.load(f)
	
	# Estimate the floor area
	area, mask = estimate_lift_area(image)
	estimate_count.append(area)

	# Count the actual floor area
	result_count.append(count_color_area(result, result_color))

	# Count the actual passengers
	passenger_count.append(count_passenger(data))

	# compare area mask and actual area
	plt.subplot(1,2,1)
	plt.imshow(result)
	plt.subplot(1,2,2)
	plt.imshow(mask)
	plt.show()

	f.close()
	counter1 += 1

#plot_result(estimate_count, result_count, passenger_count)





