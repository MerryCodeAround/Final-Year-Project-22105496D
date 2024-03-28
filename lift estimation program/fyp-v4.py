'''
LiftEstimation4d.py

Concluding all previous image processing methods

'''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import time
import math
import skfuzzy as fuzz
import tensorflow as tf
from skfuzzy import control as ctrl
import tensorflow as tf

# Global Variables
estimate_list = []  # 
result_list = []    # arrays for ploting result
passenger_list = [] #
model_name = "handwritten.model"

##################################################################
# File Handling methods

main_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/" #where all training data is located
train_data_path = main_path + "training data/" + "lift test 7/" #select which dataset to use
estimate_data_path = main_path + "training data/" + "lift test 7/" #select which dataset to use

# Reading JSON files
def get_json(data):
	'''
	return the number of passenger from json file
	'''
	passenger_count = 0 # passenger number
	passenger_colour = [] # list of passenger colour
	bounding_boxes = [] # list of bounding boxes
	for capture in data['captures']:
		if 'annotations' in capture:
			for annotation in capture['annotations']:
				if 'filename' in annotation and annotation['filename'] == 'step0.camera.passenger.png':
					#get number of passenger and segmentation colour	
					instances = annotation['instances']
					print(len(instances))
					for instance in instances:
						pixel_value = instance['pixelValue']
						passenger_colour.append(pixel_value)
					passenger_count = len(instances)
					print(passenger_count)
				if 'id' in annotation and annotation['id'] == 'bounding box':
					#get bounding box in 'xywh' format
					boxes = annotation['values']
					for box in boxes:
						bounding_box = []
						bounding_box.append(box['origin'][0])
						bounding_box.append(box['origin'][1])
						bounding_box.append(box['dimension'][0])
						bounding_box.append(box['dimension'][1])
						#print(bounding_box)
						bounding_boxes.append(bounding_box)
					
	#print(bounding_boxes)
	return passenger_count, passenger_colour, bounding_boxes


##################################################################
# Image Processing methods
# Ref: mainly myself

# return pixel count and image mask of specific colour
def get_colour_mask(path, colour = (0,0,0)):
	image = cv2.imread(path)
	npcolour = np.array(colour)
	mask = cv2.inRange(image, npcolour, npcolour)
	pixel_count = np.sum(mask / 255)
	return int(pixel_count), mask
	
# return pixel count and image mask of inverse of a specific colour
def get_inverse_colour_mask(path, colour = (0,0,0)):
	image = cv2.imread(path)
	npcolour = np.array(colour)
	h, w, _ = image.shape
	floor_area, mask = get_colour_mask(path, npcolour)
	return h * w - floor_area, cv2.bitwise_not(mask)

# plot a linear regression graph
def plot_result(estimate, result, passenger):
	'''
	Plot estimation against result with linear regression
	- red = estimation
	- blue = actual
	'''
	
	# # Display results by printing data
	# counter2 = 0
	# for counter2 in range(len(estimate)):
		# print("Picture", counter2, ": Estimtate:", estimate[counter2], " Actual:", result[counter2], " Passenger:", passenger[counter2])
		# counter2 += 1
	
	# Display result by plotting data
	m1, b1 = np.polyfit(passenger, estimate, 1)
	m2, b2 = np.polyfit(passenger, result, 1)
	regression_line1 = np.add(np.multiply(m1, passenger), b1) # y = m1 * passenger_count + b1
	regression_line2 = np.add(np.multiply(m2, passenger), b2) # y = m2 * passenger_count + b2
	plt.scatter(passenger, estimate, color = 'red')
	plt.scatter(passenger, result, color = 'blue')
	plt.plot(passenger, regression_line1, color='red', label='Linear Regression Line 1')
	plt.plot(passenger, regression_line2, color='blue', label='Linear Regression Line 2')
	plt.show()


##################################################################
# Edge Detection methods from scatch
# Ref:
# 	- https://github.com/akash18tripathi/Gaussian-Mixture-Models-for-Background-Extraction/blob/main/Gaussian%20Mixture%20Model.ipynb
#	- https://medium.com/@rohit-krishna/coding-canny-edge-detection-algorithm-from-scratch-in-python-232e1fdceac7
# 	- https://www.youtube.com/watch?v=wQg7BNmi8zs&ab_channel=BleedAIAcademy
# 	- https://www.youtube.com/watch?v=Ccqa9KBO9_U&ab_channel=AndrewsCordolinoSobral

# grey scale with RGB weight 
def grey_scale(img: np.ndarray):
	'''
	Using weighted sum of RGB since human eye are more sensitive to green light
	Algorithm:
		output = 0.2989 * R + 0.5870 * G + 0.1140 * B 
	'''
	r_coef = 0.2989
	g_coef = 0.5870
	b_coef = 0.1140
	r, g, b = img[..., 0], img[..., 1], img[..., 2]
	return r_coef * r + g_coef * g + b_coef * b

# there are different types of filter: gaussian, median, mean, mode...
# used to blur the image 
def gaussian_filter(sigma: int | float, filter_shape: list | tuple | None):
	'''
	Generate a gaussian filter kernel
	'sigma' is the standard deviation of the gaussian distribution
	'filter_shape' is a array showing 2/3D array of [row,column]
	'''
	m, n = filter_shape
	m_half = m // 2
	n_half = n // 2

	# initializing the filter
	gaussian_filter = np.zeros((m, n), np.float32)

	# generating the filter
	for y in range(-m_half, m_half):
		for x in range(-n_half, n_half):
			normal = 1 / (2.0 * np.pi * sigma**2.0)
			exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
			gaussian_filter[y+m_half, x+n_half] = normal * exp_term

	return gaussian_filter

# a filter to detect edges in image using convolution kernels
def sobel_edge(image):
	'''
	Apply sobel edge detecting algorithm on image
	return the grey scale image of edges
	'''
	image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
	image_arr = np.asarray(image_grey)
	Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
	Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
	[rows, columes] = np.shape(image_arr)
	image_arr_sobel = np.zeros(shape = (rows, columes))
	for r in range (rows - 2):
		for c in range (columes - 2):
			gx = np.sum(np.multiply(Gx, image_arr[r:r + 3, c:c + 3]))
			gy = np.sum(np.multiply(Gy, image_arr[r:r + 3, c:c + 3]))
			pixel_value = np.sqrt(gx ** 2 + gy ** 2)
			image_arr_sobel[r + 1, c + 1] = pixel_value
	return image_arr_sobel

def convolution(image: np.ndarray, kernel: list | tuple) -> np.ndarray:
	'''
	It is a "valid" Convolution algorithm implementaion.
	### Example
	>>> import numpy as np
	>>> from PIL import Image
	>>>
	>>> kernel = np.array(
	>>>	 [[-1, 0, 1],
	>>>	 [-2, 0, 2],
	>>>	 [-1, 0, 1]], np.float32
	>>> )
	>>> img = np.array(Image.open('./lenna.png'))
	>>> res = convolution(img, Kx)
	'''

	if len(image.shape) == 3:
		m_i, n_i, c_i = image.shape

	# if the image is gray then we won't be having an extra channel so handling it
	elif len(image.shape) == 2:
		image = image[..., np.newaxis]
		m_i, n_i, c_i = image.shape
	else:
		raise Exception('Shape of image not supported')

	m_k, n_k = kernel.shape

	y_strides = m_i - m_k + 1  # possible number of strides in y direction
	x_strides = n_i - n_k + 1  # possible number of strides in x direction

	img = image.copy()
	output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
	output = np.zeros(output_shape, dtype=np.float32)

	count = 0  # taking count of the convolution operation being happening

	output_tmp = output.reshape(
		(output_shape[0]*output_shape[1], output_shape[2])
	)

	for i in range(y_strides):
		for j in range(x_strides):
			for c in range(c_i): # looping over the all channels
				sub_matrix = img[i:i+m_k, j:j+n_k, c]

				output_tmp[count, c] = np.sum(sub_matrix * kernel)

			count += 1

	output = output_tmp.reshape(output_shape)

	return output


##################################################################
# Edge Detection methods using OpenCV

# plotting result of different threshold allgorithm from OpenCV
# testing only
def threshold_edge_test(image, threshold_ratio = 1):
	'''
	adaptive thresholding that take in neighbouring pixel values into algorithm
	'''
	# Writing from scratch
	# # Convert the image to HSV color space
	# image_hsv = matplotlib.colors.rgb_to_hsv(image)

	# # Compute the local threshold for each pixel in each HSV component
	# image_threshold = threshold_ratio * np.mean(image_hsv, axis=(0, 1))

	# # Perform adaptive edge thresholding in each HSV component
	# image_edge = np.zeros_like(image_hsv)
	# for i in range(3):
		# image_edge[..., i] = np.where(image_hsv[..., i] >= image_threshold[i], 1, 0)

	# # Convert the image back to RGB color space
	# image_edge = matplotlib.colors.hsv_to_rgb(image_edge)

	# # Convert the image to grayscale
	# image_edge = np.mean(image_edge, axis=2) * 255

	# return image_edge.astype(np.uint8)
	
	
	# Using opencv functions
	image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #greyscale
	_, image_edge1 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV) #binary
	_, image_edge2 = cv2.threshold(image_grey,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #otsu
	image_edge3 = cv2.adaptiveThreshold(image_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10) #mean
	image_edge4 = cv2.adaptiveThreshold(image_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #gaussian
	
	# display mask against actual
	plt.figure(figsize=(14,8))
	plt.subplot(2,3,1)
	plt.imshow(image)
	plt.title("original")
	plt.subplot(2,3,2)
	plt.imshow(image_edge1, cmap = 'gray')
	plt.title("binary")
	plt.subplot(2,3,3)
	plt.imshow(image_edge2, cmap = 'gray')
	plt.title("otsu")
	plt.subplot(2,3,4)
	plt.imshow(image_edge3, cmap = 'gray')
	plt.title("mean")
	plt.subplot(2,3,5)
	plt.imshow(image_edge4, cmap = 'gray')
	plt.title("gaussian")
	plt.show()

# return image after gaussian filter
def threshold_edge(image):
	image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #greyscale
	image_edge = cv2.adaptiveThreshold(image_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #gaussian
	return image_edge
	
# return an image after contouring
def contour_edge(image):
	# # display original image
	# cv2.imshow('edge', image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows() 
	
	image_contour = np.zeros_like(image)
	contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		epsilon = 0.001 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)
		if cv2.contourArea(approx) > 1000: #apply size filter
			if cv2.arcLength(approx, True): #whether closed shape or not
				#image_contour = np.zeros_like(image)
				cv2.drawContours(image_contour, [approx], 0, 255, 1) 
				# cv2.imshow('contour', image_contour)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows() 
	
	# contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# for contour in contours:
		# epsilon = 0.01 * cv2.arcLength(count, True)
		# approximations = cv2.approxPolyDP(contour, epsilon, True)
		# cv2.drawContours(imageread, [approximations], 0, (0), 3)
	# print("number of contours = " + str(len(contours)))
	# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
	cv2.imshow('contours', image_contour)
	cv2.waitKey(0) 
	cv2.destroyAllWindows() 
		
def canny_edge(image, sigma=1, low_threshold=30, high_threshold=100):
	
	# Step 1: Convert the image to grayscale
	#image_grey = grey_scale(image)
	image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Step 2: Apply Gaussian smoothing to reduce noise
	#Gaussian_kernel = gaussian_filter(sigma, [10,10])
	#image_blur = convolution(image_grey, Gaussian_kernel)
	image_blur = cv2.GaussianBlur(image_grey, (9,9), 3) #img, kernel_size, sigma

	# Step 3: Calculate gradients using Sobel operator
	grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)

	# Step 4: Calculate gradient magnitude and direction
	gradient_magnitude = np.hypot(grad_x, grad_y)
	gradient_direction = np.arctan2(grad_y, grad_x)

	# Step 5: Non-maximum suppression
	image_suppressed = np.zeros_like(gradient_magnitude)
	for i in range(1, gradient_magnitude.shape[0] - 1):
		for j in range(1, gradient_magnitude.shape[1] - 1):
			q = 255
			r = 255
			angle = gradient_direction[i, j] * 180.0 / np.pi
			if (0 <= angle < 22.5) or (157.5 <= angle < 180):
				q = gradient_magnitude[i, j+1]
				r = gradient_magnitude[i, j-1]
			elif (22.5 <= angle < 67.5):
				q = gradient_magnitude[i+1, j-1]
				r = gradient_magnitude[i-1, j+1]
			elif (67.5 <= angle < 112.5):
				q = gradient_magnitude[i+1, j]
				r = gradient_magnitude[i-1, j]
			elif (112.5 <= angle < 157.5):
				q = gradient_magnitude[i-1, j-1]
				r = gradient_magnitude[i+1, j+1]

			if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
				image_suppressed[i, j] = gradient_magnitude[i, j]

	# Step 6: Double thresholding
	image_threshold = np.zeros_like(image_suppressed)
	image_threshold[(image_suppressed >= high_threshold)] = 255
	image_threshold[(image_suppressed <= low_threshold)] = 0

	# Step 7: Edge tracking by hysteresis
	image_edge = np.zeros_like(image_threshold)
	for i in range(1, image_threshold.shape[0] - 1):
		for j in range(1, image_threshold.shape[1] - 1):
			if image_threshold[i, j] == high_threshold:
				image_edge[i, j] = 255
				image_edge[(i-1):(i+2), (j-1):(j+2)] = 255

	return image_edge

##################################################################
# Main Function Loops methods
# provide several ways to estimate the lift area with different image processing algorithms, which are 
#	- colour_thresholding
#	- background_subtraction
#	- edge_detection

def color_threshold():
	'''
	Main colour thresholding algorithm for estimating lift area
	Will plot a graph of linear regression line
	'''

	# Iterate over each file in the folder and plot actual and estimated results
	file_list = os.listdir(train_data_path)
	counter1 = 0
	for counter1 in range(100):
		# get files
		image_path = train_data_path  + "sequence." + str(counter1) + "/step0.camera.png"
		result_path = train_data_path + "sequence." + str(counter1) + "/step0.camera.floor.png"
		data_path = train_data_path   + "sequence." + str(counter1) + "/step0.frame_data.json"
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		data_file = open(data_path)
		data = json.load(data_file)
			
		#########################################################################
		# Estimate the floor area with colour thresholding
		# Convert the image to the HSV color space
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Define the lower and upper threshold for sharp green color
		lower_green = np.array([ 50, 160,   0])  # Adjust these values based on your specific case
		upper_green = np.array([200, 255, 100])  # Adjust these values based on your specific case
		
		# Create a mask using the green color threshold
		mask = cv2.inRange(hsv_image, lower_green, upper_green)
		
		# Bitwise AND operation to extract the green area
		estimate_area = cv2.bitwise_and(image, image, mask=mask)
		
		# Count the number of pixels with the specified color
		floor_area = np.sum(mask / 255)
		estimate_list.append(floor_area)

		#########################################################################
		# Count the actual floor area
		result_count, _ = get_colour_mask(result_path, (0,255,0))
		result_list.append(result_count)

		#########################################################################
		# Count the actual passengers
		passenger_number, _, _ = get_json(data)
		#print(passenger_number)
		passenger_list.append(passenger_number)

		data_file.close()
		counter1 += 1

		# plt.tight_layout()
		# plt.subplot(1, 2, 1)
		# plt.imshow(result)
		# plt.subplot(1, 2, 2)
		# plt.imshow(mask, cmap = "grey")
		# plt.show()
		
	plot_result(estimate_list, result_list, passenger_list)

def adaptive_colour_threshold():

	# Convert the image to the HSV color space
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# Define the central pixel position
	height, width, _ = image.shape
	print(image.shape)

	# Get the central pixel color values
	center_hsv = hsv_image(height / 2, width / 2)

	# Calculate the lower and upper threshold based on the central pixel values
	lower_threshold = np.maximum(center_hsv - radius, 0)
	upper_threshold = np.minimum(center_hsv + radius, 255)

	# Create a mask using the adaptive color threshold
	mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

	# Count the number of pixels with the specified color
	pixel_count = np.sum(mask / 255)

	return int(pixel_count), mask

def background_subtraction():
	'''
	Main background subtraction algorithm for estimating lift area
	Will plot a graph of linear regression line
	'''

	# Get Background image
	background_path = train_data_path + "background/step0.camera.png" #background image for background subtraction
	background = cv2.imread(background_path)

	height, width, _ = background.shape
	total_pixel = height * width
	
	# Iterate over each file in the folder
	file_list = os.listdir(train_data_path)
	counter1 = 0
	for counter1 in range(300):
		# get files
		image_path = train_data_path  + "sequence." + str(counter1) + "/step0.camera.png"
		result_path = train_data_path + "sequence." + str(counter1) + "/step0.camera.passenger.png"
		data_path = train_data_path   + "sequence." + str(counter1) + "/step0.frame_data.json"
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		data_file = open(data_path)
		data = json.load(data_file)
		
		# load file data 
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		f = open(data_path)
		data = json.load(f)

		#########################################################################
		# Estimate the passenger area with background subtraction

		# Convert to HSV colour space
		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
		background_grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) # convert to greyscale
		
		# Set the threshold values for each HSV channel
		h_high = 20	 # Hue threshold
		h_low  = 0	 # Hue threshold
		s_high = 50	 # Saturation threshold
		s_low  = 0	 # Saturation threshold
		v_high = 50	 # Value threshold
		v_low  = 0	 # Value threshold
	
		# Perform background subtraction using the absolute difference
		diff = cv2.absdiff(image_hsv, background_hsv)

		# Apply thresholding to obtain the foreground mask
		mask = cv2.inRange(diff, (h_low, s_low, v_low), (h_high, s_high, v_high), 255)
		mask = cv2.bitwise_not(mask)

		# Perform morphological operations to remove noise
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

		# Apply the mask to the original image to extract the foreground
		foreground = cv2.bitwise_and(image, image, mask=mask)
		floor_area = np.sum(mask / 255)
		
		estimate_list.append(floor_area)
			
		#########################################################################
		# Count the actual floor area
		result_count, _ = get_inverse_colour_mask(result_path)
		result_list.append(result_count)

		#########################################################################
		# Count the actual passengers
		passenger_number, _, _ = get_json(data)
		passenger_list.append(passenger_number)

		# plt.tight_layout()
		# plt.subplot(1, 2, 1)
		# plt.imshow(result)
		# plt.subplot(1, 2, 2)
		# plt.imshow(mask, cmap = "grey")
		# plt.show()

		data_file.close()
		counter1 += 1
	
	# plot the estimation against result
	plot_result(estimate_list, result_list, passenger_list)
	
def edge_detection():
	'''
	Main edge detection algorithm for estimation lift area
	only output the contoured image, but don't know how to progress from there... TwT
	'''
	
	# Get Background image
	background_path = train_data_path + "background/step0.camera.png" #background image for background subtraction
	background = cv2.imread(background_path)
	height, width, _ = background.shape
	total_pixel = height * width
	
	#############################################################################
	# testing edge detection on background
	threshold_edge_test(background) #testing different filtering methods
	
	#############################################################################
	# Iterate over each file in the folder
	file_list = os.listdir(train_data_path)
	counter1 = 0
	for counter1 in range(10):
		# get files
		image_path = train_data_path  + "sequence." + str(counter1) + "/step0.camera.png"
		result_path = train_data_path + "sequence." + str(counter1) + "/step0.camera.passenger.png"
		data_path = train_data_path   + "sequence." + str(counter1) + "/step0.frame_data.json"
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		data_file = open(data_path)
		data = json.load(data_file)
		
		# load file data 
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		f = open(data_path)
		data = json.load(f)

		#########################################################################
		# Apply edge detect on background image
		#background_edge = sobel_edge(background)
		#background_edge = canny_edge(background, sigma = 1, low_threshold = 0, high_threshold = 15)
		#threshold_edge(background, threshold_ratio = 0.7)
		image_edge = threshold_edge(image)
		image_contour = contour_edge(image_edge)

class fuzzy_logic_controller:
	
	def __init__(self):
		# Getting background image
		background = cv2.imread(BACKGROUND_PATH)
		background_grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) # convert to greyscale	
		height, width, _ = background.shape
		total_pixel = height * width
		
		# Maximum values
		self.max_floor_area, self._= estimate_lift_area('colour_threshold', background, background_grey)
		self.max_passenger_area = total_pixel

		# Define the input and output variables
		self.floor_area = ctrl.Antecedent(np.arange(0, self.max_floor_area + 1, 1), 'floor_area')
		self.passenger_area = ctrl.Antecedent(np.arange(0, self.max_passenger_area + 1, 1), 'passenger_area')
		self.num_passengers = ctrl.Consequent(np.arange(0, 13, 1), 'num_passengers')

		# Define the membership functions for each variable
		self.floor_area['small'] = fuzz.trimf(self.floor_area.universe, [0, 0, self.max_floor_area * 0.5])
		self.floor_area['medium'] = fuzz.trimf(self.floor_area.universe, [self.max_floor_area * 0.2, self.max_floor_area * 0.5, self.max_floor_area * 0.8])
		self.floor_area['large'] = fuzz.trimf(self.floor_area.universe, [self.max_floor_area * 0.5, self.max_floor_area, self.max_floor_area])

		self.passenger_area['small'] = fuzz.trimf(self.passenger_area.universe, [0, 0, self.max_passenger_area * 0.5])
		self.passenger_area['medium'] = fuzz.trimf(self.passenger_area.universe, [self.max_passenger_area * 0.2, self.max_passenger_area * 0.5, self.max_passenger_area * 0.8])
		self.passenger_area['large'] = fuzz.trimf(self.passenger_area.universe, [self.max_passenger_area * 0.5, self.max_passenger_area, self.max_passenger_area])

		self.num_passengers['low'] = fuzz.trimf(self.num_passengers.universe, [0, 0, 4])
		self.num_passengers['medium'] = fuzz.trimf(self.num_passengers.universe, [2, 6, 10])
		self.num_passengers['high'] = fuzz.trimf(self.num_passengers.universe, [8, 12, 12])

		# Define the fuzzy rules
		self.rule1 = ctrl.Rule(self.floor_area['small'] | self.passenger_area['small'], self.num_passengers['low'])
		self.rule2 = ctrl.Rule(self.floor_area['medium'] & self.passenger_area['medium'], self.num_passengers['medium'])
		self.rule3 = ctrl.Rule(self.floor_area['large'] & self.passenger_area['large'], self.num_passengers['high'])

		# Create the fuzzy control system
		self.passenger_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])
		self.passenger_estimation = ctrl.ControlSystemSimulation(self.passenger_ctrl)

	def run(self, floor_area_input, passenger_area_input):
		# Set the inputs
		self.passenger_estimation.input['floor_area'] = floor_area_input
		self.passenger_estimation.input['passenger_area'] = passenger_area_input

		# Compute the estimated number of passengers
		self.passenger_estimation.compute()

		# Get the output
		num_passengers_output = self.passenger_estimation.output['num_passengers']

		return num_passengers_output

	def estimate_with_passenger_floor_ratio(self):
		# get buckground
		background = cv2.imread(BACKGROUND_PATH)
		background_grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) # convert to greyscale
		height, width, _ = background.shape
		total_pixel = height * width
		
		# Iterate over each file in the folder and plot actual and estimated results
		file_list = os.listdir(FOLDER_PATH)
		counter1 = 0
		for counter1 in range(10):
			# set file path
			image_path = "{}sequence.{}/step0.camera.png".format(TRAIN_PATH, counter1)
			result_path = "{}sequence.{}/step0.camera.floor.png".format(TRAIN_PATH, counter1)
			data_path = "{}sequence.{}/step0.frame_data.json".format(TRAIN_PATH, counter1)

			# load file data 
			image = cv2.imread(image_path)
			result = cv2.imread(result_path)
			f = open(data_path)
			data = json.load(f)
			
			# Estimate the passenger area with background subtraction
			floor_area, _ = estimate_lift_area('background_subtraction', image, background)

			# Count the actual passenger area
			passenger_area = total_pixel - count_color_area(result, (0,0,0))
			
			# Count the actual passengers
			passenger_count = count_passenger(data)

			esimate_passenger_count = self.run(floor_area, passenger_area)
			#print("iteration: " + str(counter1) + " | actual: " + str(passenger_count) + " | estimate: " + str(esimate_passenger_count))

			f.close()
			counter1 += 1

class neural_network_handler:
	'''
	class variables:
	- model (trained model)
	- folder_path (contain all program and data)
	- train_data_path 
	- estimate_data_path
	- 
	
	class functions:
		build
	'''
	
	def __init__(self):
		if os.path.isdir(model_name): #there is existing model
			self.model = tf.keras.models.load_model(model_name) #handwritten.model
		else: #build and train model
			self.build_written()
			self.summary()
			self.train_mnist(1)
			
	def build_written(self):
		'''
		define the model structure for detecting hand written model
		'''
		# create model
		self.model = tf.keras.models.Sequential()

		#convolution layers
		self.model.add(tf.keras.layers.Conv2D(28, (3,3), activation = 'relu', input_shape = (28, 28, 1)))
		self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		self.model.add(tf.keras.layers.Conv2D(56, (3,3), activation = 'relu'))
		self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		self.model.add(tf.keras.layers.Conv2D(56, (3,3), activation = 'relu'))

		#convert 3D matrix into 2D array
	#	model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
		self.model.add(tf.keras.layers.Flatten())

		#dense layer for classification
		self.model.add(tf.keras.layers.Dense(64, activation='relu'))
		self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
		
		self.model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

	def build_UNet(self, class_number = 4, image_height = 256, image_width = 256, image_channels = 1):
		'''
		define the UNet model for image segmentation
		'''
		
		inputs = Input((image_height, image_width, image_channels))
		#s = lambda(lambda x: x / 255)(inputs) #for normalizing input
		s = inputs
		
		#Contraction path
		c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
		c1 = Dropout(0.1)(c1)
		c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
		p1 = MaxPooling2D((2, 2))(c1)
		
		c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
		c2 = Dropout(0.1)(c2)
		c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
		p2 = MaxPooling2D((2, 2))(c2)
		 
		c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
		c3 = Dropout(0.2)(c3)
		c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
		p3 = MaxPooling2D((2, 2))(c3)
		 
		c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
		c4 = Dropout(0.2)(c4)
		c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
		p4 = MaxPooling2D(pool_size=(2, 2))(c4)
		 
		c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
		c5 = Dropout(0.3)(c5)
		c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
		
		#Expansive path 
		u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
		u6 = concatenate([u6, c4])
		c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
		c6 = Dropout(0.2)(c6)
		c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
		 
		u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
		u7 = concatenate([u7, c3])
		c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
		c7 = Dropout(0.2)(c7)
		c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
		 
		u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
		u8 = concatenate([u8, c2])
		c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
		c8 = Dropout(0.1)(c8)
		c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
		 
		u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
		u9 = concatenate([u9, c1], axis=3)
		c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
		c9 = Dropout(0.1)(c9)
		c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
		 
		outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
		 
		model = Model(inputs=[inputs], outputs=[outputs])
		
		#NOTE: Compile the model in the main program to make it easy to test with various loss functions
		#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		
		#model.summary()
		
		return model
			
		
		
		
		
		
		# create model
		self.model = tf.keras.models.Sequential()

		#convolution layers
		self.model.add(tf.keras.layers.Conv2D(28, (3,3), activation = 'relu', input_shape = (28, 28, 1)))
		self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		self.model.add(tf.keras.layers.Conv2D(56, (3,3), activation = 'relu'))
		self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
		self.model.add(tf.keras.layers.Conv2D(56, (3,3), activation = 'relu'))

		#convert 3D matrix into 2D array
	#	model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
		model.add(tf.keras.layers.Flatten())

		#dense layer for classification
		model.add(tf.keras.layers.Dense(64, activation='relu'))
		model.add(tf.keras.layers.Dense(10, activation='softmax'))
		
		model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
		
	def summary(self):
		self.model.summary()
		
	def train_mnist(self, epochs):
		# mnist is handwritten picture dataset
		# loading in dataset from tensorflow
		self.mnist = tf.keras.datasets.mnist
		
		# split into training data and testing data (about 80 20 split)
		# x_train for pixel data, y_train for classification data
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
		
		# scale down pixel data (like integer 0-255 for grey scale)
		self.x_train = tf.keras.utils.normalize(self.x_train,axis=1)
		self.x_test = tf.keras.utils.normalize(self.x_test,axis=1)
	
		#fit / train the model
		self.model.fit(x_train, y_train, epochs)
		self.model.save('handwritten.model')

	# print the loss and accuracy of the model
	def test(self):
		loss, accuracy= self.model.evaluate(x_test, y_test)
		print("loss is: " + str(loss))
		print("accuracy is: " + str(accuracy))

	# return the estimated result 
	def estimate(self):
		image_number = 1
		while os.path.isfile(estimate_data_path + f"digit{image_number}.png") and image_number < 6:
			try:
				# only get grey scale
				img_grey = cv2.imread(estimate_data_path + f"digit{image_number}.png")[:,:,0] 
				
				# invert image, get it into an array, and predict
				img_grey = np.invert(np.array([img_grey]))
				prediction = self.model.predict(img_grey)
		
				# return with result with highest activation
				plt.imshow(img_grey[0], cmap = plt.cm.binary)
				plt.title(f"This digit is: {np.argmax(prediction)}")
				plt.show()
	#			print(f"This digit is: {np.argmax(prediction)}")
			except:
				print("Error")
			finally:
				image_number += 1

##################################################################
# Main function starts HERE

# Uncomment the functions to see the results
#color_threshold()
#background_subtraction()
edge_detection()
#fz = fuzzy_logic_controller() # Fuzzy controller usage example
#fz.estimate_with_passenger_floor_ratio()
#nn = neural_network_handler()
#nn.estimate()

# testing edge detct on real-life images
# image_path = main_path + "training data/lift test 1/lift1.jpg"
# image = cv2.imread(image_path)
# image_edge = threshold_edge(image)
# image_contour = contour_edge(image_edge) 

# For tensorflow neural network things...
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'





