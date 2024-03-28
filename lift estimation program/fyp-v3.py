import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Global Variables
FOLDER_PATH = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/"
TRAIN_PATH = FOLDER_PATH + "lift test 5/"
BACKGROUND_PATH = TRAIN_PATH + "background/step0.camera.png" 
result_color = (0, 255, 0) #green
estimate_count = []
result_count = []
passenger_count = []

def grey_scale(img: np.ndarray):
	'''
	Input RGB image
	Returns a grey image
	Algorithm:
		output = 0.2989 * R + 0.5870 * G + 0.1140 * B 
	'''
	r_coef = 0.2989
	g_coef = 0.5870
	b_coef = 0.1140
	r, g, b = img[..., 0], img[..., 1], img[..., 2]
	return r_coef * r + g_coef * g + b_coef * b

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

def threshold_edge(image, threshold_ratio):
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
	
def threshold_edge_test(image):
	image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #greyscale
	image_edge = cv2.adaptiveThreshold(image_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #gaussian
	return image_edge
	
def contour_area(image_edge):
	cv2.imshow('edge', image_edge)
	cv2.waitKey(0)
	
	image_contour = np.zeros_like(image_edge)
	contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		epsilon = 0.001 * cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, epsilon, True)
		if cv2.contourArea(approx) > 3000: #apply size filter
			if cv2.arcLength(approx, True): #whether closed shape or not
				image_contour = np.zeros_like(image_edge)
				cv2.drawContours(image_contour, [approx], 0, 255, 1) 
				cv2.imshow('contour', image_contour)
				cv2.waitKey(0)
				cv2.destroyAllWindows() 
	
	# contours, hierarchy = cv2.findContours(image_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# for contour in contours:
		# epsilon = 0.01 * cv2.arcLength(count, True)
		# approximations = cv2.approxPolyDP(contour, epsilon, True)
		# cv2.drawContours(imageread, [approximations], 0, (0), 3)
	# print("number of contours = " + str(len(contours)))
	# cv2.drawContours(image_edge, contours, -1, (0, 255, 0), 3)
	cv2.imshow('contours', image_contour)
	cv2.waitKey(0) 
	cv2.destroyAllWindows() 

def estimate_lift_area(algorithm, image, background):
	'''
	estimate the lift area with different image processing algorithms, which are 
	- 	colour_thresholding
	-	background_subtraction
	-	edge_detection
	
	the inputs are
	- algorithms	: string for above algorithms
	- image			: the image file for detecting
	- background	: the background file (background_subtraction only)
	
	the outputs are
	- output[0] 	: estimated pixel number 
	- output[1] 	: estimated floor mask
	'''
	
	if algorithm == 'colour_threshold':
		# Convert the image to the HSV color space
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Define the lower and upper threshold for sharp green color
		lower_green = np.array([ 50, 160,   0])  # Adjust these values based on your specific case
		upper_green = np.array([200, 255, 100])  # Adjust these values based on your specific case
		
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
		
		plt.subplot(1,2,1)
		plt.imshow(mask, cmap = 'gray')

		plt.subplot(1,2,2)
		plt.imshow(image)
		plt.show()
		
		return int(pixel_count), mask
		
	if algorithm == 'adaptive_colour_threshold':
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

	elif algorithm == 'background_subtraction':
		# Convert to HSV colour space
		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
	
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
		pixel_count = np.sum(mask / 255)
		# Threshold the difference image (for greyscale image)
		#_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

		# display mask against actual
		# plt.subplot(1,2,1)
		# plt.imshow(mask, cmap = 'gray')

		# plt.subplot(1,2,2)
		# plt.imshow(image)
		# plt.show()
		
		return int(pixel_count), mask
		
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
	'''
	return the number of passenger from json file
	'''
	
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

def print_results(estimate, result, passenger):
	# Display results
	counter2 = 0
	for counter2 in range(len(estimate)):
		print("Picture", counter2, ": Estimtate:", estimate[counter2], " Actual:", result[counter2], " Passenger:", passenger[counter2])
		counter2 += 1

def plot_result(estimate, result, passenger):
	'''
	Plot estimation against result with linear regression
	- red = estimation
	- blue = actual
	'''
	
	m1, b1 = np.polyfit(passenger, estimate, 1)
	m2, b2 = np.polyfit(passenger, result, 1)
	regression_line1 = np.add(np.multiply(m1, passenger), b1) # y = m1 * passenger_count + b1
	regression_line2 = np.add(np.multiply(m2, passenger), b2) # y = m2 * passenger_count + b2
	plt.scatter(passenger, estimate, color = 'red')
	plt.scatter(passenger, result, color = 'blue')
	plt.plot(passenger, regression_line1, color='red', label='Linear Regression Line 1')
	plt.plot(passenger, regression_line2, color='blue', label='Linear Regression Line 2')
	plt.show()

def color_threshold():
	'''
	Main colour thresholding algorithm for estimating lift area
	Will plot a graph of linear regression line
	'''

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
		plt.imshow(image)
		result = cv2.imread(result_path)
		f = open(data_path)
		data = json.load(f)
		
		# Estimate the floor area
		floor_area, _ = estimate_lift_area('colour_threshold', image, image)
		estimate_count.append(floor_area)

		# Count the actual floor area
		result_count.append(count_color_area(result, result_color))

		# Count the actual passengers
		passenger_count.append(count_passenger(data))

		f.close()
		counter1 += 1

	plot_result(estimate_count, result_count, passenger_count)

def background_subtraction():
	'''
	Main background subtraction algorithm for estimating lift area
	Will plot a graph of linear regression line
	'''

	# Iterate over each file in the folder
	file_list = os.listdir(TRAIN_PATH)
	counter1 = 0
	
	# Get Background image
	background = cv2.imread(BACKGROUND_PATH)
	background_grey = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) # convert to greyscale
	
	height, width, _ = background.shape
	total_pixel = height * width
	
	for counter1 in range(200):
		# set file path
		image_path = "{}sequence.{}/step0.camera.png".format(TRAIN_PATH, counter1)
		result_path = "{}sequence.{}/step0.camera.passenger.png".format(TRAIN_PATH, counter1)
		data_path = "{}sequence.{}/step0.frame_data.json".format(TRAIN_PATH, counter1)
		
		# load file data 
		image = cv2.imread(image_path)
		result = cv2.imread(result_path)
		f = open(data_path)
		data = json.load(f)

		# Estimate the passenger area with background subtraction
		area, mask = estimate_lift_area('background_subtraction', image, background)
		estimate_count.append(area)

		# Count the actual passenger area
		result_count.append(total_pixel - count_color_area(result, (0,0,0)))
		
		# Count the actual passengers
		passenger_count.append(count_passenger(data))

		f.close()
		counter1 += 1
	
	# plot the estimation against result
	plot_result(estimate_count, result_count, passenger_count)
	
def edge_detect():
	'''
	Main edge detection algorithm for estimation lift area
	'''
	# Iterate over each file in the folder
	file_list = os.listdir(FOLDER_PATH)
	counter1 = 0
	
	# Get Background image infomation
	background = cv2.imread(BACKGROUND_PATH)

	# Apply edge detect on background image
	#background_edge = sobel_edge(background)
	#background_edge = canny_edge(background, sigma = 1, low_threshold = 0, high_threshold = 15)
	#threshold_edge(background, threshold_ratio = 0.7)
	background_edge = threshold_edge_test(background)
	floor_area = contour_area(background_edge)

class FuzzyLogicController:
	
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



# Main function starts HERE

#color_threshold() #for floor area
background_subtraction() #for passenger area
#edge_detect()

# Fuzzy controller usage example
# controller = FuzzyLogicController()
# floor_area_input = 60
# passenger_area_input = 70
# controller.estimate_with_passenger_floor_ratio()

#print("Estimated number of passengers:", estimated_num_passengers)

