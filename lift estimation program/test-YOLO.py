from ultralytics import YOLO
import cv2 
import os

# pretrained
model = YOLO("yolov8n.pt")

# main folder
folder_path = "C:/Users/Morris/Documents/12 Programming - Github/Final-Year-Project/training data/lift test 6"

# Loop through main folder
counter = 0
for root, dirs, files in os.walk(folder_path):
	
	# loop through sub folders
	for dir_name in dirs:
		subfolder_path = os.path.join(root, dir_name)
		#print('Subfolder:', subfolder_path)
		
		# loop through files
		for file_name in os.listdir(subfolder_path):
			if file_name == "step1.camera.png":
				file_path = os.path.join(subfolder_path, file_name)
				data = cv2.imread(file_path)
				#print('File:', file_path)
				results = model.predict(source = file_path, show = True)
				# Wait for the 'q' key to be pressed
				while True:
					if cv2.waitKey(0) or 0xFF == ord('q'):
						break
				cv2.destroyAllWindows()
							
		counter += 1

		if counter > 10:
			break
			