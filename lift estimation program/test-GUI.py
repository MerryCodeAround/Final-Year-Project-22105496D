import tkinter as tk
from tkinter import ttk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the main window
window = tk.Tk()
window.geometry("1280x720")
window.title("Estimation of Available Space in Elevators using Image Processing and Machine Learning")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(window)
notebook.pack(padx=10, pady=10)

# Define the root folder path
root_folder = r"C:\Users\Morris\Documents\12 Programming - Github\Final-Year-Project\training data\lift test 4\sequence.23"

# Get the list of image files in the root folder
image_files = [file for file in os.listdir(root_folder) if file.endswith(".png") or file.endswith(".jpg")]

# Create the tabs
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)
tab4 = ttk.Frame(notebook)

# Add the tabs to the notebook
notebook.add(tab1, text="Colour Thresholding")
notebook.add(tab2, text="Background Subtraction")
notebook.add(tab3, text="Semantic Segmentation")
notebook.add(tab4, text="Instance Segmentation")

# Tab 1
# Create the frames for Tab 1
up_frame_tab1 = ttk.Frame(tab1)
bottom_frame_tab1 = ttk.Frame(tab1)
up_frame_tab1.pack(pady=10)
bottom_frame_tab1.pack(pady=10)

# Create the images and captions for Tab 1
images_tab1 = []
for i in range(3):
    image_path = os.path.join(root_folder, image_files[i])
    image = tk.PhotoImage(file=image_path)
    images_tab1.append(image)
    image_label = tk.Label(up_frame_tab1)
    image_label.pack(side=tk.LEFT, padx=10)

# Create the graph for Tab 1
fig_tab1 = plt.figure(figsize=(5, 4), dpi=100)
ax_tab1 = fig_tab1.add_subplot(111)
ax_tab1.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
ax_tab1.set_xlabel("x")
ax_tab1.set_ylabel("y")
canvas_tab1 = FigureCanvasTkAgg(fig_tab1, bottom_frame_tab1)
canvas_tab1.draw()
canvas_tab1.get_tk_widget().pack()

# Tab 2 (Same as Tab 1)
up_frame_tab2 = ttk.Frame(tab2)
bottom_frame_tab2 = ttk.Frame(tab2)
up_frame_tab2.pack(pady=10)
bottom_frame_tab2.pack(pady=10)

images_tab2 = []
for i in range(3):
    image_path = os.path.join(root_folder, image_files[i])
    image = tk.PhotoImage(file=image_path)
    images_tab2.append(image)
    image_label = tk.Label(up_frame_tab2)
    image_label.pack(side=tk.LEFT, padx=10)

fig_tab2 = plt.figure(figsize=(5, 4), dpi=100)
ax_tab2 = fig_tab2.add_subplot(111)
ax_tab2.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
ax_tab2.set_xlabel("x")
ax_tab2.set_ylabel("y")
canvas_tab2 = FigureCanvasTkAgg(fig_tab2, bottom_frame_tab2)
canvas_tab2.draw()
canvas_tab2.get_tk_widget().pack()

# Tab 3
# Create the frame for Tab 3
frame_tab3 = ttk.Frame(tab3)
frame_tab3.pack(pady=10)

# Create the images and captions for Tab 3
images_tab3 = []
for i in range(3):
    image_path = os.path.join(root_folder, image_files[i])
    image = tk.PhotoImage(file=image_path)
    images_tab3.append(image)
    image_label = tk.Label(frame_tab3)
    image_label.pack(side=tk.LEFT, padx=10)

# Tab 4 (Same as Tab 3)
frame_tab4 = ttk.Frame(tab4)
frame_tab4.pack(pady=10)
images_tab4 = []
for i in range(3):
    image_path = os.path.join(root_folder, image_files[i])
    image = tk.PhotoImage(file=image_path)
    images_tab4.append(image)
    image_label = tk.Label(frame_tab4)
    image_label.pack(side=tk.LEFT, padx=10)

# Run the GUI
window.mainloop()