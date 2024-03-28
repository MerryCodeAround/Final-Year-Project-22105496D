# Final Year Project
 Hi! I am Morris. This is my Final year project of electrical engineering degree at PolyU, about image processing and computer vision. The title is "**Estimation of Available Space in Elevator using Image Processing and Machine Learning**". Since the code, dataset, and results are in different file formats, I decided to create this Github repository to better manage the project structure. Nonetheless, the directory in the program files are probably wrong, so... ~~it's just for show~~
 
### Repository Contents
- [Generating Synthetic Data with Unity](#Unity-Project)
- [Training Data](#Training-Data)
- [lift estimation program](#lift-estimation-program)
- Pictures (storing images in this README)

## Unity Project
Contain all the custome code and scene setups. This is a Unity Universal Rendering Pipeline from the templates. The assets used are:
- Unity perception package: https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/index.html
- 3D passenger models: https://sketchfab.com/renderpeople
- 3D lift model: created with Blender

The lift model is imported with .obj file only since the texture and lighting are not very good if they are imported in .fbx. An area light and a directional light are place on top of the lift. Their settings are:

| **Area Light Setting** | **Direction Light Setting** |
| :---: | :---: |
| ![Image 1](/pictures/areaLightSetting.jpg) | ![Image 2](/pictures/directionalLightSetting.jpg) |

The main camera and scenario game object and hierarchy. Basically, each 'pos' are the center of a rectangle, and the passenger is spawn randomly within the rectangle. The spawning is done by the "mySpawnRandomizer.cs" and "mySpawnRandomizerTag.cs", found within the folder. The following images showed the perception camera setting, scenario setting, and hierarchy of the 3D scene. 

| **Perception Camera Setting** | **Scenario Setting** | **Hierarchy Setting** |
| :---: | :---: | :---: |
| ![Image 1](/pictures/mainCameraSetting.jpg) | ![Image 2](/pictures/scenarioSetting.jpg) | ![Image 3](/pictures/hierarchySetting.jpg)

## Training data
Contain all results from image generation from Unity project. Each iteration contain 4 files:

| **Synthetic Image** | **Passenger Mask** |
| :---: | :---: |
| ![Image 1](/pictures/camera.png) | ![Image 2](/pictures/passengerMask.png) |
| **Floor Mask** | **JSON file (annotations)** |
| ![Image 3](/pictures/floorMask.png) | ![Image 4](/pictures/jsonFile.jpg) |

There several iteration of dataset at different stages of the project, each with their own descriptions

|Training Data|Description|
|---|:---|
|lift test 1|originally plan to use real-life footage of lift CCTV camera|
|lift test 2|green floor, 320 x 180 px, 200 samples|
|lift test 3|grey floor, 960 x 540 px, 50 samples|
|lift test 4|grey floor, 320 x 180 px, 1000 samples, with bounding box|

Notice that for "lift test 4", to accomodate for YOLO model training, it is rearranged to the following file structure. 
![Image 1](/pictures/YOLOformat.png)

## Python Codes
I was originally using Notepad++ as my text editor, and command prompt to execute the files. But I was met with version conflicts among python modules, and the "NppExec Console" failed me. So I switched to using Anaconda and Spyder for my development environment. My environment is saved as "my_environment.yaml" for your reference, so it saves time to install different python modules. 

After setting up Unity, we can finally open the program in this repository. It contains all the image processing and computer vision python code in ascending versions. 

**Ver 1**
- read images
- retrieve passenger number, actual floor area from json file
- estimate floor area using colour thresholding
- plot scatter dot diagram for estimation and actual floor area
- plot linear regressed diagram for estimation and actual floor area 

**Ver 2**
- added background subtraction (useless)
- added edge detection to extract floor area (useless)

**Ver 3**
- tried using fuzzy logic controller. original idea is to:
	- input the floor area 
	- input passenger area 
	- create membership, and fuzzy rule
	- generate result
- but don't know how to implement (how to set the fuzzy rule and output)

**Ver 4**
- tried to combine all previous works
- to implement neural network from scratch (failed miserably)

**Ver 5**
- using YOLO v8 for detecting person on lift, their website: https://github.com/ultralytics/ultralytics?tab=readme-ov-file
- it has pre-trained model for detecting people
- also added GUI with Tkinter library
- also uses custom trained model




