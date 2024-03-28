# Final Year Project
 Hi! I am Morris. This is my Final year project of electrical engineering degree at PolyU, about image processing and computer vision. The title is "**Estimation of Available Space in Elevator using Image Processing and Machine Learning**"
 
###Repository Contents
- [Generating Synthetic Data with Unity](#Unity-Project)
- [Training Data](#Training-Data)
- [lift estimation program](#lift-estimation-program)
- Pictures (storing images in this README)

### Unity Project
Contain all the custome code and scene setups. This is a Unity Universal Rendering Pipeline from the templates. The assets used are:
- Unity perception package: https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/index.html
- 3D passenger models: https://sketchfab.com/renderpeople
- 3D lift model: created with Blender

> [!NOTE]
> The lift model is imported with .obj file only since the texture and lighting are not very good if they are imported in .fbx.

An area light and a directional light are place on top of the lift. Their settings are:

<img src="/pictures/areaLightSetting.jpg" height="500"><img src="/pictures/directionalLightSetting.jpg" height="500">

The main camera and scenario game object and hierarchy. Basically, each 'pos' are the center of a rectangle, and the passenger is spawn randomly within the rectangle. The spawning is done by the "mySpawnRandomizer.cs" and "mySpawnRandomizerTag.cs"

<img src="/pictures/mainCameraSetting.jpg" height="400"><img src="/pictures/scenarioSetting.jpg" height="400"><img src="/pictures/hierarchySetting.jpg" height="400">

### Training data
Contain all results from image generation from Unity project. Each iteration contain 4 files:

| Image 1 | Image 2 |
|---------|---------|
| ![Image 1](/pictures/camera.jpg) | ![Image 2](/pictures/passengerMack.png) |
| Caption for Image 1: The original input image. | Caption for Image 2: The result of edge detection applied to the input image. |

| Image 3 | Image 4 |
|---------|---------|
| ![Image 3](/pictures/floorMack.png) | ![Image 4](jsonFile.jpg) |
| Caption for Image 3: The binary mask obtained after thresholding the edge image. | Caption for Image 4: The final segmented output image after post-processing. |

<img src="/pictures/floorMask.png" width="50%"><img src="/pictures/passengerMask.png" width="50%"><img src="/pictures/camera.png" width="50%"><img src="/pictures/jsonFile.jpg" width="50%">

|Training Data|Description|
|---|:---|
|lift test 1|originally plan to use real-life footage of lift CCTV camera|
|lift test 2|green floor, 320 x 180 px, 200 samples|
|lift test 3|grey floor, 960 x 540 px, 50 samples|
|lift test 4|grey floor, 320 x 180 px, 1000 samples, with bounding box|

### Python Codes
Contain all image processing and computer vision python code 

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
- tried to implement neural network for detecting handwritten numbers
- tried to edit the network to detect person on lift
- IDFK
- Version 4d combined all previous work

**Ver 5**
- using YOLO v8.1 for detecting person on lift
- https://github.com/ultralytics/ultralytics?tab=readme-ov-file
- it has pre-trained model for detecting people
- also added GUI with Tkinter library




