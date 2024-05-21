import json

# Load the JSON data
with open("D:\\Datasets\\karate\\Synchronized\\Calibration\\K4A_Tino-ReferenceFrame3_AntonyTool.json") as f:
    data = json.load(f)

img_height = 1080

# Initialize the arrays
image_coordinates = []
world_coordinates = []

# Iterate over the data and append the coordinates to the respective arrays
for item in data:
    image_coordinates.append([int(item['ImageCoordinates']['x']), img_height-int(item['ImageCoordinates']['y'])])
    #world_coordinates.append(item['WorldCoordinates'])

# Now image_coordinates and world_coordinates are ordered arrays of coordinates
print(image_coordinates)
#print(world_coordinates)