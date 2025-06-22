import moondream as md
from PIL import Image, ImageDraw
import os

# This will run the model locally
#model = md.vl(endpoint="http://localhost:2020/v1")

# For Moondream Cloud, use your API key:
model = md.vl(api_key=os.getenv("MOONDREAM_TOKEN"))


# Load an image
base_image = Image.open("./images/street7.jpg")

# Example: Generate a caption
#caption_response = model.caption(base_image, length="short")
#print(caption_response["caption"])

print('Testing detection')
detection_response = model.detect(base_image, "trash bags")
print(detection_response)

#question = "is the trash taking a significant amount of space in a way that is making it difficult for pedestrians to pass?"    
question = "how many trash bags are there?"
#question = "are there several trash bags piled up on the sidewalk?"
#question = "are there several trash bags piled up on the sidewalk that are not in trash containers?"
#question = "are there more than 5 trash bags piled up on the sidewalk"
#question = "are there more than 5 trash bags"
print(f'testing query -> {question}')
query_response = model.query(base_image, question)
print(query_response)


# Create a copy of the base image to draw on
overlay_image = base_image.copy()
draw = ImageDraw.Draw(overlay_image)

# Get image dimensions
img_width, img_height = base_image.size

# Draw hollow rectangles for each detected object
for obj in detection_response["objects"]:
    # Convert normalized coordinates to pixel coordinates
    x_min = int(obj["x_min"] * img_width)
    y_min = int(obj["y_min"] * img_height)
    x_max = int(obj["x_max"] * img_width)
    y_max = int(obj["y_max"] * img_height)
    
    # Draw hollow rectangle with 5-pixel border
    border_width = 5
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=border_width)

# Save the result
overlay_image.save("./images/img_with_detections.jpg")
print("Image with detection overlays saved as 'frieren_with_detections.jpg'")