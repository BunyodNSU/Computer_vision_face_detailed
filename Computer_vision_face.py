import face_recognition
import cv2
import numpy as np

actor_image_path = './data/Face_Leo.jpg'

glasses_image_path = './data/Glasses_V1.jpg'
# glasses_image_path = './data/Glasses_V2.png'
# glasses_image_path = './data/Glasses_V3.png'

intermediate_result_path = './data/Actor_with_circles.jpg'
final_result_path = './data/Actor_with_sunglasses.jpg'

# Load the actor's image
actor_image = face_recognition.load_image_file(actor_image_path)
# Convert the image to BGR color space for all OpenCV operations
actor_image_cv = cv2.cvtColor(actor_image, cv2.COLOR_RGB2BGR)

# Find face landmarks
face_landmarks_list = face_recognition.face_landmarks(actor_image_cv)

# Assuming the first set of landmarks corresponds to the actor's face
if face_landmarks_list:
    face_landmarks = face_landmarks_list[0]
    
    # Selecting the face
    chin = face_landmarks['chin']
    np_chin = np.array(chin)
    (x, y, w, h) = cv2.boundingRect(np_chin)
    top_y = max(y - h // 2, 0)
    h_extended = h + h // 2
    face_center = (x + w // 2, top_y + h_extended // 2)
    face_axes = (w // 2, h_extended // 2)
    cv2.ellipse(actor_image_cv, face_center, face_axes, 0, 0, 360, (255, 0, 0), 2)
    
    # Multiplier to increase the size of eye circles
    eye_scale_factor = 2

    for eye in ['left_eye', 'right_eye']:
        np_eye = np.array(face_landmarks[eye])
        eye_center = tuple(np.mean(np_eye, axis=0).astype(int))
        eye_radius = int(max(cv2.boundingRect(np_eye)[2], cv2.boundingRect(np_eye)[3]) * eye_scale_factor / 2)
        cv2.circle(actor_image_cv, eye_center, eye_radius, (0, 255, 0), 2)
else:

    raise ValueError("No faces detected in the image")

cv2.imwrite(intermediate_result_path, actor_image_cv)

# Load sunglasses image with alpha channel (transparency)
sunglasses_img = cv2.imread(glasses_image_path, cv2.IMREAD_UNCHANGED)

# Calculate the position to place the sunglasses
# Averaging the eye centers to find the midpoint between the eyes
left_eye_center = np.mean(face_landmarks['left_eye'], axis=0)
right_eye_center = np.mean(face_landmarks['right_eye'], axis=0)
eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)

# Scale the sunglasses based on the eye width
eye_width = np.linalg.norm(right_eye_center - left_eye_center)
sunglasses_width = int(eye_width * 2)
scale_factor = sunglasses_width / sunglasses_img.shape[1]
resized_sunglasses = cv2.resize(sunglasses_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Calculate the placement coordinates
x_offset = eye_center[0] - resized_sunglasses.shape[1] // 2
y_offset = int(eye_center[1] - resized_sunglasses.shape[0] * 0.4)

has_alpha_channel = resized_sunglasses.shape[2] == 4 if len(resized_sunglasses.shape) > 2 else False

# Define a threshold to identify white pixels
white_threshold = 120

# Function to check if a pixel is considered white
def is_white(pixel, threshold):
    # Ensure the pixel array has at least three elements (RGB)
    if len(pixel) >= 3:
        return np.all(pixel[:3] >= threshold)
    return False 

print("Shape of resized_sunglasses:", resized_sunglasses.shape)
print("Data type of a pixel:", type(resized_sunglasses[0, 0]))

for i in range(resized_sunglasses.shape[0]):
    for j in range(resized_sunglasses.shape[1]):
        # Getting a pixel
        pixel = resized_sunglasses[i, j]

        # Checking if a pixel is white in RGB
        # We assume that resized_sunglasses is a color image
        is_white_pixel = np.all(pixel >= white_threshold)

        if not is_white_pixel:
            # If the pixel is not white, copy it to the actor's image
            actor_image_cv[y_offset + i, x_offset + j] = pixel


# Create a face mask that we will use for blur
face_mask = np.zeros(actor_image_cv.shape[:2], dtype=np.uint8)
# Add a white oval to the mask corresponding to the face area
cv2.ellipse(face_mask, face_center, face_axes, 0, 0, 360, 255, -1)

# For each eye, remove its area from the blur mask using circles
for eye in ['left_eye', 'right_eye']:
    np_eye = np.array(face_landmarks[eye])
    eye_center = tuple(np.mean(np_eye, axis=0).astype(int))
    # Calculating the radius of the circle in the same way as when drawing circles around the eyes (like before)
    eye_radius = int(max(cv2.boundingRect(np_eye)[2], cv2.boundingRect(np_eye)[3]) * eye_scale_factor / 2)
    cv2.circle(face_mask, eye_center, eye_radius, 0, -1)

# Applying Gaussian Blur to an Image
blurred_image = cv2.GaussianBlur(actor_image_cv, (51, 51), 0)

# Apply blur only where the mask allows (the entire face area except the eyes)
actor_image_cv[face_mask == 255] = blurred_image[face_mask == 255]

# Save the final image
cv2.imwrite(final_result_path, actor_image_cv)