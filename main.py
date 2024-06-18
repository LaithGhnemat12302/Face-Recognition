import face_recognition
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Function to load images of different formats and convert them to a compatible format
def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img)

# Function to load images from a directory with specified extensions
def load_images_from_dir(directory, extensions):
    images = []
    for ext in extensions:
        images.extend(glob.glob(directory + f"/*.{ext}"))
    return [load_image(img) for img in images]

# Define directory paths and extensions
gallery_dir = "gallery_images"
enrollment_dir = "enrollment_images"
extensions = ["jpg", "jpeg", "png", "webp"]

# Load images from directories
gallery_set = load_images_from_dir(gallery_dir, extensions)
enrollment_set = load_images_from_dir(enrollment_dir, extensions)

# Initialize lists to store distances
distances = []

# Calculate distances between each face in the gallery and enrollment sets
for gallery_face in gallery_set:
    distances_row = []
    for enrollment_face in enrollment_set:
        # Try to compute the face encodings and handle IndexError
        try:
            gallery_encoding = face_recognition.face_encodings(gallery_face)[0]
            enrollment_encoding = face_recognition.face_encodings(enrollment_face)[0]
            print("i")

            # Calculate the Euclidean distance between the face embeddings
            distance = np.linalg.norm(gallery_encoding - enrollment_encoding)
        except IndexError:
            # If no face is found in the image, assign a high distance
            distance = float('inf')
        distances_row.append(distance)
    distances.append(distances_row)

# Print distance table
print("Distance Table:")
for i, row in enumerate(distances):
    print(f"Gallery Image {i+1}:", ["{:.2f}".format(d)    for d in row])

# Set threshold value (you need to set it according to your requirements)
# threshold = 0.6
thresholds = [0.2, 0.4, 0.6, 0.8,0.9]
fmr_values = []
fnmr_values = []
# Calculate False Match Rate (FMR) and False Non-Match Rate (FNMR)
fmr = 0
fnmr = 0
for threshold in thresholds:
    for i, row in enumerate(distances):
        for distance in row:
            if distance <= threshold:
                if i != row.index(distance):
                    fmr += 1
            else:
                if i == row.index(distance):
                    fnmr += 1

    total_comparisons = len(gallery_set) * len(enrollment_set)
    fmr /= (total_comparisons - len(gallery_set))
    fmr_values.append(fmr)
    fnmr /= (len(gallery_set))
    fnmr_values.append(fnmr)
    print(f"\nFMR: {fmr:.4f}")
    print(f"FNMR: {fnmr:.4f}")

# Find the threshold where FMR and FNMR are closest to each other (EER)
eer_threshold = thresholds[np.argmin(np.abs(np.array(fmr_values) - np.array(fnmr_values)))]
eer = fmr_values[np.argmin(np.abs(np.array(fmr_values) - np.array(fnmr_values)))]
print(f"\nEqual Error Rate (EER): {eer:.4f} at threshold: {eer_threshold:.4f}")

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(thresholds, fmr_values, label='FMR', marker='o')
plt.plot(thresholds, fnmr_values, label='FNMR', marker='x')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('FMR and FNMR at Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()

