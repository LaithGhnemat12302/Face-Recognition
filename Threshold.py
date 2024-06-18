import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_face_encodings(folder_path):
    face_encodings = []
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])
    return face_encodings

def calculate_fmr_fnmr(genuine_pairs, imposter_pairs, threshold):
    false_non_matches = 0
    false_matches = 0

    # Calculate FNMR (False Non-Match Rate)
    for enc1, enc2 in genuine_pairs:
        distance = np.linalg.norm(enc1 - enc2)
        if distance > threshold:
            false_non_matches += 1

    # Calculate FMR (False Match Rate)
    for enc1, enc2 in imposter_pairs:
        distance = np.linalg.norm(enc1 - enc2)
        if distance <= threshold:
            false_matches += 1

    total_genuine_pairs = len(genuine_pairs)
    total_imposter_pairs = len(imposter_pairs)

    fnmr = false_non_matches / total_genuine_pairs
    fmr = false_matches / total_imposter_pairs

  

    return fmr, fnmr


def calculate_distance_matrix(encodings):
    num_encodings = len(encodings)
    distance_matrix = np.zeros((num_encodings, num_encodings))

    for i in range(num_encodings):
        for j in range(i + 1, num_encodings):
            distance = np.linalg.norm(encodings[i] - encodings[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


# Load encodings from images
folder_path = 'persons'
face_encodings = load_face_encodings(folder_path)

# Generate genuine pairs and imposter pairs
genuine_pairs = [(face_encodings[i], face_encodings[j]) for i in range(len(face_encodings)) for j in range(i+1, len(face_encodings))]
imposter_pairs = [(face_encodings[i], face_encodings[j]) for i in range(len(face_encodings)) for j in range(len(face_encodings)) if i != j]

# Calculate FMR and FNMR for different thresholds
thresholds = [0.2, 0.4, 0.6, 0.8]
fmr_values = []
fnmr_values = []

distance_matrix = calculate_distance_matrix(face_encodings)
print("Distance Table ")
df_distance_matrix = pd.DataFrame(distance_matrix)

# Set index and column names
df_distance_matrix.columns = [f"Encoding {i}" for i in range(len(df_distance_matrix))]
df_distance_matrix.index = [f"Encoding {i}" for i in range(len(df_distance_matrix))]

# Print the distance matrix
print("Distance Matrix:")
print(df_distance_matrix.to_string())


for threshold in thresholds:
    fmr, fnmr = calculate_fmr_fnmr(genuine_pairs, imposter_pairs, threshold)
    fmr_values.append(fmr)
    fnmr_values.append(fnmr)
    print(f"Threshold {threshold} - False Match Rate (FMR): {fmr:.4f}, False Non-Match Rate (FNMR): {fnmr:.4f}")

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
