import cv2  # OpenCV library for image processing tasks
import numpy as np  # numpy library for numerical computations
import face_recognition  # the main library for recognition functionalities
import os  # operating system library for file system interactions

path = 'persons'  # variable stores the directory path where your face images are stored (e.g., 'persons').
images = []
classNames = []
personList = os.listdir(path)  # stores the filenames of all images in the 'persons' directory.
print(personList)

for cl in personList:
    curImg = cv2.imread(f'{path}/{cl}')  # read image using Cv2.imread 
    images.append(curImg)  # appends the image to the image list 
    classNames.append(os.path.splitext(cl)[0])  # Extracts the name of the person from the filename

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts the image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]  # extract encodings
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Load the test image
test_img_path = 'Alaa copy.jpg'  # Path to the image you want to test
test_img = cv2.imread(test_img_path)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Detect faces and find encodings in the test image
faces_in_test_img = face_recognition.face_locations(test_img)
encodes_in_test_img = face_recognition.face_encodings(test_img, faces_in_test_img)

# Compare faces and find distances
for encodeFace, faceLoc in zip(encodes_in_test_img, faces_in_test_img):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        print(name)
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(test_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(test_img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

# Resize the image after annotations
output_size = (640, 480)  # Desired dimensions
resized_output_img = cv2.resize(test_img, output_size)
# Display the image
cv2.imshow('Face Recognition', resized_output_img)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Destroy all windows
