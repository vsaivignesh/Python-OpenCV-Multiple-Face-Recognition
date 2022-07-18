import cv2
import numpy as np
import face_detect as face_detect
import training_data as training_data
from google.colab.patches import cv2_imshow


label = []
def predict(test_img):
	img = cv2.imread(test_img).copy()
	print("\n \n \n")
	print("Face Prediction Running -\-")
	face, rect, length = face_detect.face_detect(test_img)
	print(len(face), "faces detected.")
	for i in range(0, len(face)):
		labeltemp, confidence = face_recognizer.predict(face[i])
		label.append(labeltemp)
	return img, label

faces, labels = training_data.training_data("/content/Python-OpenCV-Multiple-Face-Recognition/training-data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


# Read the test image.
test_img = "/content/Python-OpenCV-Multiple-Face-Recognition/test-data/test.jpg"
predicted_img , label= predict(test_img)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
print("Recognized faces = ", label)
