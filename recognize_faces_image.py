# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

import face_recognition
import argparse
import pickle
import cv2

# construct arg parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect (x, y)-coords of bounding boxes corresponding to each face in input image, then compute facial embeddings
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize list of names for each face detected
names = []

# loop over facial embeddings
for encoding in encodings:
	#  match each face in input image to known encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	# check if a match
	if True in matches:
		# find indexes of all matched faces, initialize dict to count total numtimes each face matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over matched indexes and maintain count for each recognized face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine recognized face with largest numvotes
		name = max(counts, key=counts.get)
	
	# update list of names
	names.append(name)

# loop over recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw predicted face name on img
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show output img
cv2.imshow("Image", image)
cv2.waitKey(0)
