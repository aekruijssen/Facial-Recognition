# python encode_faces.py --dataset dataset --encodings encodings.pickle

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct arg parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# paths to input images in dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over img paths
for (i, imagePath) in enumerate(imagePaths):
	# extract person name from img path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load input img and convert it from RGB OpenCV to dlib ordering
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect (x, y)-coords of bounding boxes
	# corresponding to each face in input img
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute facial embedding for face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over encodings
	for encoding in encodings:
		# add each encoding + name to set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
