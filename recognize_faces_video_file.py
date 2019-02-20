# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0

import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct arg parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize pointer to vid file and vid writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

# loop over frames from vid file stream
while True:
	# grab next frame
	(grabbed, frame) = stream.read()

	# if frame was not grabbed, then we have reached the end of stream
	if not grabbed:
		break

	# convert input frame from BGR to RGB then resize it to width of 750px
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect(x, y)-coords of bounding boxes corr. each face in input frame, then compute facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over facial embeddings
	for encoding in encodings:
		# attempt to match each face in input image to known encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check if match
		if True in matches:
			# find indexes of all matched faces, initialize dict to count total numtimes face matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over matched indexes and maintain count for each recognized face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine recognized face with largest num votes
			name = max(counts, key=counts.get)
		
		# update list of names
		names.append(name)

	# loop over recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw predicted face name on img
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# if vid writer is None && writing output vid to disk, init writer	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 24,
			(frame.shape[1], frame.shape[0]), True)

	# if writer is not None, write frame with recognized faces 
	if writer is not None:
		writer.write(frame)

	# check if displaying output frame to screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if `q` key was pressed, break 
		if key == ord("q"):
			break

# close vid file pointers
stream.release()

# check if the vid writer point needs to be released
if writer is not None:
	writer.release()
