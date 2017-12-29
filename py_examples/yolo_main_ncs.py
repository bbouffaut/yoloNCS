import sys, cv2
from yoloNCS import YoloNCS
from yolo_utils import show_results

# main loop
if len(sys.argv) != 2:
	print ("YOLOv1 Tiny example: python3 py_examples/yolo_example.py images/dog.jpg")
	sys.exit()

# load NCS graph
yoloNCS = YoloNCS('graph')

# read image
img = cv2.imread(sys.argv[1])

#process Ylol algorithm with NCS Graph
results = yoloNCS.process_image(img)

#print (results)
#cv2.imshow('YOLO detection',img_cv)
show_results(img, results, img.shape[1], img.shape[0])
cv2.waitKey(10000)

# deallocate ressources
yoloNCS.close_ressources()
