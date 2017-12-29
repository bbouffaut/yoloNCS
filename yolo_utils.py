import numpy as np
import cv2
from skimage.transform import resize
from yolo_config import cfg
import colorsys
import random

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def preprocess_image(cv2_image):
    # image preprocess
    dim = cfg.MODEL_INPUT_SIZE
    img = cv2_image
    im = resize(img.copy()/255.0,dim,1)
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = im[:,:,(2,1,0)]
    #print('NEW shape:',im.shape)
    #print(img[0,0,:],im[0,0,:])
    return im

def preprocess_boxes(output, img_width, img_height, num_class=20, num_box=2, grid_size=7):
    # extract boxes from output
    w_img = img_width
    h_img = img_height
    #debug
    #print ((w_img, h_img))

    boxes = (np.reshape(output[1078:],(grid_size, grid_size, num_box, 4)))#.copy()
    offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(2,grid_size,grid_size)),(1,2,0))
    #boxes.setflags(write=1)
    boxes[:,:,:,0] += offset
    boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
    boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
    boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
    boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

    boxes[:,:,:,0] *= w_img
    boxes[:,:,:,1] *= h_img
    boxes[:,:,:,2] *= w_img
    boxes[:,:,:,3] *= h_img

    return boxes

def filter_boxes(output, boxes, grid_size, num_box, num_class, threshold=0.2):

    # Filter boexes based on computed probability and threshol

    probs = np.zeros((grid_size,grid_size,num_box,num_class))
    class_probs = (np.reshape(output[0:980],(grid_size,grid_size,num_class)))#.copy()
    #print(class_probs)
    scales = (np.reshape(output[980:1078],(grid_size,grid_size, num_box)))#.copy()
    #print(scales)

    for i in range(2):
    	for j in range(20):
    		probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
    #print (probs)
    filter_mat_probs = np.array(probs>=threshold,dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    return probs_filtered, boxes_filtered, classes_num_filtered

def iou_filter(probs_filtered, boxes_filtered, classes_num_filtered, iou_threshold=0.5):

    for i in range(len(boxes_filtered)):
    	if probs_filtered[i] == 0 : continue
    	for j in range(i+1,len(boxes_filtered)):
    		if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold :
    			probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    return probs_filtered, boxes_filtered, classes_num_filtered

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


def show_results(img, results, img_width, img_height, colors, imshow=True):
    img_cp = img.copy()
    disp_console = True

    #	if self.filewrite_txt :
    #		ftxt = open(self.tofile_txt,'w')
    for i in range(len(results)):
    	x = int(results[i][1])
    	y = int(results[i][2])
    	w = int(results[i][3])//2
    	h = int(results[i][4])//2
    	if disp_console : print ('    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5]) )
    	xmin = x-w
    	xmax = x+w
    	ymin = y-h
    	ymax = y+h
    	if xmin<0:
    	    xmin = 0
    	if ymin<0:
    	    ymin = 0
    	if xmax>img_width:
    	    xmax = img_width
    	if ymax>img_height:
    	    ymax = img_height

        # draw boxes and class_name
    	cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),colors[cfg.CLASSES.index(results[i][0])],2)
    	#print ((xmin, ymin, xmax, ymax))
    	#cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
    	cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[cfg.CLASSES.index(results[i][0])],2)

    if imshow:
        cv2.imshow('YOLO detection',img_cp)
        cv2.waitKey(1000)

    else:
        return img_cp

def interpret_output(output, img_width, img_height):
    classes = cfg.CLASSES
    threshold = cfg.BOXES_FILTERING_THRESHOLD
    iou_threshold = cfg.IOU_FILTERING_THRESHOLD
    num_class = cfg.NUM_CLASSES
    num_box = cfg.NUM_BOXES
    grid_size = cfg.GRID_SIZE

    # extract boxes from network output
    boxes = preprocess_boxes(output, img_width, img_height, num_class, num_box, grid_size)

    #filter boxes based on threshold
    probs_filtered, boxes_filtered, classes_num_filtered = filter_boxes(output, boxes, grid_size, num_box, num_class, threshold)

    #filter boxes based on IOU threshold
    probs_filtered, boxes_filtered, classes_num_filtered = iou_filter(probs_filtered, boxes_filtered, classes_num_filtered, iou_threshold)

    # write result
    result = []
    for i in range(len(boxes_filtered)):
    	result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    return result
