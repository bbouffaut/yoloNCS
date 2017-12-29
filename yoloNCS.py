from mvnc import mvncapi as mvnc
from yolo_utils import preprocess_image, interpret_output, generate_colors
from datetime import datetime
import numpy as np
import sys
from yolo_config import cfg

class YoloNCS():

    def __init__(self, ncs_graph_file):
        network_blob = ncs_graph_file
        # configuration NCS
        mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
        	print('No devices found')
        	quit()

        self.device = mvnc.Device(devices[0])

        # try to close the NVC device in case it has not been properly closed before
        try:
            self.device.CloseDevice()
        except Exception as e:
            print('No device to be shutdown: {}'.format(e))

        self.device.OpenDevice()
        opt = self.device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
        # load blob
        with open(network_blob, mode='rb') as f:
        	blob = f.read()

        self.graph = self.device.AllocateGraph(blob)
        self.graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
        iterations = self.graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)

        # Generate colors for drawing bounding boxes.
        self.colors = generate_colors(cfg.CLASSES)

    def close_ressources(self):
        # deallocate ressources
        self.graph.DeallocateGraph()
        self.device.CloseDevice()

    def process_image(self, cv2_image):
        # preprocess cv2 image (resize, reshape...)
        im = preprocess_image(cv2_image)

        start = datetime.now()
        # start MOD
        self.graph.LoadTensor(im.astype(np.float16), 'user object')
        out, userobj = self.graph.GetResult()
        #print('Graph output shape = {}'.format(out.shape))
        #
        end = datetime.now()
        elapsedTime = end-start
        print ('process {} image in {} milliseconds'.format(cv2_image.shape, elapsedTime.total_seconds()*1000))
        results = interpret_output(out.astype(np.float32), cv2_image.shape[1], cv2_image.shape[0]) # fc27 instead of fc12 for yolo_small

        return results
