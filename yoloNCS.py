from mvnc import mvncapi as mvnc
from yolo_utils import preprocess_image, interpret_output, generate_colors
from datetime import datetime
import numpy as np
import sys
from yolo_config import cfg

class YoloNCS():

    def __init__(self, ncs_graph_file):
        """
        Initialize the graph.

        """
        network_blob = ncs_graph_file
        # configuration NCS
        mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
        	print('No devices found')
        	quit()

        self.device = mvnc.Device(devices[0])

        #try to close the NVC device in case it has not been properly closed before
        #try:
        #    self.device.CloseDevice()
        #except Exception as e:
        #    print('No device to be shutdown: {}'.format(e))

        try:
            self.device.open()
        except Exception as e:
            print('Error opening NCS device: {}'.format(e))

        # load blob
        with open(network_blob, mode='rb') as f:
        	blob = f.read()

        self.graph = mvnc.Graph('YOLONet')

        # Allocate the graph to the device
        self.input_fifo, self.output_fifo = self.graph.allocate_with_fifos(self.device, blob)

        # Generate colors for drawing bounding boxes.
        self.colors = generate_colors(cfg.CLASSES)

    def close_ressources(self):
        """
        Close all output device.

        """
        # deallocate ressources
        self.input_fifo.destroy()
        self.output_fifo.destroy()
        self.graph.destroy()
        self.device.close()
        self.device.destroy()

    def process_image(self, cv2_image):
        """
        Processes image to image

        """
        # preprocess cv2 image (resize, reshape...)
        im = preprocess_image(cv2_image)

        start = datetime.now()
        # start MOD

        # Queue the inference
        self.graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, im.astype(np.float32), None)

        # Get the results from the output queue
        out, userobj = self.output_fifo.read_elem()
        #print('Graph output shape = {}'.format(out.shape))
        #
        end = datetime.now()
        elapsedTime = end-start
        print ('process {} image in {} milliseconds'.format(cv2_image.shape, elapsedTime.total_seconds()*1000))
        results = interpret_output(out.astype(np.float32), cv2_image.shape[1], cv2_image.shape[0]) # fc27 instead of fc12 for yolo_small

        return results
