import sys, cv2, os
from yoloNCS import YoloNCS
from yolo_utils import show_results
import time


def read_camera(camera, yoloNCS):

    while True:
        img = camera.get_frame_cv2_format()

        if img is not None:
            #process Ylol algorithm with NCS Graph
            results = yoloNCS.process_image(img)

            #print (results)
            #cv2.imshow('YOLO detection',img_cv)
            show_results(img, results, img.shape[1], img.shape[0], yoloNCS.colors)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # deallocate ressources
            yoloNCS.close_ressources()
            break

def get_camera():
    if os.uname()[4].startswith("arm"):
        #Raaspberry Pi version
        print('Use R-Pi camera')
        from streaming.camera_pi import VideoCameraPi
        camera = VideoCameraPi()
    else:
        # laptop internal webcam
        print('Use laptop webcam')
        from streaming.camera import VideoCamera
        camera = VideoCamera()
    return camera


if __name__ == '__main__':
    # load NCS graph
    yoloNCS = YoloNCS('graphs/graph_ncappzoo')

    # get the right camera
    camera = get_camera()

    #wait camera has started
    time.sleep(1)

    read_camera(camera, yoloNCS)
    #read_camera(camera, None)
