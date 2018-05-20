import sys, cv2, signal
from yoloNCS import YoloNCS
from yolo_utils import show_results
import time, os
from flask import Flask, render_template, Response
from flask_script import Manager, Server

app = Flask(__name__)
manager = Manager(app)

class CustomServer(Server):
    def __call__(self, app, *args, **kwargs):
        # init NCS and camera_pi
        init()

        #Hint: Here you could manipulate app
        return Server.__call__(self, app, *args, **kwargs)


class CustomServerRecorder(Server):
    def __call__(self, app, *args, **kwargs):
        # init NCS and camera_pi
        init(True)

        #Hint: Here you could manipulate app
        return Server.__call__(self, app, *args, **kwargs)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera, yoloNCS
    return Response(read_camera(camera, yoloNCS),mimetype='multipart/x-mixed-replace; boundary=frame')


def close():
    global yoloNCS, camera, recorder_output

    print('Closing resources now...')

    #stop recording video
    if recorder_output is not None:
        recorder_output.release()

    #stop camera's thread
    camera.join()

    # stop NCS ressources
    yoloNCS.close_ressources()

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')

    #close ressources
    close()

    #exit
    #sys.exit(0)
    raise SystemExit

def read_camera(camera, yoloNCS):
    global recorder_output

    while True:
        img = camera.get_frame_cv2_format()

        if img is not None:

            #process Ylol algorithm with NCS Graph
            results = yoloNCS.process_image(img)

            #print (results)
            #cv2.imshow('YOLO detection',img_cv)
            img = show_results(img, results, img.shape[1], img.shape[0], yoloNCS.colors, False)

            if recorder_output is not None:
                # write the flipped frame
                recorder_output.write(img)

            image_bytes = cv2.imencode('.jpg', img)[1].tostring()

            yield (b'--frame\r\n'
                   b'Content-Type: image/bmp\r\n\r\n' + image_bytes + b'\r\n\r\n')

        else:
            raise Exception('image is None')


def get_camera():
    if os.uname()[4].startswith("arm"):
        #Raaspberry Pi version
        print('Raspberry Pi platform detected')
        from streaming.camera_pi import VideoCameraPi
        camera = VideoCameraPi()
    else:
        # laptop internal webcam
        print('linux platform detected')
        from streaming.camera import VideoCamera
        camera = VideoCamera()
    return camera


def init(record=False):
    global yoloNCS, camera, recorder_output

    recorder_output = None

    if record:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        recorder_output = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    # load NCS graph
    yoloNCS = YoloNCS('graphs/graph_ncappzoo')

    # load the right camera
    camera = get_camera()

    if camera is None:
        raise Exception("camera can't be opened")


if __name__ == '__main__':

    try:

        #add runserver command to the manager
        manager.add_command('runserver', CustomServer())
        manager.add_command('runserver_and_recorder', CustomServerRecorder())

        # catch Ctrl+C signal
        signal.signal(signal.SIGINT, signal_handler)

        #run flask server
        manager.run()

    except Exception as e:
        print('Exit program with Exception {}'.format(e.args))
