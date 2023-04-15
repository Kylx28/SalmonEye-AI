from flask import Flask, render_template, Response
from datetime import datetime
from ultralytics import YOLO
import cv2

model = YOLO('static/detect_salmon.pt')
app = Flask(__name__)

camera = cv2.VideoCapture('static/salmon_run.mp4')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

salmon_count = 0
jump_dict = {}

def gen_frames():  # generate frame by frame from camera
    threshold = 25
    counter = 0
    global salmon_count
    global jump_dict
    while True:
        # Capture frame-by-frame
        success, frame = camera.read() # read the camera frame
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0) # repeat video
        else:
            results = model.predict(source=frame, classes=15, verbose=False)
            annotated_frame = results[0].plot()

            salmon_in_frame = len(results[0].boxes)
            counter += salmon_in_frame
            if counter >= threshold:
                now = datetime.now()
                date_time = now.strftime("%m/%d, %H:%M:%S")

                # print("Salmon Detected:", now)
                salmon_count += 1
                if salmon_count in jump_dict:
                    pass
                else:   
                    jump_dict[salmon_count] = date_time
                #print("Jump: ", jump_dict[salmon_count])
                counter = 0

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jump_message')
def jump_message():
    global salmon_count
    text = "<br>Total Salmon Jumps: " + str(salmon_count)
    #text = "<br>" + str(salmon_count)
    return render_template('jump_message.html', message = text)

@app.route('/jump_time')
def time_message():
    global salmon_count
    global jump_dict
    text2 = "<br>Last Jump: " + jump_dict[salmon_count]
    return render_template('jump_message.html', message2 = text2)
    

@app.route('/')
def index():
    #"""Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    camera.release()
