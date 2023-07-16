from flask import Flask, render_template, Response, send_file, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
import cv2
import math
from ultralytics import YOLO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'trafficsign'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['OUTPUT_FOLDER'] = 'static/output'


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Run")



def generate_frames_web():
    model = YOLO("C:/Users/Adithya Roy/Desktop/project traffic signs/best.pt")
    ClassNames = ["no entry", "no overtaking", "uneven road", "dangerous dip", "right reverse bend",
                  "staggered junction road", "no heavy vehicles", "give way", "blind school", "noparking", "stop",
                  "warning sign", "50 Speed limit", "no parking", "pedistrain", "narrow bridge", "no horn", "u turn",
                  "take right"]

    cap = cv2.VideoCapture(0)  # Access the default webcam (0) or you can specify the webcam index if you have multiple cameras
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter("trafficsign.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = ClassNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        out.write(frame)
        cv2.imshow("Image",frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()
    cv2.destroyAllWindows()


@app.route('/webcam')
def webcam():
    return Response(generate_frames_web(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)
