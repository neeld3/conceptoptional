from flask import Flask, Response, request, redirect, url_for
import cv2
import numpy as np
from picamera2 import Picamera2
import threading
from queue import Queue
from twilio.rest import Client

app = Flask(__name__)

detection_enabled = False
detected = False

TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
TARGET_PHONE_NUMBER = None  # Set initially to None

weight = "hope_best.weights"
cfg = "hope.cfg"
net = cv2.dnn.readNetFromDarknet(cfg, weight)
classes = ["cat"]

cam = Picamera2()
cam.start()

frame_queue = Queue(maxsize=10)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert(message):
    try:
        if TARGET_PHONE_NUMBER:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=TARGET_PHONE_NUMBER
            )
            print("Alert sent!")
        else:
            print("No target phone number provided.")
    except Exception as e:
        print(f"Failed to send alert: {e}")

def process_frames():
    global detected
    while True:
        frame = cam.capture_array()

        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(rgb_frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        try:
            unconnected_out_layers = net.getUnconnectedOutLayers()
            if isinstance(unconnected_out_layers[0], np.ndarray):
                output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
            else:
                output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
        except Exception as e:
            print(f"Error getting output layers: {e}")
            continue

        detections = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        object_detected = False

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    object_detected = True

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = (0, 255, 0)
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(rgb_frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if detection_enabled and object_detected and not detected:
                send_alert(f"Object detected: {label} with confidence {confidence}")
                detected = True  # Set flag to prevent multiple alerts

        if not object_detected:
            detected = False

        if detection_enabled:
            line_thickness = 5
            color = (0, 0, 255)  # Red color in BGR
            cv2.line(rgb_frame, (50, 50), (width - 50, height - 50), color, line_thickness)
            cv2.line(rgb_frame, (width - 50, 50), (50, height - 50), color, line_thickness)

        if not frame_queue.full():
            frame_queue.put(rgb_frame)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            ret, buffer = cv2.imencode('.jpg', frame)
            jpeg_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    global TARGET_PHONE_NUMBER
    if request.method == 'POST':
        phone_number = request.form.get('phone')
        if phone_number:
            TARGET_PHONE_NUMBER = phone_number
        return redirect(url_for('video'))
    return '''
        <html>
            <head>
                <title>Object Detection Setup</title>
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                        height: 100vh;
                        margin: 0;
                        background-color: #f4f4f4;
                        font-family: Arial, sans-serif;
                    }
                    h1 {
                        margin-bottom: 20px;
                    }
                    form {
                        margin-top: 20px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    input[type="text"] {
                        padding: 10px;
                        font-size: 16px;
                        border-radius: 5px;
                        border: 1px solid #ddd;
                        margin-bottom: 10px;
                        width: 100%;
                        max-width: 300px;
                    }
                    .button-container {
                        display: flex;
                        justify-content: center;
                        gap: 10px;
                    }
                    .button {
                        padding: 15px 30px;
                        font-size: 16px;
                        color: #fff;
                        background-color: #007bff;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s;
                    }
                    .button:hover {
                        background-color: #0056b3;
                    }
                </style>
            </head>
            <body>
                <h1>Enter Phone Number for Alerts</h1>
                <form method="post">
                    <input type="text" name="phone" placeholder="Enter phone number" required>
                    <div class="button-container">
                        <button type="submit" class="button">Receive Alerts</button>
                        <a href="/video"><button type="button" class="button">View Feed</button></a>
                    </div>
                </form>
            </body>
        </html>
    '''

@app.route('/video', methods=['GET', 'POST'])
def video():
    global detection_enabled
    if request.method == 'POST':
        if 'on' in request.form:
            detection_enabled = True
        elif 'off' in request.form:
            detection_enabled = False
        return redirect(url_for('video'))
    return '''
        <html>
            <head>
                <title>Object Detection</title>
                <style>
                    body {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                        height: 100vh;
                        margin: 0;
                        background-color: #f4f4f4;
                        font-family: Arial, sans-serif;
                    }
                    h1 {
                        margin-bottom: 20px;
                    }
                    .button-container {
                        margin-top: 20px;
                        display: flex;
                        gap: 10px;
                    }
                    .button {
                        padding: 15px 30px;
                        font-size: 16px;
                        color: #fff;
                        background-color: #007bff;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s;
                    }
                    .button:hover {
                        background-color: #0056b3;
                    }
                    .video-feed {
                        width: 640px;
                        height: 480px;
                        border: 2px solid #ddd;
                        border-radius: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Object Detection</h1>
                <div>
                    <img src="/video_feed" class="video-feed">
                </div>
                <div class="button-container">
                    <form method="post">
                        <button type="submit" name="on" class="button">ON</button>
                        <button type="submit" name="off" class="button">OFF</button>
                    </form>
                </div>
            </body>
        </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
