import cv2
from flask import Flask, render_template, Response

# Initialize HOG descriptor + SVM for people detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Flask app setup
app = Flask(__name__)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Detect people in the frame
        boxes, _ = hog.detectMultiScale(frame_resized, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw rectangles around detected people
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame as JPEG and yield it as a byte string
        ret, buffer = cv2.imencode('.jpg', frame_resized)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)