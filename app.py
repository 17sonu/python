from flask import Flask, jsonify
from flask_cors import CORS
import os
import cv2

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error loading Haar Cascade!")
else:
    print("Haar Cascade loaded successfully!")

output_dir = "video_to_image"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

@app.route('/start_capture', methods=['GET'])
def start_capture():
    cap = cv2.VideoCapture(0)
    count = 1
    stable_face_center = None
    last_face_center = None
    movement_threshold = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return jsonify({"message": "Unable to read the frame"}), 500

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center = (x + w // 2, y + h // 2)

            if stable_face_center is None:
                stable_face_center = face_center

            if last_face_center is None:
                last_face_center = face_center

            movement_distance = ((face_center[0] - last_face_center[0]) ** 2 +
                                 (face_center[1] - last_face_center[1]) ** 2) ** 0.5

            if movement_distance > movement_threshold:
                capture = cv2.imwrite(f"{output_dir}/photo{count}_moved.jpeg", frame)
                if capture:
                    print(f"Movement detected! Frame saved at {output_dir}/photo{count}_moved.jpeg")
                count += 1
                last_face_center = face_center

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": f"Captured {count-1} images", "output_dir": output_dir}), 200

if __name__ == '__main__':
    app.run(debug=True)
