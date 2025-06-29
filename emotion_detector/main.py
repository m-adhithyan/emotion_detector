import cv2
import mtcnn
from tensorflow.keras.models import load_model
import numpy as np
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'best_model_finetuned.h5')
    model_path = os.path.abspath(model_path)

    print("Model path:", model_path)
    print("Exists:", os.path.exists(model_path))

    model = load_model(model_path)
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    face_detector = mtcnn.MTCNN()
    capture = cv2.VideoCapture(0)
    log_on_left = True    

    # Predefined color map
    emotion_colors = {
        'Angry': (0, 0, 255),
        'Disgust': (0, 255, 0),
        'Fear': (128, 0, 128),
        'Happy': (0, 255, 255),
        'Neutral': (200, 200, 200),
        'Sad': (255, 0, 0),
        'Surprise': (255, 165, 0)
    }

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(rgb)

        best_probs = None

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            if w < 48 or h < 48:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=-1)
            face_input = face_img.reshape(1, 48, 48, 1)

            probs = model.predict(face_input, verbose=0)[0]
            emotion_i = int(np.argmax(probs))
            emotion = labels[emotion_i]
            conf = probs[emotion_i]

            cv2.putText(frame,
                        f'{emotion} ({conf*100:.1f}%)',
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            if best_probs is None or conf > max(best_probs):
                best_probs = probs

        # Draw histogram inside the loop
        if best_probs is not None:
            bar_start_x = 20 if log_on_left else frame.shape[1] - 250
            bar_start_y = 40
            bar_height = 25
            bar_width_max = 200
            bar_spacing = 8

            for i, prob in enumerate(best_probs):
                label = labels[i]
                bar_width = int(prob * bar_width_max)
                top_left = (bar_start_x, bar_start_y + i * (bar_height + bar_spacing))
                bottom_right = (bar_start_x + bar_width, top_left[1] + bar_height)

                color = emotion_colors.get(label, (100, 255, 100))
                overlay = frame.copy()
                cv2.rectangle(overlay, top_left, bottom_right, color, -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.putText(frame,
                            f'{label} ({prob * 100:.1f}%)',
                            (bar_start_x, top_left[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()