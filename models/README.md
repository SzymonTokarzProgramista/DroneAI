# Models

Projekt oczekuje lokalnych modeli dla dwóch etapów pipeline'u:

- embedding twarzy OpenCV SFace
- detekcja twarzy MediaPipe Tasks, jeśli zainstalowany `mediapipe` nie wystawia `solutions`

## SFace embedder

Domyślna ścieżka:

`models/face_recognition_sface_2021dec_int8.onnx`

Model referencyjny:

https://huggingface.co/opencv/face_recognition_sface

Override:

`DRONE_AI_EMBEDDER_MODEL=/sciezka/do/modelu.onnx`

## MediaPipe detector fallback

Domyślna ścieżka:

`models/blaze_face_short_range.tflite`

Jeżeli Twoja wersja `mediapipe` nie ma `solutions`, aplikacja spróbuje użyć `mediapipe.tasks`
z tym lokalnym modelem detekcji.

Override:

`DRONE_AI_DETECTOR_MODEL=/sciezka/do/modelu.tflite`
