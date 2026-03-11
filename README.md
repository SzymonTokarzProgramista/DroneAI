# DroneAI

Initial prototype of an application for working with a **DJI Tello** drone: a local control API, live preview from the front camera, face detection, and face recognition with embeddings stored in SQLite.

---

# Requirements

- Python **3.10+**
- `uv`
- **DJI Tello drone** (for connection testing)
- Local **SFace embedding model** (`.onnx`)
- Local **MediaPipe face detector model** (`.tflite`) if using the `mediapipe.tasks` backend
- System `tkinter` package for the desktop GUI

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

# Running the Application

The simplest way:

```bash
./start.sh
```

---

# DJITelloPy

The project uses the **DJITelloPy** library as the communication layer for the Tello drone.

---

# Vision Stack

The face recognition pipeline is divided into separate layers:

- **MediaPipe** – face detector
- **OpenCV SFace** – face embedding model
- **SQLite** – storage for identities and embeddings
- **OpenCV GUI** – live preview with bounding boxes and identity labels

Default database:

```
data/drone_ai.sqlite3
```

Default embedding model:

```
models/face_recognition_sface_2021dec_int8.onnx
```

Fallback detector for MediaPipe Tasks:

```
models/blaze_face_short_range.tflite
```

Models must be downloaded locally. See details in:

```
models/README.md
```

---

# Application

The main entry point of the project:

```bash
python3 app.py
```

The application performs the following tasks:

- starts a **local HTTP API**
- connects to the **Tello drone**
- processes frames from the camera
- displays a **live preview with bounding boxes and recognized identities**
- provides a **GUI for registering the largest visible face as an embedding**


- **face tracking**
- maintaining a **constant distance to the tracked face**
