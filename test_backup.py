import time
import cv2
from threading import Thread, Event
from djitellopy import Tello

def wait_for_valid_frame(frame_read, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        frame = frame_read.frame
        if frame is not None and hasattr(frame, "shape") and frame.size != 0:
            h, w = frame.shape[:2]
            if h > 0 and w > 0:
                return frame
        time.sleep(0.01)
    raise RuntimeError("Nie udało się pobrać poprawnej klatki z Tello (timeout).")

def fix_colors(frame):
    # Tello bywa w RGB -> my chcemy BGR do OpenCV
    try:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception:
        return frame

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

first = wait_for_valid_frame(frame_read, timeout=8.0)
h, w = first.shape[:2]

OUT_SIZE = (w, h)
FPS = 30

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("video.avi", fourcc, FPS, OUT_SIZE)
if not video.isOpened():
    tello.streamoff()
    raise RuntimeError("VideoWriter nie otworzył pliku. Zmień kodek na MJPG/XVID albo doinstaluj FFmpeg.")

stop_event = Event()

def video_recorder():
    while not stop_event.is_set():
        frame = frame_read.frame
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            time.sleep(0.005)
            continue

        frame = fix_colors(frame)

        fh, fw = frame.shape[:2]
        if (fw, fh) != OUT_SIZE:
            frame = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_LINEAR)

        if frame.dtype != "uint8":
            frame = frame.astype("uint8")

        video.write(frame)
        time.sleep(1.0 / FPS)

recorder = Thread(target=video_recorder, daemon=True)
recorder.start()

try:
    # Start lotu w osobnym wątku, żeby live działał cały czas
    def flight():
        tello.takeoff()
        tello.move_up(200)
        tello.rotate_counter_clockwise(360)
        tello.land()

    flight_thread = Thread(target=flight, daemon=True)
    flight_thread.start()

    # LIVESTREAM w głównym wątku (OpenCV tego wymaga)
    while not stop_event.is_set():
        frame = frame_read.frame
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            time.sleep(0.01)
            continue

        frame = fix_colors(frame)

        # (opcjonalnie) dopasuj rozmiar okna bez zmiany nagrania
        preview = frame
        cv2.imshow("Tello Live", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # awaryjnie kończymy wszystko
            stop_event.set()
            break

        # jeśli lot się skończył, wyjdź z live po chwili
        if not flight_thread.is_alive():
            time.sleep(0.5)
            stop_event.set()
            break

finally:
    stop_event.set()
    recorder.join(timeout=2.0)
    video.release()

    cv2.destroyAllWindows()

    # spróbuj wyłączyć stream
    for _ in range(3):
        try:
            tello.streamoff()
            break
        except Exception:
            time.sleep(0.5)

    tello.end()
