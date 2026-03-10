# DroneAI

Pierwszy szkic aplikacji do pracy z DJI Tello: lokalne API sterujące, live preview z przedniej kamery, detekcja twarzy i rozpoznawanie twarzy z zapisami embeddingów w SQLite.

## Wymagania

- Python 3.10+
- `uv`
- dron DJI Tello do testów połączenia
- połączenie z siecią Wi‑Fi drona Tello
- lokalny model embeddingów SFace `.onnx`
- lokalny model detekcji MediaPipe `.tflite`, jeśli używany jest backend `mediapipe.tasks`

Instalacja `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Uruchomienie

Najprostszy sposób:

```bash
./start.sh
```

Skrypt:

- sprawdza, czy `uv` jest dostępne
- tworzy lokalne `.venv`
- synchronizuje środowisko z `pyproject.toml`
- uruchamia aplikację przez `uv run python app.py`

## DJITelloPy

Projekt używa biblioteki `DJITelloPy` jako warstwy komunikacji z dronem Tello:

- pakiet runtime: `djitellopy`
- podstawowy handshake: `connect()`
- odczyt statusu: `get_battery()`
- pobranie obrazu z kamery przedniej przez `get_frame_read()`
- start streamu: `streamon()`

## Vision stack

Pipeline rozpoznawania twarzy jest podzielony na osobne warstwy:

- MediaPipe jako detektor twarzy
- OpenCV SFace jako model embeddingów
- SQLite jako magazyn klas i embeddingów
- OpenCV GUI jako live preview z bounding boxami i etykietą klasy

Domyślna baza:

`data/drone_ai.sqlite3`

Domyślny model embeddingów:

`models/face_recognition_sface_2021dec_int8.onnx`

Fallback dla detektora MediaPipe Tasks:

`models/blaze_face_short_range.tflite`

Modele trzeba pobrać lokalnie. Szczegóły są w [models/README.md](/home/lsriw/grzegorz-braun/DroneAI/models/README.md).

## Aplikacja

Główny entrypoint projektu to:

```bash
python3 app.py
```

Aplikacja robi dwie rzeczy:

- uruchamia lokalne HTTP API
- łączy się z Tello, analizuje obraz i pokazuje live preview z bounding boxami oraz przypisaną klasą

Przykładowe użycie:

```bash
uv run python app.py
uv run python app.py --host 0.0.0.0 --port 8000
uv run python app.py --skip-preview
```

Domyślne endpointy API:

- `GET /health`
- `GET /status`
- `GET /identities`
- `POST /connect`
- `POST /disconnect`
- `POST /faces/register`

Przykład rejestracji twarzy z aktualnie widocznej największej twarzy:

```bash
curl -X POST http://127.0.0.1:8000/faces/register \
  -H "Content-Type: application/json" \
  -d '{"name":"grzegorz"}'
```

## Aktualny zakres

Ten etap zawiera:

- pojedynczy entrypoint `app.py`
- moduł integracji z Tello oparty o `DJITelloPy`
- lokalne API do podstawowego sterowania
- live preview z przedniej kamery Tello
- MediaPipe detektor twarzy
- SFace embeddingi twarzy
- SQLite do przechowywania klas i embeddingów
- skrypt bootstrapujący `start.sh`

Śledzenie twarzy i utrzymywanie stałej odległości będą dodane w kolejnych krokach.
