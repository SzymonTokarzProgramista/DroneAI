"""Packaged application entrypoint for DroneAI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from drone_ai.application import DroneApplication
from drone_ai.config import AppConfig

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DroneAI application with local Tello preview, MediaPipe detection, and face recognition API."
    )
    parser.add_argument("--host", default="127.0.0.1", help="API bind host.")
    parser.add_argument("--port", type=int, default=8000, help="API bind port.")
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Start the API and drone connection without opening the local preview window.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    application: DroneApplication | None = None

    try:
        application = DroneApplication(AppConfig.from_env(ROOT_DIR))
        application.start_api(args.host, args.port)
        print(f"DroneAI API listening on http://{args.host}:{args.port}")
        status = application.connect(enable_stream=True)
        print(
            f"Tello connected. Battery: {status.battery}%. Stream enabled: {status.stream_enabled}"
        )
        print("Face detection: MediaPipe. Face embeddings: OpenCV SFace. Storage: SQLite.")

        if args.skip_preview:
            print("Preview disabled. Press Ctrl+C to stop the app.")
            while True:
                time.sleep(1)

        return application.preview_loop()
    except KeyboardInterrupt:
        print("Stopping DroneAI application...")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error: unexpected application failure: {exc}")
        return 1
    finally:
        try:
            if application is not None:
                application.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
