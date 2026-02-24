from typing import TypedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from multiprocessing import Process, Manager
import time
import cv2
import numpy as np

class SharedFrames(TypedDict):
    """ Shared data for MJPEGHandler. """

    jpeg_bytes: bytes
    """ Pre-encoded JPG bytes. """

    timestamp:  int
    """ Timestamp when the frame was updated. """

def _start_server(
    shared_dict: SharedFrames,
    path:        str,
    host:        str           = "0.0.0.0",
    port:        int           = 8000
    ):
    """
    Starts a http server with MJPEGHandler in a separate process.
    """

    # Use ThreadingMixIn so every client gets its own thread
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != path:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_timestamp = 0

            while True:
                try:
                    # check if a new frame is available
                    current_timestamp = shared_dict.get("timestamp", 0)

                    if current_timestamp != last_timestamp:
                        jpeg_data = shared_dict.get("jpeg_bytes")

                        if jpeg_data is not None:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg_data)}\r\n\r\n".encode())
                            self.wfile.write(jpeg_data)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            last_timestamp = current_timestamp
                    else:
                        # Prevent high CPU usage during idle wait
                        time.sleep(0.01)

                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected
                    break
                except Exception as e:
                    print(f"Stream error: {e}")
                    break

    # Start the threaded server
    server = ThreadingHTTPServer((host, port), MJPEGHandler)
    print(f"MJPEG Stream running at http://{host}:{port}{path}")
    server.serve_forever()

class MJPEGServer:
    def __init__(
        self,
        path: str,
        host: str  = "0.0.0.0",
        port: int  = 8000
        ):
        self.manager = Manager()
        self.shared_dict = self.manager.dict()

        # Initialize dictionary to prevent KeyErrors before first frame
        self.shared_dict["jpeg_bytes"] = None
        self.shared_dict["timestamp"]  = 0

        self.process = Process(target=_start_server, args=(self.shared_dict, path, host, port), daemon=True)
        self.process.start()

    def update(self, image: np.ndarray):
        """
        Update the image used in the stream.
        Encodes the image to JPEG ONCE here to save CPU in client threads.

        Args:
            image: array (H, W, channels) in RGB format.
        """
        # Convert RGB to BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 4. Encode ONCE here.
        _, buf = cv2.imencode(".jpg", bgr_image)

        # Store raw bytes
        self.shared_dict["jpeg_bytes"] = buf.tobytes()
        self.shared_dict["timestamp"]  = time.perf_counter_ns()

    def close(self):
        self.process.terminate()
        self.process.join()