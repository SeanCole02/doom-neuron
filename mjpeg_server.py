from typing import TypedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process, Manager
import time
import cv2
import numpy as np

class SharedFrames(TypedDict):
    """ Shared image data for MJPEGHandler. """

    frame:     np.ndarray
    """ Image frames as an array, this will be converted to JPG format with OpenCV. """

    timestamp: int
    """ Timestamp which frames was updated. """

def _start_server(
    shared_dict: SharedFrames,
    path:        str,
    host:        str           = "0.0.0.0",
    port:        int           = 8000
    ):
    """
    Starts a http server with MJPEGHandler, intended to be used in a separate process.
    Image content is updated via the shared_dict.

    Args:
        shared_dict: SharedFrames dict, which provides "frame" and "timestamp".
        path:        host path
        host:        host ip. Defaults to "0.0.0.0".
        port:        host port. Defaults to 8000.
    """
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != path:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            self._last_timestamp = 0
            while True:
                frame         = shared_dict["frame"]
                new_timestamp = shared_dict["timestamp"]
                if frame is not None and self._last_timestamp != new_timestamp:
                    """ Update the image stream when the content has changed. """
                    _, buf = cv2.imencode(".jpg", frame)
                    jpg    = buf.tobytes()

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    self._last_timestamp = new_timestamp

    HTTPServer((host, port), MJPEGHandler).serve_forever()

class MJPEGServer:
    def __init__(
        self,
        path: str,
        host: str  = "0.0.0.0",
        port: int  = 8000
        ):
        self.manager     = Manager()
        self.shared_dict = self.manager.dict()
        self.process     = Process(target=_start_server, args=(self.shared_dict, path, host, port), daemon=True)
        self.process.start()

    def update(self, image: np.ndarray):
        """
        Update the image used in the stream.

        Args:
            image: array that will be encoded with cv2.imencode(). Shape is (H, W, channels), where channels is RGB.
        """
        self.shared_dict["frame"]     = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR) # OpenCV uses BGR
        self.shared_dict["timestamp"] = time.perf_counter_ns()

    def close(self):
        self.process.terminate()
