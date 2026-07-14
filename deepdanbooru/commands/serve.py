import os
import json
import cgi
import tensorflow as tf
from functools import partial
from six import BytesIO
from base64 import b64decode
import deepdanbooru as dd
from http.server import HTTPServer, BaseHTTPRequestHandler
from .evaluate import evaluate_image

class WebRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, model, tags, verbose, *args, **kwargs):
        self.model = model
        self.tags = tags
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write("Requests should be POSTed to this endpoint.".encode("utf-8"))

    def do_POST(self):
        self._set_headers()

        form = cgi.FieldStorage(
                 fp=self.rfile,
                 headers=self.headers,
                 environ={"REQUEST_METHOD": "POST",
                          "CONTENT_TYPE": self.headers['Content-Type']})

        b64 = form.getvalue('image')
        img_data = b64.split(',')[1]
        image = BytesIO( b64decode(img_data) )
        threshold = float(form.getvalue('threshold'))

        tags_dict = {}
        for tag, score in evaluate_image(image, self.model, self.tags, threshold):
            if self.verbose:
                print(f"({score:05.3f}) {tag}")
            tags_dict[tag] = float("{:.3f}".format(score))

        self.wfile.write(json.dumps(tags_dict).encode("utf-8"))

def serve(
    project_path: str, model_path: str, tags_path: str, host: str, port: int, 
    allow_gpu: bool, verbose: bool
):
    if not allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not model_path and not project_path:
        raise Exception("You must provide project path or model path.")
    if not host or not port:
        raise Exception("You must provide a host and port.")

    if model_path:
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = tf.keras.models.load_model(model_path)
    else:
        if verbose:
            print(f"Loading model from project {project_path} ...")
        model = dd.project.load_model_from_project(project_path)

    if tags_path:
        if verbose:
            print(f"Loading tags from {tags_path} ...")
        tags = dd.data.load_tags(tags_path)
    else:
        if verbose:
            print(f"Loading tags from project {project_path} ...")
        tags = dd.project.load_tags_from_project(project_path)

    handler = partial(WebRequestHandler, model, tags, verbose)
    server = HTTPServer((host, port), handler)

    print("Hoting on http://{}:{}".format(host, port))
    server.serve_forever()
