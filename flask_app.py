import os
from flask import Flask, send_from_directory, abort

# === Flask Setup ===
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WEBGL_FOLDER = os.path.join(BASE_DIR, "volumetric_build")

# === Serve index.html ===
@app.route('/')
def index():
    index_path = os.path.join(WEBGL_FOLDER, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(WEBGL_FOLDER, 'index.html')
    else:
        return abort(404, description="index.html not found in volumetric_build")

# === Serve supporting WebGL files ===
@app.route('/<path:path>')
def serve_file(path):
    file_path = os.path.join(WEBGL_FOLDER, path)
    if os.path.exists(file_path):
        return send_from_directory(WEBGL_FOLDER, path)
    else:
        return abort(404, description=f"{path} not found in volumetric_build")

# === Run Server ===
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
