from modal import (
    App,
    Image,
    Mount,
    wsgi_app,
    Volume
)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "flask==3.0.3",
        "matplotlib==3.9.2",
        "scikit-image==0.24.0",
        "opencv-python==4.10.0.84",
        "torch==2.5.0",
        "pandas==2.2.3",
        "requests==2.32.3",
        "ultralytics==8.2.48"
    )
)

app = App("yolo-server")

@app.function(image=image, mounts=[Mount.from_local_python_packages("yolo_backend")], volumes={"/my_vol": Volume.from_name("my-test-volume")})
@wsgi_app()
def flask_app():
    import time
    from pathlib import Path
    from io import BytesIO
    import base64
    from pprint import pprint

    import numpy as np
    from flask import Flask, jsonify, request, send_from_directory

    import yolo_backend


    import os  
    print(os.path.exists('/content/yolo-server/yolo5small.pt'))    # Create necessary directories
    DATA_PATH = Path("data")
    to_create = [str(DATA_PATH), "data/to_prc", "data/gen_img"]
    for fpath in to_create:
        if not (p := Path(fpath)).exists():
            p.mkdir()

    # Flask app
    app = Flask(__name__)

    @app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict_img():
        file = request.data
        cur_id = (
            str(int(time.time())) + "_" + str(np.random.randint(0, int(1e10))).zfill(10)
        )

        image_data = BytesIO(base64.b64decode(file))
        try:
          result = yolo_backend.predict_and_draw(
              image_data,
              DATA_PATH / "gen_img" / f"{cur_id}.jpg",
          ) | {"gen_img": f"gen_img/{cur_id}.jpg"}

          pprint(result)
          return jsonify(result)
        except ValueError:
          result = {"gen_img": f"gen_img/{cur_id}.jpg"}
          return jsonify(result)

    @app.route("/gen_img/<path:filepath>")
    def gen_img(filepath):
        return send_from_directory(DATA_PATH / "gen_img", filepath)

    return app

# if __name__ == "__main__":
#     app.run(debug=True, port=2222)