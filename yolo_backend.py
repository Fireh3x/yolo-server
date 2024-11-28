import hashlib
from ultralytics import YOLO
import cv2
import matplotlib
import torch
from skimage import io
import pandas as pd
from ultralytics.nn.modules import DFL

# Define YOLO model ops
# yolo_model = torch.hub.load(
#     "ultralytics/yolov5", "yolov5s"
# )  # or yolov5n - yolov5x6, custom
yolo_model = YOLO("/my_vol/yolo5small.pt")
#yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/my_vol/yolo5small.pt',force_reload=True)
cmap = matplotlib.pyplot.get_cmap("jet")


def hash_to_range(number, N):
    # Convert the number to a string and then encode it to bytes
    byte_representation = str(number).encode("utf-8")

    # Use SHA-256 hash function
    hashed = hashlib.sha256(byte_representation)

    # Convert the hash to an integer
    hash_integer = int(hashed.hexdigest(), 16)

    # Map the hash to the range [1, N]
    return 1 + (hash_integer % N)

names = {'0': 'anger', "1": 'fear', "2": 'happy', "3": 'neutral', "4": 'sad'}
def predict(img_path):
    results = yolo_model(img_path)


    return results[0]

def predict_and_draw(img, out_img_path):
    img = io.imread(img)
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result = predict(img_cv2)
    result.save(out_img_path)
    print(out_img_path)
    
    print(result.tojson())
    result = pd.read_json(result.tojson())

    result["color"] = result.apply(
        lambda x: tuple(
            c * 255
            for c in cmap(
                hash_to_range(x["class"], 5) / 5
            )
        )[:3],
        axis=1,
    )

    # for _, b in result.iterrows():
    #     el = b
    #     p1, p2 = (el["xmin"], el["ymin"]), (el["xmax"], el["ymax"])
    #     p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
    #     cv2.rectangle(img_cv2, p1, p2, el["color"], 2)
    #img = io.imread(out_img_path)
    #img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #cv2.imwrite(str(out_img_path), img_cv2)
    #cv2.imwrite(str(out_img_path), img_cv2)



    result_dict = result.to_dict(orient="index")
    return {
        "boxes": [v for _, v in result_dict.items()],
    }
