import os
import re
import sys
import cv2
import onnx
import platform
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH  = os.path.join(BASE_PATH, 'model')

if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def get_image(path: str) -> np.ndarray:
    return cv2.cvtColor(src=cv2.imread(path, cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)


def show_images(
    image_1: np.ndarray,
    image_2: np.ndarray, 
    title_1: str="Original",
    title_2: str="Depth"
    ) -> None:

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gnuplot2")
    plt.axis("off")
    if title_1: plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.axis("off")
    if title_2: plt.title(title_2)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


class CFG(object):
    def __init__(self) -> None:
        self.ort_session = None
        self.size: int = 256
        self.mean: list = [0.5, 0.5, 0.5]
        self.std: list  = [0.5, 0.5, 0.5]
        self.path: str = os.path.join(MODEL_PATH, 'model.onnx')
        ort.set_default_logger_severity(3)
    
    def setup(self) -> None:
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape

        image = image / 255
        image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.mean[i]) / self.std[i]
        image = np.expand_dims(image, axis=0)
        input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
        result = self.ort_session.run(None, input)
        result = result[0].transpose(1, 2, 0)
        result = cv2.applyColorMap(src=cv2.convertScaleAbs(src=result, alpha=0.8), colormap=cv2.COLORMAP_JET)
        return cv2.resize(src=result, dsize=(w, h), interpolation=cv2.INTER_AREA)


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--file", "-f")
    args_3: tuple = ("--downscale", "-ds")
    args_4: tuple = ("--save", "-s")

    mode: str = "image"
    filename: str = "Test_1.jpg"
    downscale: float = None
    save: bool = False

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: filename = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: filename = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[0]) + 1])
    if args_3[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_3[1]) + 1])

    if args_4[0] in sys.argv or args_4[1] in sys.argv: save = True

    breaker()
    cfg = CFG()
    cfg.setup()

    if re.match(r"image", mode, re.IGNORECASE):
        image = get_image(os.path.join(INPUT_PATH, filename))
        result = cfg.infer(image)
        if save: cv2.imwrite(os.path.join(OUTPUT_PATH, filename[:-4] + " - Result.png"), cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB))
        else: show_images(image_1=image, image_2=result)
    
    elif re.match(r"video", mode, re.IGNORECASE):
        cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))

        while True:
            ret, frame = cap.read()
            if not ret: break
            if downscale:
                frame = cv2.resize(src=frame, dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), interpolation=cv2.INTER_AREA)
            result = cfg.infer(frame)
            frame = np.concatenate((frame, cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB)), axis=1)
            cv2.imshow("Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
        cap.release()
        cv2.destroyAllWindows()

    elif re.match(r"realtime", mode, re.IGNORECASE):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True:
            ret, frame = cap.read()
            if not ret: continue
            result = cfg.infer(cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB))            
            frame = np.concatenate((frame, cv2.cvtColor(src=result, code=cv2.COLOR_BGR2RGB)), axis=1)
            cv2.imshow("Feed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("--- Unknown Mode ---\n".upper())
    
    print("Program Run Complete. Terminating...")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
