import sys
import cv2
import tensorflow as tf

sys.path.insert(0, "mlsd-official")
from utils import pred_lines

model_path = "mlsd-official/tflite_models/M-LSD_512_tiny_fp32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input:", input_details[0]["shape"], "| Outputs:", [o["shape"] for o in output_details])

img = cv2.cvtColor(cv2.imread("mlsd-pytorch/data/frame_1.jpg"), cv2.COLOR_BGR2RGB)
lines = pred_lines(img, interpreter, input_details, output_details,
                   input_shape=[512, 512], score_thr=0.1, dist_thr=20.0)
print("So doan thang:", len(lines))

out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
for x1, y1, x2, y2 in lines:
    cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 200), 2)
cv2.imwrite("official_out.jpg", out)
