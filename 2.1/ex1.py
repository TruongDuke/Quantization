import numpy as np

w = np.array([0.6, -1.2, 0.35]);  x = np.array([0.827, -0.30, 1.46]);  b = 0.05
s_w, s_x, s_y = 0.01, 0.02, 0.02

# FP32
print("FP32 :", w @ x + b)

# INT8
w_int = np.round(w / s_w).astype(np.int32)
x_int = np.round(x / s_x).astype(np.int32)
b_int = np.round(b / (s_w * s_x)).astype(np.int32)
acc   = w_int @ x_int + b_int                 # chỉ số nguyên
print("w_int:", w_int, " x_int:", x_int, " acc:", acc)
print("INT  :", acc * s_w * s_x)              # gắn lại scale 1 lần
print("y_int8 (requantize):", np.round(acc * s_w * s_x / s_y))
