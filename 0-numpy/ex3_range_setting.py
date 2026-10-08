"""Bài 3 – Min-max vs MSE range setting ở 8/6/4 bit (tái hiện ý của Bảng 1, §3.1).

Đo bằng SQNR (dB): cao hơn = tốt hơn, mỗi bit thêm ≈ +6 dB.
Chạy:  python 0-numpy/ex3_range_setting.py
"""
import numpy as np
from qlib import range_minmax, range_mse, fake_quant, sqnr_db

rng = np.random.default_rng(0)
N = 100_000
gauss_outlier = rng.standard_normal(N)
gauss_outlier[:10] *= 15
data = {
    "Gaussian":            rng.standard_normal(N),
    "Laplace (đuôi dày)":  rng.laplace(size=N),       # weight thật thường giống thế này
    "Gaussian + outlier":  gauss_outlier,
}

print(f"{'dữ liệu':22s} {'bit':>4s} {'min-max':>9s} {'MSE':>9s} {'chênh':>8s}")
for name, x in data.items():
    X = x.reshape(1, -1)
    for b in (8, 6, 4):
        res = []
        for fn in (range_minmax, range_mse):
            s, z = fn(X, b, symmetric=True)
            res.append(sqnr_db(X, fake_quant(X, s, z, b, True)))
        print(f"{name:22s} {b:4d} {res[0]:8.1f}dB {res[1]:8.1f}dB {res[1] - res[0]:+7.1f}")
    print()
print("-> bit cao: hai cách gần như ngang nhau. Bit thấp / có outlier: MSE thắng rõ (giống Bảng 1).")
