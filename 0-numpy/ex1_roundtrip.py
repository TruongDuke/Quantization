"""Bài 1 – Lượng tử hoá rồi giải lượng tử, kiểm chứng tính chất của §2.2.

Chạy:  python 0-numpy/ex1_roundtrip.py
"""
import numpy as np
from qlib import calc_qparams, quantize, dequantize, fake_quant, mse

np.set_printoptions(precision=4, suppress=True)

# --- 1. Ví dụ tính tay: khoảng [-1, 2], 8-bit, bất đối xứng ---
s, z = calc_qparams(-1.0, 2.0, b=8, symmetric=False)
print(f"[1] khoảng [-1, 2], 8-bit asym  ->  s = {s:.5f}, z = {z:.0f}")
for x in [0.123, 0.0, 3.0]:
    xi = quantize(x, s, z, 8, False)
    xh = dequantize(xi, s, z)
    print(f"    x = {x:6.3f}  ->  x_int = {xi:3.0f}  ->  x_hat = {xh:.4f}   lệch {abs(x - xh):.4f}")
print(f"    q_min = -s*z = {-s * z:.3f},  q_max = s*(255-z) = {s * (255 - z):.3f}")

# --- 2. Trong khoảng: sai số làm tròn luôn <= s/2 ---
x = np.random.uniform(-1, 2, 100_000)
err = np.abs(x - fake_quant(x, s, z, 8, False))
print(f"\n[2] 100k số trong [-1, 2]: max |lỗi| = {err.max():.5f}  <=  s/2 = {s / 2:.5f}  ->  {err.max() <= s / 2 + 1e-12}")

# --- 3. Số 0 được biểu diễn chính xác (vì z là số nguyên) ---
print(f"\n[3] q(0) = {fake_quant(0.0, s, z, 8, False)}  (phải đúng bằng 0)")

# --- 4. Symmetric vs asymmetric trên 3 kiểu phân phối (Hình 3) ---
print("\n[4] 8-bit: MSE và số vạch thực sự được dùng (trên 256)")
data = {
    "đối xứng N(0,1)":       np.random.randn(100_000),
    "sau ReLU (chỉ dương)":  np.maximum(np.random.randn(100_000), 0),
    "lệch [-1, 3]":          np.random.uniform(-1, 3, 100_000),
}
print(f"    {'dữ liệu':24s} {'symmetric':>22s} {'asymmetric':>22s}")
for name, x in data.items():
    row = []
    for sym in (True, False):
        s_, z_ = calc_qparams(x.min(), x.max(), 8, sym)
        xi = quantize(x, s_, z_, 8, sym)
        row.append(f"{mse(x, dequantize(xi, s_, z_)):.2e} / {len(np.unique(xi)):3d} vạch")
    print(f"    {name:24s} {row[0]:>22s} {row[1]:>22s}")
print("    -> dữ liệu càng lệch, symmetric càng phí vạch (sau ReLU phí gần một nửa).")
