"""Bài 4 – Per-tensor vs per-channel vs per-group (§2.2.3, §2.4.2).

Tạo ma trận weight W (64 x 512) có:
  - các kênh (hàng) chênh nhau về độ lớn ~100 lần  (giống Hình 5, MobileNetV2)
  - vài outlier lẻ trong một số hàng                (giống LLM)
rồi so sai số của W_hat và của output W_hat @ x.
Chạy:  python 0-numpy/ex4_granularity.py   -> lưu ex4_granularity.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlib import quant_tensor, sqnr_db

rng = np.random.default_rng(0)
C_out, C_in = 64, 512
row_scale = 10 ** rng.uniform(-2, 0, (C_out, 1))           # mỗi kênh một độ lớn: 0.01 .. 1
W = rng.standard_normal((C_out, C_in)) * row_scale
for r in rng.choice(C_out, 8, replace=False):               # 8 hàng có 1 outlier
    W[r, rng.integers(C_in)] = 8 * row_scale[r, 0] * rng.choice([-1, 1])
x = rng.standard_normal((C_in, 256))                        # 256 input để đo output

configs = [("tensor", None), ("channel", None), ("group", 128), ("group", 32)]
print(f"{'granularity':16s} {'#scale':>7s} | {'W SQNR 8b':>9s} {'4b':>7s} | {'Wx SQNR 8b':>10s} {'4b':>7s}")
for g, gs in configs:
    out = []
    for b in (8, 4):
        W_hat, n_scale = quant_tensor(W, b, symmetric=True, granularity=g, group_size=gs or 128)
        out += [sqnr_db(W, W_hat), sqnr_db(W @ x, W_hat @ x)]
    label = g if gs is None else f"group({gs})"
    print(f"{label:16s} {n_scale:7d} | {out[0]:8.1f}dB {out[2]:6.1f}dB | {out[1]:9.1f}dB {out[3]:6.1f}dB")

# Sai số theo từng kênh ở 4-bit: per-tensor phá hỏng kênh nhỏ
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.boxplot(W.T, showfliers=False, whis=(0, 100))
a1.set_title("Khoảng weight theo kênh (giống Hình 5)"); a1.set_xlabel("output channel"); a1.set_xticks([])
order = np.argsort(row_scale[:, 0])
for g, gs in configs:
    W_hat, _ = quant_tensor(W, 4, True, g, gs or 128)
    per_ch = [sqnr_db(W[i], W_hat[i]) for i in order]
    a2.plot(per_ch, label=g if gs is None else f"group({gs})")
a2.set_title("4-bit: SQNR từng kênh (xếp từ kênh nhỏ -> lớn)")
a2.set_xlabel("kênh (tăng dần độ lớn)"); a2.set_ylabel("SQNR (dB)"); a2.legend()
plt.tight_layout()
out = __file__.replace(".py", ".png")
plt.savefig(out, dpi=120)
print("\nđã lưu", out)
