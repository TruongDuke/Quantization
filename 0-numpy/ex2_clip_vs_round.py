"""Bài 2 – Trade-off clipping vs rounding (§2.2, §3.1).

Quét ngưỡng cắt c (lưới phủ [-c, c]), tách riêng 2 loại sai số:
  - rounding error: từ các giá trị nằm trong khoảng
  - clipping error: từ các giá trị nằm ngoài khoảng (bị ép về ±c)
Chạy:  python 0-numpy/ex2_clip_vs_round.py   -> lưu ex2_clip_vs_round.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qlib import calc_qparams, fake_quant

rng = np.random.default_rng(0)
x = rng.standard_normal(200_000)
x[:20] = rng.choice([-1, 1], 20) * rng.uniform(8, 12, 20)   # vài outlier, giống weight thật

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, b in zip(axes, (4, 8)):
    cs = np.linspace(0.5, np.abs(x).max(), 400)
    clip_e, round_e = [], []
    for c in cs:
        s, z = calc_qparams(-c, c, b, symmetric=True)
        e2 = (x - fake_quant(x, s, z, b, True)) ** 2
        inside = np.abs(x) <= c
        round_e.append(e2[inside].sum() / x.size)
        clip_e.append(e2[~inside].sum() / x.size)
    clip_e, round_e = np.array(clip_e), np.array(round_e)
    total = clip_e + round_e
    best = cs[total.argmin()]

    ax.plot(cs, round_e, label="rounding error")
    ax.plot(cs, clip_e, label="clipping error")
    ax.plot(cs, total, "k", lw=2, label="tổng MSE")
    ax.axvline(best, ls="--", c="g", label=f"tối ưu MSE: c = {best:.2f}")
    ax.axvline(cs[-1], ls=":", c="r", label=f"min-max: c = {cs[-1]:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("ngưỡng cắt c"); ax.set_ylabel("MSE")
    ax.set_title(f"{b}-bit: min-max tệ hơn tối ưu {total[-1] / total.min():.1f}x")
    ax.legend(fontsize=8)
    print(f"{b}-bit: c tối ưu = {best:.2f} (min-max = {cs[-1]:.2f}),  "
          f"MSE tối ưu = {total.min():.2e},  MSE min-max = {total[-1]:.2e}")

plt.suptitle("N(0,1) + 20 outlier: clipping vs rounding")
plt.tight_layout()
out = __file__.replace(".py", ".png")
plt.savefig(out, dpi=120)
print("đã lưu", out)
