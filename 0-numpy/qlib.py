"""Thư viện lượng tử hoá tối giản – cài đúng công thức §2.2 và §3.1 của paper.

Dùng lại cho các chặng sau. Mọi hàm đều vectorized: s, z có thể là số hoặc
mảng (mỗi hàng một bộ s, z) để làm per-channel / per-group.
"""
import numpy as np


# ---------- §2.2: lưới số nguyên, s, z ----------

def int_range(b, symmetric):
    """Giới hạn lưới số nguyên.
    symmetric (signed, eq 8c): [-2^(b-1), 2^(b-1)-1]
    asymmetric (unsigned, eq 4): [0, 2^b - 1]
    """
    if symmetric:
        return -(2 ** (b - 1)), 2 ** (b - 1) - 1
    return 0, 2 ** b - 1


def calc_qparams(xmin, xmax, b, symmetric):
    """Từ khoảng [xmin, xmax] muốn phủ -> (s, z)."""
    xmin = np.minimum(xmin, 0.0)  # khoảng phải chứa 0 để 0 được biểu diễn chính xác
    xmax = np.maximum(xmax, 0.0)
    n, p = int_range(b, symmetric)
    if symmetric:
        s = np.maximum(-xmin, xmax) / p
        z = np.zeros_like(s)
    else:
        s = (xmax - xmin) / (p - n)
        s = np.maximum(s, 1e-12)
        z = np.round(-xmin / s)          # z là số nguyên: vạch ứng với số 0 thật
    return np.maximum(s, 1e-12), z


def quantize(x, s, z, b, symmetric):
    """eq (4): x_int = clamp(round(x/s) + z; n, p)"""
    n, p = int_range(b, symmetric)
    return np.clip(np.round(x / s) + z, n, p)


def dequantize(x_int, s, z):
    """eq (6): x_hat = s * (x_int - z)"""
    return s * (x_int - z)


def fake_quant(x, s, z, b, symmetric):
    """eq (7): q(x) = dequantize(quantize(x)). Vào float, ra float nằm trên lưới."""
    return dequantize(quantize(x, s, z, b, symmetric), s, z)


# ---------- §3.1: chọn khoảng (range setting) ----------
# x có dạng (R, N): mỗi hàng được lượng tử hoá bằng một bộ (s, z) riêng.

def range_minmax(x, b, symmetric):
    """eq (14)-(15): phủ toàn bộ min..max -> không clipping, nhạy với outlier."""
    return calc_qparams(x.min(1, keepdims=True), x.max(1, keepdims=True), b, symmetric)


def range_mse(x, b, symmetric, n_grid=100):
    """eq (16): thu nhỏ khoảng min-max theo tỉ lệ alpha, chọn alpha có MSE nhỏ nhất (grid search)."""
    xmin, xmax = x.min(1, keepdims=True), x.max(1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf)
    best_s, best_z = None, None
    for alpha in np.linspace(1.0, 0.05, n_grid):
        s, z = calc_qparams(alpha * xmin, alpha * xmax, b, symmetric)
        err = ((x - fake_quant(x, s, z, b, symmetric)) ** 2).mean(1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_s = s if best_s is None else np.where(better, s, best_s)
        best_z = z if best_z is None else np.where(better, z, best_z)
    return best_s, best_z


RANGE_METHODS = {"minmax": range_minmax, "mse": range_mse}


# ---------- §2.2.3: độ mịn (granularity) ----------

def quant_tensor(W, b, symmetric=True, granularity="tensor", group_size=128, method="minmax"):
    """Fake-quant một ma trận W (C_out, C_in).
    granularity: 'tensor'  -> 1 bộ (s, z) cho cả W
                 'channel' -> 1 bộ cho mỗi hàng (output channel)
                 'group'   -> 1 bộ cho mỗi group_size phần tử liên tiếp trong hàng
    Trả về (W_hat, số lượng scale đã dùng).
    """
    if granularity == "tensor":
        rows = W.reshape(1, -1)
    elif granularity == "channel":
        rows = W.reshape(W.shape[0], -1)
    elif granularity == "group":
        assert W.shape[1] % group_size == 0, "C_in phải chia hết cho group_size"
        rows = W.reshape(-1, group_size)
    else:
        raise ValueError(granularity)
    s, z = RANGE_METHODS[method](rows, b, symmetric)
    return fake_quant(rows, s, z, b, symmetric).reshape(W.shape), s.size


# ---------- đo sai số ----------

def mse(x, x_hat):
    return float(((x - x_hat) ** 2).mean())


def sqnr_db(x, x_hat):
    """Signal-to-quantization-noise ratio (dB). Cao hơn = tốt hơn; mỗi bit thêm ≈ +6 dB."""
    return float(10 * np.log10((x ** 2).mean() / max(mse(x, x_hat), 1e-30)))
