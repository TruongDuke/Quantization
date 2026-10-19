"""QAT cho DistilBERT (SST-2) -> TFLite INT8 (chỉ quantize 38 lớp Linear / FULLY_CONNECTED).

1. FP32: PyTorch và TFLite.
2. PTQ bằng ai-edge-quantizer: dynamic, static toàn bộ op, static chỉ FULLY_CONNECTED.
3. QAT: chèn fake quantize vào mọi nn.Linear (weight INT8 per-channel đối xứng, input INT8 per-tensor,
   khoảng của input lấy bằng EMA 0.95 giống ai-edge-quantizer), fine-tune bằng distillation + nhãn thật,
   rồi convert và quantize lại giống bước 2.

Chạy: python qat_distilbert.py --n_train 4000 --epochs 1
"""
import os
import copy
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import litert_torch
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=4000)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--device", default="mps")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{HERE}/results", exist_ok=True)
MAXLEN = 64
dev = torch.device(args.device)
torch.manual_seed(0)

# ===== Dữ liệu SST-2 =====
tok = AutoTokenizer.from_pretrained(f"{HERE}/model_fp32")
train = pq.read_table(f"{HERE}/data/sst2/train-00000-of-00001.parquet").to_pydict()
val = pq.read_table(f"{HERE}/data/sst2/validation-00000-of-00001.parquet").to_pydict()

def encode(sents):
    e = tok(sents, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]

rng = np.random.default_rng(0)
idx = rng.permutation(len(train["sentence"]))
calib_idx, train_idx = idx[:100], idx[100:100 + args.n_train]           # calibration và train không trùng nhau
calib_ids, calib_mask = encode([train["sentence"][i] for i in calib_idx])
tr_ids, tr_mask = encode([train["sentence"][i] for i in train_idx])
tr_y = torch.tensor([train["label"][i] for i in train_idx])
val_ids, val_mask = encode(val["sentence"])
val_y = np.array(val["label"])
print(f"Train {len(train_idx)} cau, calibration {len(calib_idx)} cau, validation {len(val_y)} cau")

def load_fp32():
    return AutoModelForSequenceClassification.from_pretrained(f"{HERE}/model_fp32", attn_implementation="eager").eval()

class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).logits

# ===== Fake quantize (QAT), viết bằng phép tính thường để chạy được trên MPS =====
def fq(x, scale, zp, qmin, qmax):
    q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
    return x + ((q - zp) * scale - x).detach()                           # STE: forward làm tròn, backward bỏ qua làm tròn

class QATLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin
        self.register_buffer("in_min", torch.tensor(0.0))
        self.register_buffer("in_max", torch.tensor(0.0))
        self.observe = True
        self.started = False

    def forward(self, x):
        if self.observe:                                                 # khoảng input: EMA 0.95 giống ai-edge-quantizer
            mn, mx = x.detach().min(), x.detach().max()
            if not self.started:
                self.in_min.copy_(mn); self.in_max.copy_(mx); self.started = True
            else:
                self.in_min.mul_(0.95).add_(0.05 * mn); self.in_max.mul_(0.95).add_(0.05 * mx)
        lo, hi = torch.minimum(self.in_min, torch.zeros_like(self.in_min)), torch.maximum(self.in_max, torch.zeros_like(self.in_max))
        s = torch.clamp((hi - lo) / 255.0, min=1e-8)
        zp = torch.round(-128 - lo / s)
        x = fq(x, s, zp, -128, 127)                                      # input INT8 bất đối xứng, per-tensor
        w = self.lin.weight
        sw = torch.clamp(w.detach().abs().amax(1, keepdim=True) / 127.0, min=1e-8)
        w = fq(w, sw, 0, -127, 127)                                      # weight INT8 đối xứng, per-channel
        return F.linear(x, w, self.lin.bias)

def add_qat(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, QATLinear(child))
    return model

def remove_qat(model):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, QATLinear):
                setattr(mod, cname, child.lin)
    return model

# ===== Đánh giá =====
@torch.no_grad()
def acc_torch(model, device="cpu"):
    model = model.to(device).eval()
    preds = [model(input_ids=val_ids[i:i + 64].to(device), attention_mask=val_mask[i:i + 64].to(device)).logits.argmax(1).cpu()
             for i in range(0, len(val_y), 64)]
    return 100 * float((torch.cat(preds).numpy() == val_y).mean())

def acc_tflite(path):
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    ins = sorted(it.get_input_details(), key=lambda d: d["name"])      # args_0 = ids, args_1 = mask
    out = it.get_output_details()[0]
    preds = []
    for i in range(len(val_y)):
        it.set_tensor(ins[0]["index"], val_ids[i:i + 1].numpy().astype(ins[0]["dtype"]))
        it.set_tensor(ins[1]["index"], val_mask[i:i + 1].numpy().astype(ins[1]["dtype"]))
        it.invoke()
        y = it.get_tensor(out["index"])
        if np.issubdtype(out["dtype"], np.integer):
            s, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * s
        preds.append(int(y.argmax()))
    return 100 * float((np.array(preds) == val_y).mean())

def speed(path, runs=50):
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    for d in it.get_input_details():
        it.set_tensor(d["index"], val_ids[:1].numpy().astype(d["dtype"]) if "args_0" in d["name"] else val_mask[:1].numpy().astype(d["dtype"]))
    for _ in range(5):
        it.invoke()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); it.invoke(); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))

def to_tflite(model, path):
    litert_torch.convert(Wrap(model.cpu().eval()), (val_ids[:1], val_mask[:1])).export(path)
    return path

def static_fc_recipe():
    r = copy.deepcopy(recipe.static_wi8_ai8()[0])
    r["operation"] = "FULLY_CONNECTED"
    return [r]

def quantize(fp32_path, kind, out_path):
    rcp = {"dynamic": recipe.dynamic_wi8_afp32(), "static_all": recipe.static_wi8_ai8(), "static_fc": static_fc_recipe()}[kind]
    qt = quantizer.Quantizer(fp32_path, rcp)
    if kind.startswith("static"):
        sig_name, sig = list(Interpreter(fp32_path).get_signature_list().items())[0]
        names = sorted(sig["inputs"])
        data = [{names[0]: calib_ids[i:i + 1].numpy(), names[1]: calib_mask[i:i + 1].numpy()} for i in range(len(calib_ids))]
        qt.quantize(qt.calibrate({sig_name: data})).export_model(out_path, overwrite=True)
    else:
        qt.quantize().export_model(out_path, overwrite=True)
    return out_path

rows = []
def report(name, path, extra=None):
    r = {"name": name, "acc": acc_tflite(path), "size_mb": os.path.getsize(path) / 1e6, "ms": speed(path), **(extra or {})}
    rows.append(r)
    print(f"{name:34s} acc {r['acc']:6.2f}%  {r['size_mb']:6.1f} MB  {r['ms']:6.2f} ms", flush=True)

# ===== 1-2. FP32 và PTQ =====
base = load_fp32()
print(f"PyTorch FP32 acc {acc_torch(base):.2f}%")
p32 = to_tflite(base, f"{OUT}/fp32.tflite")
report("TFLite FP32", p32)
report("PTQ dynamic (weight INT8)", quantize(p32, "dynamic", f"{OUT}/ptq_dynamic.tflite"))
report("PTQ static toan bo op", quantize(p32, "static_all", f"{OUT}/ptq_static_all.tflite"))
report("PTQ static chi FC", quantize(p32, "static_fc", f"{OUT}/ptq_static_fc.tflite"))

# ===== 3. QAT =====
teacher = load_fp32().to(dev)
student = add_qat(load_fp32()).to(dev)
with torch.no_grad():                                                     # khởi tạo khoảng input bằng calibration
    for i in range(len(calib_ids)):
        student(input_ids=calib_ids[i:i + 1].to(dev), attention_mask=calib_mask[i:i + 1].to(dev))
print(f"QAT gia lap truoc khi train (PyTorch) acc {acc_torch(student, dev):.2f}%")

opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
student.train()
t0, hist = time.time(), []
for ep in range(args.epochs):
    order = torch.randperm(len(tr_y))
    for s in range(0, len(order), args.bs):
        b = order[s:s + args.bs]
        ids, mask, y = tr_ids[b].to(dev), tr_mask[b].to(dev), tr_y[b].to(dev)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=mask).logits
        logits = student(input_ids=ids, attention_mask=mask).logits
        T = 2.0
        l_kd = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean") * T * T
        l_ce = F.cross_entropy(logits, y)
        loss = 0.5 * l_kd + 0.5 * l_ce
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append([float(l_kd), float(l_ce)])
        if (s // args.bs) % 50 == 0:
            print(f"ep {ep} step {s // args.bs:4d}  kd {np.mean([h[0] for h in hist[-50:]]):.4f}  ce {np.mean([h[1] for h in hist[-50:]]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
train_sec = time.time() - t0
student.eval()
for m in student.modules():
    if isinstance(m, QATLinear):
        m.observe = False
qat_sim = acc_torch(student, dev)
print(f"QAT gia lap sau khi train (PyTorch) acc {qat_sim:.2f}%")

qat_fp = remove_qat(student).cpu()                                       # bỏ fake quantize, giữ weight đã train
p32q = to_tflite(qat_fp, f"{OUT}/qat_fp32.tflite")
report("QAT -> static chi FC", quantize(p32q, "static_fc", f"{OUT}/qat_static_fc.tflite"))
report("QAT -> static toan bo op", quantize(p32q, "static_all", f"{OUT}/qat_static_all.tflite"))
report("QAT -> dynamic", quantize(p32q, "dynamic", f"{OUT}/qat_dynamic.tflite"))

json.dump({"rows": rows, "qat_sim_acc": qat_sim, "train": {"n_train": len(train_idx), "epochs": args.epochs, "bs": args.bs,
           "lr": args.lr, "seconds": train_sec, "device": args.device, "loss_history": hist}},
          open(f"{HERE}/results/qat_distilbert.json", "w"), indent=1)
