"""Loại op nào làm DistilBERT hỏng khi quantize INT8 toàn bộ?

Quantize 38 lớp Linear (FULLY_CONNECTED) + thêm đúng 1 nhóm op, đo accuracy trên 872 câu SST-2 validation.
Dùng lại dữ liệu và hàm của qat_distilbert.py (không chạy phần train).
"""
import os
import sys
import copy
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
src = open(f"{HERE}/qat_distilbert.py").read()
exec(src[:src.index("rows = []")])                    # dữ liệu, acc_tflite, ... (không train)

STATIC = recipe.static_wi8_ai8()[0]
GROUPS = {
    "chi Linear": [],
    "+ matmul trong attention (BATCH_MATMUL)": ["BATCH_MATMUL"],
    "+ Softmax": ["SOFTMAX"],
    "+ GELU": ["GELU"],
    "+ LayerNorm (MEAN, SUB, SQUARED_DIFF, RSQRT)": ["MEAN", "SUB", "SQUARED_DIFFERENCE", "RSQRT"],
    "+ ADD, MUL (residual, LayerNorm, mask)": ["ADD", "MUL"],
    "tat ca tru ADD, MUL": ["BATCH_MATMUL", "SOFTMAX", "GELU", "MEAN", "SUB", "SQUARED_DIFFERENCE", "RSQRT"],
}
# Embedding không có trong danh sách: ai-edge-quantizer không hỗ trợ static INT8 cho EMBEDDING_LOOKUP

def rule(op):
    r = copy.deepcopy(STATIC)
    r["operation"] = op
    return r

p32 = f"{OUT}/fp32.tflite"
sig_name, sig = list(Interpreter(p32).get_signature_list().items())[0]
names = sorted(sig["inputs"])
calib = {sig_name: [{names[0]: calib_ids[i:i + 1].numpy(), names[1]: calib_mask[i:i + 1].numpy()} for i in range(len(calib_ids))]}

rows = []
for name, ops in GROUPS.items():
    rcp = [rule("FULLY_CONNECTED")] + [rule(op) for op in ops]
    path = f"{OUT}/opsens.tflite"
    qt = quantizer.Quantizer(p32, rcp)
    qt.quantize(qt.calibrate(calib)).export_model(path, overwrite=True)
    acc = acc_tflite(path)
    rows.append({"group": name, "ops": ops, "acc": acc})
    print(f"{name:48s} acc {acc:6.2f}%", flush=True)

json.dump(rows, open(f"{HERE}/results/op_sensitivity.json", "w"), indent=1)
