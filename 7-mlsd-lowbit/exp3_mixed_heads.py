"""Thí nghiệm 3: mixed precision có chọn lọc = 3 layer nhạy nhất (theo exp2) + head displacement/center giữ float."""
import os, json, copy
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp2_sensitivity_mixed.py")).read()
exec(src[:src.index("# ===== (a)")])          # nạp dữ liệu, danh sách layer, calibration, hàm build/score

sens = json.load(open(f"{ROOT}/results/exp2.json"))["sensitivity"]
order = sorted(range(len(sens)), key=lambda i: sens[i]["lines_kept"])
top3 = [sens[i]["regex"] for i in order[:3]]
rx = {c["short"]: c["regex"] for c in convs}
disp_rx, center_rx = rx["conv.Conv2d_disp"], rx["conv.Conv2d_center"]

configs = {
    "INT8 het (tach head)": [],
    "float: 3 layer nhay": top3,
    "float: head disp": [disp_rx],
    "float: 3 layer nhay + head disp": top3 + [disp_rx],
    "float: 3 layer nhay + 2 head": top3 + [disp_rx, center_rx],
}
rows = []
for name, keep in configs.items():
    path = f"{MODELS}/split_" + str(abs(hash(name)) % 10**6) + ".tflite"
    m = score(build([copy.deepcopy(STATIC)] + [no_q(r) for r in keep], path))
    rows.append({"name": name, "size_mb": os.path.getsize(path) / 1e6, "ms": benchmark(path), **m})
    print(f"{name:34s} giu {m['lines_kept']:5.1f} center {m['center_recall']:5.1f} Dmed {m['disp_err_median_px']:.2f} "
          f"D<2px {m['disp_within_2px']:5.1f} doan {m['n_lines']:5d} | {rows[-1]['size_mb']:.2f} MB {rows[-1]['ms']:.1f} ms")
print("3 layer nhay:", [sens[i]["short"] for i in order[:3]])
json.dump(rows, open(f"{ROOT}/results/exp3.json", "w"), indent=1)
