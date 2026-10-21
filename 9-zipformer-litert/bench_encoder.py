"""Đo encoder: thời gian 1 chunk (39 frame, ~320 ms âm thanh), độ lệch output so với ONNX gốc, số op theo loại.

Chạy: python bench_encoder.py <file.tflite> [<file.tflite> ...]
"""
import sys
import time
import collections
import numpy as np
from asr_pipeline import OnnxModel, TFLiteModel, ONNX, load_audio, fbank, T, SHIFT

THREADS = 4

def chunks(n=20):
    feats = fbank(load_audio("model_onnx/test_wavs/1.wav")[0])
    return [feats[s:s + T][None] for s in range(0, len(feats) - T, SHIFT)][:n]

def run_stream(model, xs, state_names, spec):
    states = {k: np.zeros(spec[k][0], spec[k][1]) for k in state_names}
    outs = []
    for x in xs:
        o = model({"x": x, **states})
        states = {k: o["new_" + k] for k in state_names}
        outs.append(o["encoder_out"])
    return np.concatenate(outs, 1), states

def latency(model, xs, state_names, spec, warmup=5, runs=50):
    _, st = run_stream(model, xs[:3], state_names, spec)
    feeds = {"x": xs[3], **st}
    for _ in range(warmup):
        model(feeds)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); model(feeds); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 90))

def op_count(path):
    from ai_edge_litert import schema_py_generated as s
    m = s.Model.GetRootAsModel(open(path, "rb").read(), 0)
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if not k.startswith("_")}
    codes = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        codes.append(oc.CustomCode().decode() if oc.CustomCode() else names.get(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())))
    g = m.Subgraphs(0)
    return collections.Counter(codes[g.Operators(j).OpcodeIndex()] for j in range(g.OperatorsLength()))

if __name__ == "__main__":
    xs = chunks()
    ref = OnnxModel(ONNX["encoder"], THREADS)
    spec = ref.input_spec()
    names = [n for n in spec if n != "x"]
    ref_out, _ = run_stream(ref, xs, names, spec)
    med, p90 = latency(ref, xs, names, spec)
    print(f"{'ONNX Runtime (goc)':48s} {med:6.2f} ms (p90 {p90:.2f}) | 1 chunk = 320 ms am thanh")
    for path in sys.argv[1:]:
        m = TFLiteModel(path, THREADS, ONNX["encoder"])
        out, _ = run_stream(m, xs, names, spec)
        d = ref_out.astype(np.float64) - out
        snr = 10 * np.log10((ref_out.astype(np.float64) ** 2).sum() / max((d ** 2).sum(), 1e-30))
        med, p90 = latency(m, xs, names, spec)
        c = op_count(path)
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(6))
        print(f"{path.split('/')[-1][:48]:48s} {med:6.2f} ms (p90 {p90:.2f}) | SNR {snr:6.1f} dB, max err {np.abs(d).max():.1e} "
              f"| {sum(c.values())} op: {top}")
