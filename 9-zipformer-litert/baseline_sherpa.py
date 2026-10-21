"""Mốc chuẩn: chạy model ONNX gốc bằng sherpa-onnx (streaming, greedy search) trên 3 file âm thanh thử."""
import time
import numpy as np
import sherpa_onnx
import soundfile as sf

D = "model_onnx"
rec = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{D}/tokens.txt", encoder=f"{D}/encoder-epoch-99-avg-1.onnx",
    decoder=f"{D}/decoder-epoch-99-avg-1.onnx", joiner=f"{D}/joiner-epoch-99-avg-1.onnx",
    num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
refs = dict(l.strip().split(" ", 1) for l in open(f"{D}/test_wavs/trans.txt"))
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)

err = tot = 0
for name in ["0.wav", "1.wav", "8k.wav"]:
    audio, sr = sf.read(f"{D}/test_wavs/{name}", dtype="float32")
    t0 = time.perf_counter()
    s = rec.create_stream()
    s.accept_waveform(sr, np.zeros(int(0.5 * sr), np.float32))   # 0.5 s lặng ở đầu: encoder streaming cần ngữ cảnh, không có thì mất vài từ đầu
    s.accept_waveform(sr, audio)
    s.accept_waveform(sr, [0.0] * int(0.66 * sr))          # thêm khoảng lặng để đẩy hết frame cuối
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    dt = time.perf_counter() - t0
    hyp = rec.get_result(s)
    print(f"{name}: {len(audio)/sr:.2f}s am thanh, {dt*1000:.0f} ms, RTF {dt/(len(audio)/sr):.3f}")
    print("  HYP:", hyp)
    print("  REF:", refs[name])
    e, n = wer(refs[name], hyp); err += e; tot += n
print(f"WER tong: {100 * err / tot:.2f}% ({err}/{tot} tu)")
