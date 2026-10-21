"""Pipeline nhận dạng streaming Zipformer transducer, dùng chung cho ONNX Runtime và LiteRT.

  âm thanh -> fbank 80 chiều (kaldi-native-fbank) -> encoder theo từng chunk 39 frame (dịch 32 frame, kèm 35 tensor trạng thái)
           -> giải mã greedy: decoder (2 token gần nhất) + joiner cho từng frame output của encoder.

Chạy: python asr_pipeline.py --backend onnx
      python asr_pipeline.py --backend tflite --encoder out_x/encoder.tflite --decoder ... --joiner ...
"""
import time
import argparse
import numpy as np
import soundfile as sf
import kaldi_native_fbank as knf

D = "model_onnx"
ONNX = {k: f"{D}/{k}-epoch-99-avg-1.onnx" for k in ["encoder", "decoder", "joiner"]}
T, SHIFT, CONTEXT, BLANK = 39, 32, 2, 0


# ============================== Backend ==============================
class OnnxModel:
    def __init__(self, path, threads=4):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.sess.get_inputs()}

    def input_spec(self):
        dt = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32}
        return {n: ([1 if not isinstance(d, int) else d for d in i.shape], dt[i.type]) for n, i in self.inputs.items()}

    def __call__(self, feeds):
        names = [o.name for o in self.sess.get_outputs()]
        return dict(zip(names, self.sess.run(None, {k: v for k, v in feeds.items() if k in self.inputs})))


def _clean(name):
    for p in ("serving_default_", ):
        if name.startswith(p):
            name = name[len(p):]
    return name.split(":")[0]

class TFLiteModel:
    """onnx2tf đổi layout input sang channel-last (chiều thứ 2 dời xuống cuối). Lớp này nhận dữ liệu theo
    layout của ONNX rồi tự transpose, nên pipeline bên ngoài không cần biết."""
    def __init__(self, path, threads=4, onnx_path=None):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.ins = {_clean(d["name"]): d for d in self.it.get_input_details()}
        self.outs = {_clean(d["name"]): d for d in self.it.get_output_details()}
        self.onnx_shape, self.perm = {}, {}
        if onnx_path:
            import onnx
            for i in onnx.load(onnx_path).graph.input:
                s = [d.dim_value if d.dim_value else 1 for d in i.type.tensor_type.shape.dim]
                self.onnx_shape[i.name] = s
                tfl = [int(x) for x in self.ins[i.name]["shape"]]
                if s != tfl:
                    p = [0] + list(range(2, len(s))) + [1]          # chiều 1 -> cuối
                    assert [s[k] for k in p] == tfl, (i.name, s, tfl)
                    self.perm[i.name] = p

    def input_spec(self):
        if self.onnx_shape:
            return {n: (self.onnx_shape[n], d["dtype"]) for n, d in self.ins.items()}
        return {n: (list(d["shape"]), d["dtype"]) for n, d in self.ins.items()}

    def __call__(self, feeds):
        for n, d in self.ins.items():
            v = feeds[n]
            if n in self.perm:
                v = np.ascontiguousarray(np.transpose(v, self.perm[n]))
            self.it.set_tensor(d["index"], v.astype(d["dtype"], copy=False))
        self.it.invoke()
        return {n: self.it.get_tensor(d["index"]) for n, d in self.outs.items()}


# ============================== Đặc trưng âm thanh ==============================
def load_audio(path, lead=0.5, tail=0.66):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 16000:
        from scipy.signal import resample_poly
        a = resample_poly(a, 16000, sr).astype(np.float32)
    z = lambda s: np.zeros(int(s * 16000), np.float32)
    return np.concatenate([z(lead), a, z(tail)]), len(a) / 16000   # lặng đầu: encoder streaming cần ngữ cảnh

def fbank(audio):
    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.snip_edges = False
    o.frame_opts.samp_freq = 16000
    o.mel_opts.num_bins = 80
    o.mel_opts.low_freq = 20
    o.mel_opts.high_freq = -400
    f = knf.OnlineFbank(o)
    f.accept_waveform(16000, audio.tolist())
    f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


# ============================== Nhận dạng ==============================
class Recognizer:
    def __init__(self, encoder, decoder, joiner, tokens=f"{D}/tokens.txt"):
        self.enc, self.dec, self.joi = encoder, decoder, joiner
        self.sym = {int(i): s for s, i in (l.split() for l in open(tokens))}
        self.state_names = [n for n in encoder.input_spec() if n != "x"]

    def init_states(self):
        spec = self.enc.input_spec()
        return {n: np.zeros(spec[n][0], spec[n][1]) for n in self.state_names}

    def decoder_out(self, hyp):
        y = np.array([hyp[-CONTEXT:]], dtype=np.int64)
        return self.dec({"y": y})["decoder_out"]

    def recognize(self, path, timing=None):
        audio, dur = load_audio(path)
        feats = fbank(audio)
        states = self.init_states()
        hyp = [BLANK] * CONTEXT
        dec_out = self.decoder_out(hyp)
        t_enc = t_dj = 0.0
        start = 0
        while start + T <= len(feats):
            x = feats[start:start + T][None]
            t0 = time.perf_counter()
            out = self.enc({"x": x, **states})
            t_enc += time.perf_counter() - t0
            states = {n: out["new_" + n] for n in self.state_names}
            t0 = time.perf_counter()
            for frame in out["encoder_out"][0]:
                logit = self.joi({"encoder_out": frame[None], "decoder_out": dec_out})["logit"]
                y = int(logit.argmax())
                if y != BLANK:
                    hyp.append(y)
                    dec_out = self.decoder_out(hyp)
            t_dj += time.perf_counter() - t0
            start += SHIFT
        text = "".join(self.sym[t] for t in hyp[CONTEXT:]).replace("▁", " ").strip()
        if timing is not None:
            timing.update({"audio_s": dur, "encoder_s": t_enc, "dec_join_s": t_dj})
        return text


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def build(backend, enc=None, dec=None, joi=None, threads=4):
    if backend == "onnx":
        return Recognizer(OnnxModel(enc or ONNX["encoder"], threads), OnnxModel(dec or ONNX["decoder"], threads),
                          OnnxModel(joi or ONNX["joiner"], threads))
    return Recognizer(TFLiteModel(enc, threads, ONNX["encoder"]), TFLiteModel(dec, threads, ONNX["decoder"]),
                      TFLiteModel(joi, threads, ONNX["joiner"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["onnx", "tflite"], default="onnx")
    ap.add_argument("--encoder"); ap.add_argument("--decoder"); ap.add_argument("--joiner")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    rec = build(a.backend, a.encoder, a.decoder, a.joiner, a.threads)
    refs = dict(l.strip().split(" ", 1) for l in open(f"{D}/test_wavs/trans.txt"))
    err = tot = 0
    for name in ["0.wav", "1.wav", "8k.wav"]:
        tm = {}
        hyp = rec.recognize(f"{D}/test_wavs/{name}", tm)
        e, n = wer(refs[name], hyp); err += e; tot += n
        print(f"{name}: encoder {tm['encoder_s']*1000:.0f} ms, decoder+joiner {tm['dec_join_s']*1000:.0f} ms, "
              f"RTF {(tm['encoder_s'] + tm['dec_join_s']) / tm['audio_s']:.3f} | {hyp}")
    print(f"WER: {100 * err / tot:.2f}% ({err}/{tot})")
