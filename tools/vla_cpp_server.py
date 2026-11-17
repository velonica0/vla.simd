#!/usr/bin/env python3
"""Serve a vla.simd SmolVLA or TurboVLA checkpoint over vla.cpp's ZeroMQ + protobuf protocol.

vla.cpp's simulator clients (eval/client/run_sim_client_direct.py for LIBERO)
then run against vla.simd unchanged: only --vla-addr differs, so task-success
numbers for the two engines come from the same client, preprocessing and seeds.

    python tools/vla_cpp_server.py --model smolvla --model-dir smolvla-libero.gguf \\
        --proto ~/vla.cpp/src/serving/vla.proto --bind tcp://127.0.0.1:5556
    # client, in the vla.cpp checkout:
    python eval/client/run_sim_client_direct.py --arch smolvla --vla-addr tcp://127.0.0.1:5556 ...

The protocol module comes from --pb-dir (a directory holding a vla_pb2.py
generated from vla.cpp's src/serving/vla.proto) or is compiled from --proto
with protoc, as vla.cpp's client does. The socket is unauthenticated: bind to
loopback unless the client runs on another machine you trust.

Request mapping, vla.cpp -> vla.simd:

  smolvla   images  float RGB in [0,1], already flipped and resize-padded by
                    the client -> uint8 (round to nearest), the engine's format
            tokens  SmolVLM2 ids -> ids + attention mask padded to the
                    checkpoint's tokenizer length
            state   raw, zero-padded to max_state_dim -> first state_dim values
            noise   [chunk * 32] when the client pins it -> passed through
            reply   actions in env units, zero-padded to --max-action-dim columns
  turbovla  images  as above (256x256 resize-padded by the client)
            tokens  BERT ids -> decoded back to the instruction text, which the
                    engine tokenizes and pads per instruction itself
            state   normalized by the client with the release statistics ->
                    un-normalized with the checkpoint's stats.bin, since the
                    engine normalizes raw state
            reply   NORMALIZED actions (post-tanh): the client maps them to env
                    units, gripper included
            Both conversions are checked on every new instruction against the
            engine's own trace (input ids and normalized state); a mismatch
            means the client and checkpoint statistics or vocabularies differ,
            and the request fails instead of running on shifted inputs.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_simd import gguf_stage  # noqa: E402
from vla_simd.policy_server import (F32P, I32P, U8P, SmolvlaEngine,  # noqa: E402
                                    TurboVlaEngine)

TVLA_INPUT_IDS, TVLA_STATE_NORM = 4, 13   # vla_turbovla_tensor_id (include/vla_simd.h)


def load_pb(pb_dir, proto):
    if pb_dir:
        sys.path.insert(0, str(Path(pb_dir).expanduser()))
    else:
        proto = Path(proto or os.environ.get("VLA_CPP_PROTO", "")).expanduser()
        if not proto.is_file():
            sys.exit("pass --pb-dir (holding vla_pb2.py) or --proto <vla.cpp>/src/serving/vla.proto")
        out = tempfile.mkdtemp(prefix="vla-pb-")
        try:
            subprocess.check_call(["protoc", f"--proto_path={proto.parent}", f"--python_out={out}", str(proto)])
        except FileNotFoundError:
            from grpc_tools import protoc   # pip install grpcio-tools, when there is no protoc
            if protoc.main(["protoc", f"--proto_path={proto.parent}", f"--python_out={out}", str(proto)]):
                sys.exit(f"cannot compile {proto}")
        sys.path.insert(0, out)
    import vla_pb2
    return vla_pb2


def frames_u8(req, pb, n_views):
    if len(req.images) != n_views:
        raise ValueError(f"{len(req.images)} images, checkpoint wants {n_views}")
    frames = []
    for im in req.images:
        if im.encoding == pb.Image.F32_RGB_01:
            a = np.frombuffer(im.data, np.float32).reshape(im.height, im.width, 3)
            a = np.clip(np.rint(a*255.0), 0, 255).astype(np.uint8)
        elif im.encoding == pb.Image.RGB_U8:
            a = np.frombuffer(im.data, np.uint8).reshape(im.height, im.width, 3)
        else:
            raise ValueError("JPEG images are not supported")
        frames.append(a)
    return np.ascontiguousarray(np.stack(frames))


class Smolvla:
    def __init__(self, args, model_dir):
        self.eng = SmolvlaEngine(SimpleNamespace(
            lib=args.lib, model_dir=model_dir, tok_dir=None,
            rtc_horizon=0, rtc_max_guidance=10.0, rtc_delay=None, fps=20))
        self.max_action_dim = args.max_action_dim

    def describe(self):
        return f"smolvla: {self.eng.describe()} | cams {self.eng.cam_names}"

    def predict(self, req, pb):
        eng = self.eng
        frames = frames_u8(req, pb, eng.n_views)
        toks = np.asarray(req.lang_tokens, np.int32)[:eng.tok_maxlen]
        ids = np.zeros(eng.tok_maxlen, np.int32)
        mask = np.zeros(eng.tok_maxlen, np.int32)
        ids[:len(toks)] = toks
        mask[:len(toks)] = 1
        state = eng._state(np.asarray(req.state, np.float32)[:eng.state_dim])
        noise = None
        if len(req.noise):
            want = eng.chunk*self.max_action_dim
            if len(req.noise) != want:
                raise ValueError(f"noise has {len(req.noise)} values, expected {want}")
            noise = np.ascontiguousarray(req.noise, np.float32)
        out = eng._run(frames.ctypes.data_as(U8P), eng.n_views, frames.shape[1], frames.shape[2],
                       ids.ctypes.data_as(I32P), mask.ctypes.data_as(I32P), eng.tok_maxlen,
                       state.ctypes.data_as(F32P),
                       noise.ctypes.data_as(F32P) if noise is not None else None,
                       ctypes.c_uint64(req.request_id or time.time_ns()))
        padded = np.zeros((eng.chunk, self.max_action_dim), np.float32)
        padded[:, :eng.action_dim] = out
        return padded


class Turbovla:
    def __init__(self, args, model_dir):
        self.eng = eng = TurboVlaEngine(SimpleNamespace(lib=args.lib, model_dir=model_dir, tok_dir=None))
        st = np.fromfile(os.path.join(model_dir, "stats.bin"), np.float32)
        s = eng.state_dim
        self.mean, self.std = st[:s], st[s:2*s]
        self.vocab = Path(model_dir, "vocab.txt").read_text(encoding="utf-8").split("\n")
        self.special = {"[CLS]", "[SEP]", "[PAD]"}
        self.checked = set()
        self.tensor = eng._fn("tensor")
        self.tensor.argtypes = [ctypes.c_void_p, ctypes.c_int32, F32P, ctypes.c_int32]
        self.tensor.restype = ctypes.c_int32

    def describe(self):
        return f"turbovla: {self.eng.describe()} | cams {self.eng.cam_names}"

    def _decode(self, ids):
        words = []
        for i in ids:
            if not 0 <= i < len(self.vocab):
                raise ValueError(f"token id {i} outside the {len(self.vocab)}-entry vocabulary")
            w = self.vocab[i]
            if w in self.special:
                continue
            if w.startswith("##") and words:
                words[-1] += w[2:]
            else:
                words.append(w)
        return " ".join(words)

    def _trace(self, which):
        n = self.tensor(self.eng.h, which, None, 0)
        out = np.empty(max(n, 0), np.float32)
        if n <= 0 or self.tensor(self.eng.h, which, out.ctypes.data_as(F32P), n) != n:
            raise RuntimeError(f"cannot read engine trace tensor {which}")
        return out

    def predict(self, req, pb):
        eng = self.eng
        frames = eng._frames(frames_u8(req, pb, eng.n_views),
                             (eng.n_views, eng.img_size, eng.img_size, 3))
        ids = list(req.lang_tokens)
        task = self._decode(ids)
        if not task:
            raise ValueError("empty instruction")
        state_n = np.asarray(req.state, np.float32)[:eng.state_dim]
        state = eng._state(state_n*(self.std + 1e-6) + self.mean)
        out = eng._run(frames.ctypes.data_as(U8P), state.ctypes.data_as(F32P), task.encode(), 0)
        if task not in self.checked:
            got = self._trace(TVLA_INPUT_IDS).astype(np.int64)
            if list(got[:len(ids)]) != ids:
                raise ValueError(f"instruction {task!r} re-tokenizes to {got[:len(ids)].tolist()}, "
                                 f"client sent {ids}")
            d = float(np.abs(self._trace(TVLA_STATE_NORM) - state_n).max())
            if d > 1e-4:
                raise ValueError(f"normalized state differs from the client's by {d:.2e}: "
                                 "client and checkpoint statistics disagree")
            self.checked.add(task)
            print(f"checked {task!r}: ids match, state normalization within {d:.1e}", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("smolvla", "turbovla"), required=True)
    ap.add_argument("--model-dir", required=True, help="a vla.simd or vla.cpp .gguf, or a staged directory")
    ap.add_argument("--lib", default=None, help="engine library (default: the installed or built one)")
    ap.add_argument("--bind", default="tcp://127.0.0.1:5556")
    ap.add_argument("--pb-dir", default=None, help="directory holding vla_pb2.py")
    ap.add_argument("--proto", default=None, help="vla.cpp's src/serving/vla.proto, compiled with protoc")
    ap.add_argument("--max-action-dim", type=int, default=32, help="SmolVLA reply width (vla-server's)")
    ap.add_argument("--save-requests", default=None, help="directory to save the first 5 raw requests")
    args = ap.parse_args()

    pb = load_pb(args.pb_dir, args.proto)
    if args.lib is None:
        from vla_simd.policy_server import MODELS
        args.lib = MODELS[args.model].default_lib
    model_dir = gguf_stage.stage(os.path.expanduser(args.model_dir), args.model)
    model = (Smolvla if args.model == "smolvla" else Turbovla)(args, model_dir)
    print(f"vla.simd {model.describe()} | {model_dir}", flush=True)
    save_dir = Path(args.save_requests).expanduser() if args.save_requests else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(args.bind)
    print(f"vla_cpp_server: bound to {args.bind}. ready.", flush=True)
    n = 0
    while True:
        body = sock.recv()
        t0 = time.perf_counter()
        req, resp = pb.PredictRequest(), pb.PredictResponse()
        try:
            req.ParseFromString(body)
            resp.request_id = req.request_id
            if save_dir and n < 5:
                (save_dir / f"req_{n:03d}.bin").write_bytes(body)
            chunk = model.predict(req, pb)
            resp.action_chunk.extend(chunk.reshape(-1).tolist())
            resp.chunk_size, resp.action_dim = chunk.shape
        except Exception as e:  # report to the client instead of dying
            resp.error = f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0)*1000
        resp.latency_ms_total = resp.latency_ms_inference = ms
        sock.send(resp.SerializeToString())
        print(f"request {n}: {ms:.0f} ms, {len(req.lang_tokens)} tokens, "
              f"noise={'pinned' if len(req.noise) else 'drawn'}"
              + (f", ERROR {resp.error}" if resp.error else ""), flush=True)
        n += 1


if __name__ == "__main__":
    main()
