import sys

import numpy as np
import onnx
import onnxruntime as ort

REQ = {
    "inputs": {
        "x_parent": [1, 1, 8, 8, 8],
        "x_biome": [1, "NB", 8, 8, 1],  # NB > 1
        "x_height": [1, 1, 8, 8, 1],
        "x_lod": [1, 1],
    },
    "outputs": {"air_mask": [1, 1, 16, 16, 16], "block_logits": [1, 1104, 16, 16, 16]},
    "opset_min": 17,
    "forbid_ops": {"Loop", "If", "Scan"},
}


def get_shape(v):
    t = v.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.dim_value:
            dims.append(d.dim_value)
        elif d.dim_param:
            return None  # dynamic
        else:
            return None
    return dims


def main(path):
    m = onnx.load(path)
    onnx.checker.check_model(m)

    # opset
    opset = max(o.version for o in m.opset_import)
    if opset < REQ["opset_min"]:
        print(f"FAIL: opset {opset} < {REQ['opset_min']}")
        return 1

    # control-flow ban
    bad = {n.op_type for n in m.graph.node} & REQ["forbid_ops"]
    if bad:
        print(f"FAIL: control-flow ops present: {sorted(bad)}")
        return 1

    # names → shapes
    ins = {i.name: get_shape(i) for i in m.graph.input}
    outs = {o.name: get_shape(o) for o in m.graph.output}

    # required names
    for name, shp in REQ["inputs"].items():
        if name not in ins:
            print(f"FAIL: missing input {name}")
            return 1
        if ins[name] is None:
            print(f"FAIL: input {name} has dynamic shape")
            return 1
    for name, shp in REQ["outputs"].items():
        if name not in outs:
            print(f"FAIL: missing output {name}")
            return 1
        if outs[name] is None:
            print(f"FAIL: output {name} has dynamic shape")
            return 1

    # shape checks (NB allowed to vary but >1)
    def eq(a, b):
        return all(x == y for x, y in zip(a, b))

    if ins["x_parent"] != REQ["inputs"]["x_parent"]:
        print(f"FAIL: x_parent shape - got: {ins['x_parent']}, want: {REQ['inputs']['x_parent']}")
        return 1
    if ins["x_height"] != REQ["inputs"]["x_height"]:
        print(f"FAIL: x_height shape - got: {ins['x_height']}, want: {REQ['inputs']['x_height']}")
        return 1
    if ins["x_lod"] != REQ["inputs"]["x_lod"]:
        print(f"FAIL: x_lod shape - got: {ins['x_lod']}, want: {REQ['inputs']['x_lod']}")
        return 1
    xb = ins["x_biome"]
    if not (
        len(xb) == 5 and xb[0] == 1 and xb[2:] == [8, 8, 1] and isinstance(xb[1], int) and xb[1] > 1
    ):
        print(f"FAIL: x_biome shape {xb}")
        return 1

    if outs["air_mask"] != REQ["outputs"]["air_mask"]:
        print(f"FAIL: air_mask shape {outs['air_mask']}")
        return 1
    if outs["block_logits"] != REQ["outputs"]["block_logits"]:
        print(f"FAIL: block_logits shape {outs['block_logits']}")
        return 1

    # runtime smoke: single-thread, dummy inputs
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 1
    sess_opt.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=sess_opt, providers=["CPUExecutionProvider"])
    NB = xb[1]
    x = {
        "x_parent": np.zeros((1, 1, 8, 8, 8), np.float32),
        "x_biome": np.eye(NB, dtype=np.float32)[np.zeros((1, 8, 8, 1), int)].transpose(
            0, 4, 1, 2, 3
        ),  # one-hot(0)
        "x_height": np.zeros((1, 1, 8, 8, 1), np.float32),
        "x_lod": np.zeros((1, 1), np.float32),
    }
    y = sess.run(None, x)
    ok = y[1].shape == tuple(REQ["outputs"]["air_mask"]) and y[0].shape == tuple(
        REQ["outputs"]["block_logits"]
    )
    if not ok:
        print(f"FAIL: runtime output shapes {y[0].shape}, {y[1].shape}")
        return 1

    print(f"READY: opset {opset}, NB={NB}, shapes OK, runtime OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/quick_test/model.onnx"))
