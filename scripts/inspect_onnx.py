import onnx, sys, json
m = onnx.load(sys.argv[1])
ir = m.ir_version
opset = [(o.domain or "ai.onnx", o.version) for o in m.opset_import]
inputs = []
for i in m.graph.input:
    dims = []
    for d in i.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else (d.dim_param or "?"))
    inputs.append({"name": i.name, "dims": dims, "dtype": i.type.tensor_type.elem_type})
outputs = []
for o in m.graph.output:
    dims = []
    for d in o.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else (d.dim_param or "?"))
    outputs.append({"name": o.name, "dims": dims, "dtype": o.type.tensor_type.elem_type})
print(json.dumps({
    "ir_version": ir,
    "opsets": opset,
    "inputs": inputs,
    "outputs": outputs,
    "n_nodes": len(m.graph.node),
    "producer": m.producer_name,
}, indent=2))
