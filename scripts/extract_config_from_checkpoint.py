import argparse
import collections
import io
import pickle
import zipfile
from pathlib import Path


class Storage:
    def __init__(self, dtype, key=None, location=None, size=None):
        self.dtype = dtype
        self.key = key
        self.location = location
        self.size = size


class Tensor:
    def __init__(self, storage, offset, size, stride, requires_grad=False, hooks=None):
        self.storage = storage
        self.shape = tuple(size)


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return Tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks)


def rebuild_tensor(storage, storage_offset, size, stride):
    return Tensor(storage, storage_offset, size, stride)


def rebuild_parameter(data, requires_grad, backward_hooks):
    return data


STORAGE_CLASSES = {
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "DoubleStorage": "float64",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "ByteStorage": "uint8",
    "CharStorage": "int8",
    "BoolStorage": "bool",
    "BFloat16Storage": "bfloat16",
}


class SafeTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
            "_rebuild_parameter",
        }:
            return {
                "_rebuild_tensor_v2": rebuild_tensor_v2,
                "_rebuild_tensor": rebuild_tensor,
                "_rebuild_parameter": rebuild_parameter,
            }[name]
        if module == "torch" and name in STORAGE_CLASSES:
            return type(name, (), {"_dtype": STORAGE_CLASSES[name]})
        raise pickle.UnpicklingError(f"Blocked global {module}.{name}")

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            return Storage(getattr(pid[1], "_dtype", str(pid[1])), pid[2], pid[3], pid[4])
        return Storage("unknown", str(pid))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract meta.config from an MMCV/PyTorch checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.checkpoint) as archive:
        obj = SafeTorchUnpickler(io.BytesIO(archive.read("archive/data.pkl"))).load()

    meta = obj.get("meta", {})
    config = meta.get("config")
    if not config:
        raise SystemExit("No meta.config found in checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote config: {args.output}")
    print(f"exp_name: {meta.get('exp_name')}")
    print(f"mmseg_version: {meta.get('mmseg_version')}")
    print(f"mmcv_version: {meta.get('mmcv_version')}")
    print(f"classes: {meta.get('CLASSES')}")


if __name__ == "__main__":
    main()
