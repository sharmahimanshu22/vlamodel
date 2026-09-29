"""Sanity-check the environment on a GPU node: CUDA, bf16, video decoding, LIBERO rendering.

Writes a rendered LIBERO frame to $RUN_DIR (or the current dir) so rendering can be checked by eye.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np


def section(title: str) -> None:
    print(f"\n== {title}", flush=True)


def check_torch() -> None:
    import torch

    section("torch")
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available"
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        x @ x
    torch.cuda.synchronize()
    print(f"bf16 4096^2 matmul x10: {time.time() - t:.3f}s")


def check_torchcodec() -> None:
    section("torchcodec")
    try:
        import torchcodec
        from torchcodec.decoders import VideoDecoder  # noqa: F401  (fails if FFmpeg libs are missing)
    except RuntimeError:
        _explain_torchcodec_load_failure()
        raise
    print("torchcodec", torchcodec.__version__)


def _explain_torchcodec_load_failure() -> None:
    """torchcodec hides the loader error; dlopen its libraries directly to show the real cause."""
    import ctypes
    import importlib.util
    from importlib.metadata import version

    import torch  # noqa: F401  (loads libtorch symbols the torchcodec libraries link against)

    print("torchcodec", version("torchcodec"), "- real loader errors:")
    pkg_dir = Path(importlib.util.find_spec("torchcodec").submodule_search_locations[0])
    for lib in sorted(pkg_dir.glob("libtorchcodec_core*.so")):
        try:
            ctypes.CDLL(str(lib))
            print(f"  OK   {lib.name}")
        except OSError as e:
            print(f"  FAIL {lib.name}: {e}")
    env_lib = Path(sys.prefix) / "lib"
    print("  FFmpeg libs in env:", sorted(p.name for p in env_lib.glob("libavcodec.so.*")) or "none")


def check_libero_render(out_dir: Path) -> None:
    section("LIBERO render (EGL)")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    print("task:", task.language)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    img = obs["agentview_image"]
    print("agentview_image", img.shape, img.dtype, "mean pixel", float(img.mean()))
    assert img.mean() > 1, "rendered image is black: EGL rendering is not working"

    from PIL import Image

    path = out_dir / "libero_render.png"
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(path)  # LIBERO frames are upside down
    print("saved", path)
    env.close()


def main() -> None:
    out_dir = Path(os.environ.get("RUN_DIR", "."))
    print("python", sys.version.split()[0])
    for k in ["MUJOCO_GL", "HF_HOME", "HF_HUB_OFFLINE", "LIBERO_CONFIG_PATH", "CUDA_VISIBLE_DEVICES"]:
        print(f"{k}={os.environ.get(k)}")
    check_torch()
    check_torchcodec()
    check_libero_render(out_dir)
    section("lerobot")
    from importlib.metadata import version

    print("lerobot", version("lerobot"))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
