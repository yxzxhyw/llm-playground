"""llm 环境自检脚本。

用法（在 llm 环境激活后）：
    python check_env.py

它会检查：Python 版本、关键包、GPU 可用性、bf16 支持、矩阵乘法性能、
以及跑一小步训练确认自动求导正常。
"""

import platform
import sys
import time


def title(text: str) -> None:
    print()
    print("=" * 62)
    print(text)
    print("=" * 62)


title("0. 系统")
print("OS      :", platform.platform())
print("Python  :", sys.version.split()[0])
print("解释器  :", sys.executable)

# ---------------------------------------------------------------- 1
title("1. 关键包")
PACKAGES = [
    "torch", "torchvision", "torchaudio",
    "numpy", "pandas", "matplotlib",
    "sklearn", "transformers", "datasets",
    "peft", "accelerate", "tokenizers",
]
missing = []
for name in PACKAGES:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:<14} {ver}")
    except Exception:
        print(f"  {name:<14} -- 未安装")
        missing.append(name)

# ---------------------------------------------------------------- 2
title("2. GPU")
try:
    import torch
except ImportError:
    print("torch 未安装，跳过 GPU 检查。")
    print("请先执行：pip install torch --index-url https://download.pytorch.org/whl/cu128")
    raise SystemExit(1)

print("torch 编译的 CUDA :", torch.version.cuda)
print("cuda.is_available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    print()
    print("!! 检测不到 GPU。常见原因：")
    print("   1) 装成了 CPU 版 torch（torch.version.cuda 为 None）")
    print("   2) 显卡驱动太旧")
    print("   3) 在错误的 conda 环境里运行")
else:
    props = torch.cuda.get_device_properties(0)
    print("显卡            :", props.name)
    print("算力            : sm_%d%d" % (props.major, props.minor))
    print("显存            : %.1f GB" % (props.total_memory / 1024**3))
    print("bf16 支持       :", torch.cuda.is_bf16_supported())
    print("cuDNN 版本      :", torch.backends.cudnn.version())

    # ---- 矩阵乘法测速（确认真的在 GPU 上算，而不是悄悄回退到 CPU）
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    for _ in range(3):          # 预热
        x @ x
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(10):
        x @ x
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    tflops = 2 * 2048**3 / dt / 1e12
    print("2048³ bf16 矩阵乘: %.1f ms  (%.1f TFLOPS)" % (dt * 1000, tflops))
    if tflops < 1:
        print("  ^ 这个数字偏低，可能没走 GPU 或驱动有问题")

# ---------------------------------------------------------------- 3
title("3. 自动求导：跑一小步训练")
import torch.nn as nn

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1),
).to(dev)

opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
X = torch.randn(1024, 64, device=dev)
y = (X.sum(dim=1, keepdim=True) * 0.5 + 1.0)

first = last = None
for step in range(200):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    if step == 0:
        first = loss.item()
    last = loss.item()

print(f"设备: {dev}")
print(f"loss: {first:.4f}  ->  {last:.4f}")
if last < first * 0.5:
    print("✅ 自动求导和优化器工作正常")
else:
    print("⚠️ loss 没怎么下降，检查环境或代码")

# ---------------------------------------------------------------- 4
title("4. 结论")
if not missing:
    print("✅ 所有关键包已安装")
else:
    print("还缺这些包：")
    for m in missing:
        print("  -", m)
    print()
    print("一次性装齐：")
    print("  pip install numpy pandas matplotlib scikit-learn \\")
    print("              transformers datasets peft accelerate tokenizers")

print()
print("环境检查结束。")
