#!/usr/bin/env python3
"""
GPU / CUDA 环境自检脚本（Recirculation 复现前置检查）
用法:  python3 check_gpu.py
"""
import sys
import platform

print("=" * 60)
print("GPU / CUDA 环境自检")
print("=" * 60)

# 1. 系统信息
print("\n[1] 系统信息")
print(f"  Python      : {platform.python_version()}")
print(f"  系统        : {platform.system()} {platform.release()}")

# 2. PyTorch 与 CUDA 编译版本
try:
    import torch
    print("\n[2] PyTorch 信息")
    print(f"  torch 版本          : {torch.__version__}")
    print(f"  torch 编译 CUDA 版本 : {torch.version.cuda}")
    print(f"  cudnn 版本          : {torch.backends.cudnn.version()}")
    print(f"  torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count() : {torch.cuda.device_count()}")
except ImportError as e:
    print(f"\n[2] PyTorch 未安装或导入失败: {e}")
    sys.exit(1)

# 3. 设备枚举（即使 is_available 为 False 也尝试原生枚举）
print("\n[3] CUDA 设备枚举（原生 API）")
try:
    raw_count = torch.cuda._raw_device_count_nvml()
    print(f"  NVML 原始设备数: {raw_count}")
except Exception as e:
    print(f"  NVML 原始枚举失败: {type(e).__name__}: {e}")

try:
    raw_count2 = torch.cuda._raw_device_count()
    print(f"  CUDA 原始设备数: {raw_count2}")
except Exception as e:
    print(f"  CUDA 原始枚举失败: {type(e).__name__}: {e}")

# 4. 设备属性（逐设备打印）
n = torch.cuda.device_count()
if n > 0:
    for i in range(n):
        try:
            p = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {p.name}")
            print(f"    显存      : {p.total_memory / 1024**3:.2f} GB")
            print(f"    SM 数     : {p.multi_processor_count}")
            print(f"    计算能力  : {p.major}.{p.minor}")
        except Exception as e:
            print(f"  GPU {i} 属性读取失败: {e}")
else:
    print("  (无可用 CUDA 设备)")

# 5. 实际计算测试（有设备才跑）
if torch.cuda.is_available():
    print("\n[4] 实际计算测试")
    try:
        dev = "cuda:0"
        a = torch.randn(1024, 1024, device=dev)
        b = torch.matmul(a, a)
        torch.cuda.synchronize()
        print(f"  矩阵乘法测试通过: {a.shape} -> {b.shape} (示例值 {b[0, 0].item():.4f})")
        print(f"  当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB / "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("\n结论: CUDA 完全可用，可以直接跑 GPU 训练/推理。")
    except Exception as e:
        print(f"  计算测试失败: {type(e).__name__}: {e}")
        print("\n结论: CUDA 设备可见但计算异常，请检查驱动。")
else:
    print("\n[4] 实际计算测试: 跳过（无 CUDA 设备）")
    print("\n结论: PyTorch 无法使用 GPU。可能原因:")
    print("  1. WSL2 会话启动时 GPU 未透传 → 在 Windows 执行 'wsl --shutdown' 后重开 WSL")
    print("  2. NVIDIA 驱动不支持 WSL → 重装最新驱动（支持 WSL 的版本）")
    print("  3. 当前是 WSL1 → 需迁移到 WSL2")
    print("  4. 无 NVIDIA GPU 或 GPU 被禁用")

# 6. 附加信息
print("\n[5] 附加环境信息")
try:
    import transformers
    print(f"  transformers 版本: {transformers.__version__}")
except ImportError:
    print("  transformers 未安装")
try:
    import datasets
    print(f"  datasets 版本: {datasets.__version__}")
except ImportError:
    print("  datasets 未安装")
try:
    import accelerate
    print(f"  accelerate 版本: {accelerate.__version__}")
except ImportError:
    print("  accelerate 未安装")

print("\n" + "=" * 60)
print("自检完成，请把完整输出发给我。")
print("=" * 60)
