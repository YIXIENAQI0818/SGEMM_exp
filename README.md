# SGEMM — GPU 矩阵乘法渐进优化

> 并行计算课程论文 — 2025 秋 · 第五学期 · 程宣赫 (2023202250)

## 项目概述

本项目实现了 **SGEMM（单精度通用矩阵乘法）** 在 NVIDIA GPU 上的**渐进式优化**，从基础 CUDA 实现出发，逐步引入共享内存、Warp-level 操作，最终到达 **Tensor Core + 流水线执行** 的高度优化版本。

矩阵规模：**4096 × 4096**，迭代 10 次取平均。

## 优化路线

```
基础 CUDA 实现
    ↓
全局内存合并访问优化
    ↓
共享内存分块 (Tiling)
    ↓
Warp Matrix Multiply-Accumulate (WMMA)
    ↓
Tensor Core + 流水线执行 (Pipeline)
```

## 文件结构

```
SGEMM/
├── SGEMM/                        # 源码与脚本
│   ├── main.cu                   #   Stage 1: 朴素 CUDA SGEMM
│   ├── main2.cu                  #   Stage 2: 合并访存优化
│   ├── main_MMP.cu               #   Stage 3: Warp-level 矩阵乘加
│   ├── main_MMP_share.cu         #   Stage 4: 共享内存分块
│   ├── main_MMP_pipeline.cu      #   Stage 5: Tensor Core + 流水线执行
│   ├── Makefile                  #   编译脚本
│   ├── run.sh                    #   批量运行脚本
│   ├── plt.ipynb                 #   性能可视化 (Jupyter Notebook)
│   └── README                    #   原始实验说明
│
├── report/                       # 报告与参考资料
│   ├── 程宣赫2023202250汇报.pptx   #   课程汇报 PPT
│   ├── gpu_sgemm_performance.pdf  #   性能对比图表
│   ├── Progressive Optimization of SGEMM on GPUs ... .pdf  # 参考论文
│   ├── CUDA_matrix_mult.pdf      #   参考资料
│   └── moduleDocument.pdf        #   参考资料
│
└── README.md                     # 本文件
```

## 各阶段技术要点

| Stage | 文件 | 关键技术 | 优化手段 |
|-------|------|---------|---------|
| 1 | `main.cu` | 朴素 CUDA | 基准实现，每线程计算一行 |
| 2 | `main2.cu` | 合并访存 | 线程-数据映射优化，减少全局内存事务 |
| 3 | `main_MMP.cu` | WMMA | 利用 Warp-level 矩阵乘加指令 |
| 4 | `main_MMP_share.cu` | 共享内存 | 片上 SRAM 分块，减少全局内存带宽压力 |
| 5 | `main_MMP_pipeline.cu` | Tensor Core + Pipeline | FP16 计算、异步拷贝、流水线掩盖延迟 |

## 🔧 编译运行

```bash
cd SGEMM

# 编译所有版本
make

# 运行所有版本并输出性能对比
bash run.sh

# 单独编译运行某一版本
nvcc -arch=sm_75 -O2 main_MMP_pipeline.cu -o main_MMP_pipeline -lcublas
./main_MMP_pipeline

# 性能可视化
jupyter notebook plt.ipynb
```

> **注意**: Stage 5 (`main_MMP_pipeline.cu`) 需要 **Compute Capability ≥ 7.5** (Turing 及以上)，使用 Tensor Core 和 FP16。

## 📊 性能对比

详见 `report/gpu_sgemm_performance.pdf` 和 `SGEMM/plt.ipynb`，展示了从朴素实现到 Tensor Core 流水线的 GFLOPS 提升曲线。

## 📚 参考资料

- `report/Progressive Optimization of SGEMM on GPUs ... .pdf` — 核心参考论文
- `report/CUDA_matrix_mult.pdf` — CUDA 矩阵乘法文档
- `report/moduleDocument.pdf` — CUDA 模块文档

## 📄 License

This repository is for educational purposes as part of the Parallel Computing coursework.
