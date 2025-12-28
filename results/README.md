# 中英机器翻译项目提交材料

## 📁 目录结构

```
submission/
├── code/                     # 核心代码
│   ├── config.py             # 配置文件
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 验证评估
│   ├── evaluate_test.py      # 测试集评估
│   ├── main.py               # 主入口
│   ├── inference.py          # 翻译推理
│   ├── visualize.py          # 可视化
│   ├── dataprocess.py        # 数据处理
│   ├── run_experiments.py    # 实验运行器
│   ├── run_all_experiments.sh # 一键运行脚本
│   ├── requirements.txt      # 依赖包
│   └── models/               # 模型定义
│       ├── rnn_nmt.py        # RNN模型
│       ├── transformer_nmt.py # Transformer模型
│       └── t5_nmt.py         # T5模型
│
├── checkpoints/              # 最佳模型检查点
│   ├── transformer_medium_100k_best.pt  # 🏆 最佳模型
│   ├── transformer_medium_best.pt
│   ├── rnn_gru_additive_best.pt
│   └── rnn_best_100k_best.pt
│
├── figures/                  # 实验图表 (30张)
│   ├── compare_val_bleu_*.png    # BLEU对比图
│   ├── compare_train_loss_*.png  # 训练损失图
│   └── compare_val_loss_*.png    # 验证损失图
│
└── results/                  # 实验结果
    ├── experiment_report.md      # 📊 完整实验报告
    ├── complete_latex_tables.tex # 📝 LaTeX表格
    └── README.md                 # 本文件
```

## 🏆 最佳结果

| 模型 | 训练数据 | Val BLEU | Test BLEU |
|------|----------|----------|-----------|
| Transformer Medium | 100k | **15.66** | 11.55 |
| Transformer Medium | 10k | 14.78 | **14.51** |
| RNN GRU+Additive | 10k | 4.56 | 7.72 |

## 📊 实验统计

- **总实验数**: 38组
- **RNN实验**: 16组 (cell type, attention, teacher forcing, decoding)
- **Transformer实验**: 17组 (position encoding, normalization, scale)
- **T5实验**: 2组 (10k, 100k)
- **其他实验**: 3组 (超参数调优)

## 🚀 快速复现

```bash
# 1. 安装依赖
pip install -r code/requirements.txt

# 2. 运行单个实验
python code/main.py --model transformer --exp_name test

# 3. 运行所有实验
bash code/run_all_experiments.sh

# 4. 测试集评估
python code/evaluate_test.py

# 5. 翻译推理
python code/inference.py
```

## 📈 关键发现

1. **Transformer >> RNN**: Transformer BLEU约为RNN的3倍
2. **数据量关键**: 100k数据显著优于10k
3. **Position Encoding**: Absolute优于Relative
4. **Attention**: Additive > Dot > Multiplicative
5. **Early Stopping**: patience=3防止过拟合

## 📝 LaTeX使用

直接复制 `complete_latex_tables.tex` 中的表格到论文即可。

需要的LaTeX包:
```latex
\usepackage{booktabs}
\usepackage{multirow}
```

## 💻 实验环境

- GPU: NVIDIA RTX 4090 (24GB)
- Python: 3.11
- PyTorch: 2.9.0+cu128
- CUDA: 12.8
