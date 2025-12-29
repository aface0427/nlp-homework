# NLP Homework: Chinese-English Neural Machine Translation

本项目实现了中英机器翻译的神经网络模型，包括 RNN、Transformer 和 T5 三种架构。

## 项目结构

```
nlp-homework/
├── code/                       # 源代码目录
│   ├── main.py                 # 主入口 (统一的命令行接口)
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── inference.py            # 推理脚本
│   ├── visualize.py            # 可视化脚本
│   ├── config.py               # 配置文件
│   ├── dataprocess.py          # 数据处理
│   ├── run_experiments.py      # 批量实验脚本
│   ├── run_all_experiments.sh  # 一键运行所有实验
│   ├── data/                   # 处理后的数据
│   ├── dataset/                # 原始数据集
│   │   ├── train_10k.jsonl     # 10k训练集
│   │   ├── train_100k.jsonl    # 100k训练集
│   │   ├── valid.jsonl         # 验证集
│   │   └── test.jsonl          # 测试集
│   └── models/                 # 模型实现
│       ├── rnn_nmt.py          # RNN Seq2Seq + Attention
│       ├── transformer_nmt.py  # Transformer from scratch
│       └── t5_nmt.py           # T5 微调
├── figures/                    # 实验结果图表
├── results/                    # 实验结果记录
├── report.tex                  # LaTeX实验报告
├── report.pdf                  # PDF实验报告
└── requirements.txt            # 依赖包
```

## 环境配置

### 1. 创建 Python 环境

推荐使用 Python 3.10 或 3.11：

```bash
# 使用 conda 创建虚拟环境
conda create -n nmt python=3.11 -y
conda activate nmt
```

### 2. 安装依赖

```bash
# 进入项目目录
cd nlp-homework

# 安装依赖
pip install -r requirements.txt
```

依赖列表：
- `torch>=2.0.0` - PyTorch 深度学习框架
- `transformers>=4.30.0` - Hugging Face Transformers (用于 T5)
- `sentencepiece>=0.1.99` - 子词分词
- `sacrebleu>=2.3.0` - BLEU 评估指标
- `jieba>=0.42.1` - 中文分词
- `nltk>=3.8.1` - 自然语言处理工具包
- `tqdm>=4.65.0` - 进度条
- `numpy>=1.24.0` - 数值计算
- `matplotlib>=3.7.0` - 可视化
- `seaborn>=0.12.0` - 统计可视化
- `pandas>=2.0.0` - 数据处理

### 3. 下载 NLTK 数据 (可选)

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 快速开始

进入代码目录：

```bash
cd code
```

### 训练模型

#### 1. 训练 RNN 模型

```bash
# 基础 RNN + GRU + Additive Attention
python main.py train --model_type rnn --cell_type GRU --attention_type additive --epochs 20

# 使用 LSTM
python main.py train --model_type rnn --cell_type LSTM --attention_type additive --epochs 20

# 不同注意力机制: dot, multiplicative, additive
python main.py train --model_type rnn --attention_type dot --epochs 20
python main.py train --model_type rnn --attention_type multiplicative --epochs 20
```

#### 2. 训练 Transformer 模型

```bash
# 基础 Transformer (绝对位置编码 + LayerNorm)
python main.py train --model_type transformer --epochs 20 --use_amp

# 相对位置编码
python main.py train --model_type transformer --position_encoding relative --epochs 20

# 使用 RMSNorm
python main.py train --model_type transformer --norm_type rmsnorm --epochs 20

# 自定义模型大小
python main.py train --model_type transformer --d_model 512 --nhead 8 --num_layers 6 --epochs 20
```

#### 3. 微调 T5 模型

```bash
python main.py train --model_type t5 --epochs 5 --batch_size 16
```

#### 4. 使用 100k 大数据集训练

```bash
python main.py train --model_type transformer --train_size 100k --epochs 30 --use_amp
```

### 评估模型

```bash
# 评估已训练的模型
python main.py eval --model_type transformer --checkpoint checkpoints/transformer_best.pt

# 使用 beam search 解码
python main.py eval --model_type transformer --checkpoint checkpoints/transformer_best.pt --decode_method beam --beam_size 5
```

### 翻译文本

```bash
# 使用训练好的模型翻译
python main.py translate --model_type transformer --checkpoint checkpoints/transformer_best.pt --text "今天天气很好"
```

### 运行消融实验

```bash
# RNN 消融实验 (注意力机制、Cell类型、Teacher Forcing比率)
python main.py experiments --exp_type rnn_ablation --epochs 15

# Transformer 消融实验 (位置编码、归一化方式)
python main.py experiments --exp_type transformer_ablation --epochs 15

# T5 微调实验
python main.py experiments --exp_type t5 --epochs 5

# 运行所有实验
python main.py experiments --exp_type all --epochs 15
```

### 一键运行所有实验

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh

# 查看实验进度
tail -f logs/experiment_progress.log
```

## 模型说明

### RNN Seq2Seq

- **编码器**: 2层单向 RNN (GRU/LSTM)
- **解码器**: 2层单向 RNN + Attention
- **注意力机制**: 
  - Dot Product Attention
  - Multiplicative (Luong) Attention  
  - Additive (Bahdanau) Attention
- **训练策略**: Teacher Forcing (可调比率)

### Transformer

- **架构**: 标准 Encoder-Decoder Transformer
- **位置编码**: 绝对位置编码 / 相对位置编码
- **归一化**: LayerNorm / RMSNorm
- **默认配置**: d_model=256, nhead=8, num_layers=4

### T5 Fine-tuning

- **预训练模型**: t5-small (Hugging Face)
- **任务格式**: "translate Chinese to English: <中文句子>"

## 实验结果

实验结果保存在以下位置：

- `code/checkpoints/` - 模型权重
- `code/logs/` - 训练日志
- `code/figures/` - 训练曲线图
- `results/` - 实验报告

## 常见问题

### GPU 内存不足

减小 batch size：

```bash
python main.py train --model_type transformer --batch_size 32 --use_amp
```

### 加速训练

使用混合精度训练：

```bash
python main.py train --model_type transformer --use_amp
```

### T5 下载失败

T5 模型需要从 Hugging Face 下载，确保网络连接正常。如有网络问题，可设置代理或使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 作者

NLP Course Project

## License

MIT License
