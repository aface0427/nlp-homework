"""
Configuration file for NMT project - 中英机器翻译
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # 原始数据集
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Data paths - 使用本地dataset目录
TRAIN_10K_PATH = os.path.join(DATASET_DIR, "train_10k.jsonl")
TRAIN_100K_PATH = os.path.join(DATASET_DIR, "train_100k.jsonl")
VALID_PATH = os.path.join(DATASET_DIR, "valid.jsonl")
TEST_PATH = os.path.join(DATASET_DIR, "test.jsonl")

# Data config
MAX_SEQ_LEN = 128
MIN_FREQ = 2  # Minimum word frequency for vocabulary

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Model config - RNN
RNN_CONFIG = {
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.3,
    "cell_type": "GRU",  # GRU or LSTM
    "attention_type": "additive",  # dot, multiplicative, additive
    "bidirectional_encoder": False,  # Use unidirectional as per requirement
}

# Model config - Transformer
TRANSFORMER_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "max_seq_len": MAX_SEQ_LEN,
    "position_encoding": "absolute",  # absolute, relative, none
    "norm_type": "layernorm",  # layernorm, rmsnorm
}

# Training config
TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 20,
    "clip_grad": 1.0,
    "teacher_forcing_ratio": 0.5,  # 1.0 = Teacher Forcing, 0.0 = Free Running (Autoregressive)
    "label_smoothing": 0.1,
    "warmup_steps": 500,
    "use_amp": True,  # Automatic Mixed Precision
    "early_stopping_patience": 5,  # 早停机制
}

# Beam search config
BEAM_CONFIG = {
    "beam_size": 3,
    "length_penalty": 0.6,
    "max_decode_len": 100,
}

# T5 fine-tuning config
T5_CONFIG = {
    "model_name": "t5-small",  # 使用Hugging Face在线模型
    "max_input_length": 128,
    "max_target_length": 128,
    "learning_rate": 3e-5,
    "batch_size": 16,
    "epochs": 5,
}
