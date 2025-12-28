#!/bin/bash
#
# NMT实验一键启动脚本
# 所有实验在后台运行，日志输出到logs目录
#
# 使用方法:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# 查看进度:
#   tail -f logs/experiment_progress.log
#

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活conda环境
source ~/miniconda3/bin/activate py311

# 创建目录
mkdir -p logs checkpoints figures

# 日志文件
PROGRESS_LOG="logs/experiment_progress.log"
RESULTS_LOG="logs/experiment_results.log"

# 清空进度日志
echo "========================================" > "$PROGRESS_LOG"
echo "NMT实验开始时间: $(date)" >> "$PROGRESS_LOG"
echo "========================================" >> "$PROGRESS_LOG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS_LOG"
}

run_experiment() {
    local exp_name=$1
    local cmd=$2
    local log_file="logs/${exp_name}.log"
    
    log "开始实验: $exp_name"
    log "命令: $cmd"
    
    # 运行实验并记录时间
    start_time=$(date +%s)
    
    eval "$cmd" > "$log_file" 2>&1
    exit_code=$?
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log "✓ 完成: $exp_name (耗时: ${duration}秒)"
        
        # 提取BLEU分数
        best_bleu=$(grep -o "Best BLEU: [0-9.]*" "$log_file" | tail -1 | grep -o "[0-9.]*" || echo "N/A")
        echo "$exp_name: BLEU=$best_bleu, 耗时=${duration}秒" >> "$RESULTS_LOG"
    else
        log "✗ 失败: $exp_name (exit code: $exit_code)"
        echo "$exp_name: FAILED" >> "$RESULTS_LOG"
    fi
    
    return $exit_code
}

# ========================================
# 实验列表 - 按优先级排序
# ========================================

log "========== 第1阶段: RNN消融实验 =========="

# 1.1 Cell Type对比
run_experiment "rnn_gru_additive" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --exp_name rnn_gru_additive --batch_size 64 --lr 0.001"

run_experiment "rnn_lstm_additive" \
    "python train.py --model_type rnn --cell_type LSTM --attention_type additive --exp_name rnn_lstm_additive --batch_size 64 --lr 0.001"

# 1.2 注意力机制对比
run_experiment "rnn_gru_dot" \
    "python train.py --model_type rnn --cell_type GRU --attention_type dot --exp_name rnn_gru_dot --batch_size 64 --lr 0.001"

run_experiment "rnn_gru_multiplicative" \
    "python train.py --model_type rnn --cell_type GRU --attention_type multiplicative --exp_name rnn_gru_multiplicative --batch_size 64 --lr 0.001"

# 1.3 Teacher Forcing比率对比
run_experiment "rnn_tf1.0" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --teacher_forcing_ratio 1.0 --exp_name rnn_tf1.0 --batch_size 64"

run_experiment "rnn_tf0.5" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --teacher_forcing_ratio 0.5 --exp_name rnn_tf0.5 --batch_size 64"

run_experiment "rnn_tf0.0" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --teacher_forcing_ratio 0.0 --exp_name rnn_tf0.0 --batch_size 64"

log "========== 第2阶段: Transformer消融实验 =========="

# 2.1 位置编码对比
run_experiment "transformer_absolute_layernorm" \
    "python train.py --model_type transformer --position_encoding absolute --norm_type layernorm --exp_name transformer_absolute_layernorm --batch_size 64 --lr 0.0001"

run_experiment "transformer_relative_layernorm" \
    "python train.py --model_type transformer --position_encoding relative --norm_type layernorm --exp_name transformer_relative_layernorm --batch_size 64 --lr 0.0001"

# 2.2 归一化方法对比
run_experiment "transformer_absolute_rmsnorm" \
    "python train.py --model_type transformer --position_encoding absolute --norm_type rmsnorm --exp_name transformer_absolute_rmsnorm --batch_size 64 --lr 0.0001"

# 2.3 模型规模对比
run_experiment "transformer_small" \
    "python train.py --model_type transformer --position_encoding absolute --norm_type layernorm --d_model 128 --num_encoder_layers 2 --num_decoder_layers 2 --nhead 4 --dim_feedforward 512 --exp_name transformer_small --batch_size 64 --lr 0.0001"

run_experiment "transformer_medium" \
    "python train.py --model_type transformer --position_encoding absolute --norm_type layernorm --d_model 256 --num_encoder_layers 4 --num_decoder_layers 4 --nhead 8 --dim_feedforward 1024 --exp_name transformer_medium --batch_size 64 --lr 0.0001"

run_experiment "transformer_large" \
    "python train.py --model_type transformer --position_encoding absolute --norm_type layernorm --d_model 512 --num_encoder_layers 6 --num_decoder_layers 6 --nhead 8 --dim_feedforward 2048 --exp_name transformer_large --batch_size 32 --lr 0.0001"

log "========== 第3阶段: T5微调实验 =========="

run_experiment "t5_finetune" \
    "python train.py --model_type t5 --exp_name t5_finetune --batch_size 16 --lr 3e-5 --epochs 10"

log "========== 第4阶段: 解码策略对比 =========="

# 使用最佳RNN模型进行解码对比
run_experiment "rnn_decode_greedy" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --decoding_strategy greedy --exp_name rnn_decode_greedy --batch_size 64"

run_experiment "rnn_decode_beam3" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --decoding_strategy beam --beam_size 3 --exp_name rnn_decode_beam3 --batch_size 64"

run_experiment "rnn_decode_beam5" \
    "python train.py --model_type rnn --cell_type GRU --attention_type additive --decoding_strategy beam --beam_size 5 --exp_name rnn_decode_beam5 --batch_size 64"

log "========== 第5阶段: 生成可视化 =========="

# 生成可视化图表
python visualize.py \
    --log_files logs/rnn_gru_additive_log.json logs/rnn_lstm_additive_log.json logs/rnn_gru_dot_log.json logs/rnn_gru_multiplicative_log.json \
    --labels "GRU+Additive" "LSTM+Additive" "GRU+Dot" "GRU+Multiplicative" \
    --output_dir figures >> "$PROGRESS_LOG" 2>&1 || true

python visualize.py \
    --log_files logs/transformer_absolute_layernorm_log.json logs/transformer_relative_layernorm_log.json logs/transformer_absolute_rmsnorm_log.json \
    --labels "Absolute+LayerNorm" "Relative+LayerNorm" "Absolute+RMSNorm" \
    --output_dir figures >> "$PROGRESS_LOG" 2>&1 || true

python visualize.py \
    --log_files logs/transformer_small_log.json logs/transformer_medium_log.json logs/transformer_large_log.json \
    --labels "Small(d=128,L=2)" "Medium(d=256,L=4)" "Large(d=512,L=6)" \
    --output_dir figures >> "$PROGRESS_LOG" 2>&1 || true

log "========================================" 
log "所有实验完成!"
log "========================================"

# 打印结果摘要
echo ""
echo "========================================" 
echo "实验结果摘要:"
echo "========================================" 
cat "$RESULTS_LOG"
echo "========================================" 

log "结果已保存到 $RESULTS_LOG"
log "可视化图表保存到 figures/"
