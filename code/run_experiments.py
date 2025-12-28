#!/usr/bin/env python
"""
Experiment Runner for NMT Project
Runs all ablation studies and records results

Experiments:
1. RNN Ablation:
   - Cell type: GRU vs LSTM
   - Attention: dot vs multiplicative vs additive
   - Teacher forcing ratio: 1.0 vs 0.5 vs 0.0
   
2. Transformer Ablation:
   - Position encoding: absolute vs relative
   - Normalization: LayerNorm vs RMSNorm
   - Model scale: small(d=128,L=2) vs medium(d=256,L=4) vs large(d=512,L=6)
   
3. T5 Fine-tuning
"""
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def run_command(cmd: List[str], exp_name: str) -> Tuple[bool, float]:
    """Run a training command and return success status and elapsed time"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - start_time
        success = result.returncode == 0
        return success, elapsed
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False, time.time() - start_time


def run_rnn_ablation(epochs: int = 15, batch_size: int = 64, use_amp: bool = True):
    """Run RNN ablation experiments"""
    print("\n" + "="*80)
    print("RNN ABLATION EXPERIMENTS")
    print("="*80)
    
    results = []
    
    # Base command
    base_cmd = [
        sys.executable, 'train.py',
        '--model_type', 'rnn',
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', '0.001',
    ]
    
    experiments = [
        # Cell type comparison (with additive attention)
        {'name': 'RNN_GRU_additive', 'args': ['--attention_type', 'additive']},
        {'name': 'RNN_LSTM_additive', 'args': ['--attention_type', 'additive'], 
         'config_override': {'cell_type': 'LSTM'}},
        
        # Attention mechanism comparison (with GRU)
        {'name': 'RNN_GRU_dot', 'args': ['--attention_type', 'dot']},
        {'name': 'RNN_GRU_multiplicative', 'args': ['--attention_type', 'multiplicative']},
        
        # Teacher forcing ratio comparison
        {'name': 'RNN_GRU_tf1.0', 'args': ['--attention_type', 'additive', '--teacher_forcing_ratio', '1.0']},
        {'name': 'RNN_GRU_tf0.5', 'args': ['--attention_type', 'additive', '--teacher_forcing_ratio', '0.5']},
        {'name': 'RNN_GRU_tf0.0', 'args': ['--attention_type', 'additive', '--teacher_forcing_ratio', '0.0']},
    ]
    
    for exp in experiments:
        cmd = base_cmd.copy()
        cmd.extend(['--exp_name', exp['name']])
        cmd.extend(exp['args'])
        
        success, elapsed = run_command(cmd, exp['name'])
        
        # Load results if available
        log_path = os.path.join(config.LOG_DIR, f"{exp['name']}_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                history = json.load(f)
                best_bleu = max([h['val_bleu'] for h in history])
                results.append({
                    'name': exp['name'],
                    'best_bleu': best_bleu,
                    'elapsed_time': elapsed,
                    'success': success
                })
    
    # Save summary
    summary_path = os.path.join(config.LOG_DIR, 'rnn_ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("RNN Ablation Results Summary")
    print("="*80)
    for r in results:
        print(f"{r['name']}: BLEU={r['best_bleu']:.2f}, Time={r['elapsed_time']:.1f}s")
    
    return results


def run_transformer_ablation(epochs: int = 15, batch_size: int = 64, use_amp: bool = True):
    """Run Transformer ablation experiments"""
    print("\n" + "="*80)
    print("TRANSFORMER ABLATION EXPERIMENTS")
    print("="*80)
    
    results = []
    
    base_cmd = [
        sys.executable, 'train.py',
        '--model_type', 'transformer',
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', '0.0001',
    ]
    
    experiments = [
        # Position encoding comparison
        {'name': 'Transformer_absolute_layernorm', 
         'args': ['--position_encoding', 'absolute', '--norm_type', 'layernorm']},
        {'name': 'Transformer_relative_layernorm', 
         'args': ['--position_encoding', 'relative', '--norm_type', 'layernorm']},
        
        # Normalization comparison
        {'name': 'Transformer_absolute_rmsnorm', 
         'args': ['--position_encoding', 'absolute', '--norm_type', 'rmsnorm']},
        
        # Model scale comparison
        {'name': 'Transformer_small', 
         'args': ['--position_encoding', 'absolute', '--norm_type', 'layernorm',
                  '--d_model', '128', '--num_encoder_layers', '2', '--num_decoder_layers', '2',
                  '--nhead', '4', '--dim_feedforward', '512']},
        {'name': 'Transformer_medium', 
         'args': ['--position_encoding', 'absolute', '--norm_type', 'layernorm',
                  '--d_model', '256', '--num_encoder_layers', '4', '--num_decoder_layers', '4',
                  '--nhead', '8', '--dim_feedforward', '1024']},
        {'name': 'Transformer_large', 
         'args': ['--position_encoding', 'absolute', '--norm_type', 'layernorm',
                  '--d_model', '512', '--num_encoder_layers', '6', '--num_decoder_layers', '6',
                  '--nhead', '8', '--dim_feedforward', '2048']},
    ]
    
    for exp in experiments:
        cmd = base_cmd.copy()
        cmd.extend(['--exp_name', exp['name']])
        cmd.extend(exp['args'])
        
        success, elapsed = run_command(cmd, exp['name'])
        
        log_path = os.path.join(config.LOG_DIR, f"{exp['name']}_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                history = json.load(f)
                best_bleu = max([h['val_bleu'] for h in history])
                results.append({
                    'name': exp['name'],
                    'best_bleu': best_bleu,
                    'elapsed_time': elapsed,
                    'success': success
                })
    
    summary_path = os.path.join(config.LOG_DIR, 'transformer_ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Transformer Ablation Results Summary")
    print("="*80)
    for r in results:
        print(f"{r['name']}: BLEU={r['best_bleu']:.2f}, Time={r['elapsed_time']:.1f}s")
    
    return results


def run_t5_experiment(epochs: int = 5, batch_size: int = 16):
    """Run T5 fine-tuning experiment"""
    print("\n" + "="*80)
    print("T5 FINE-TUNING EXPERIMENT")
    print("="*80)
    
    cmd = [
        sys.executable, 'train.py',
        '--model_type', 't5',
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--exp_name', 'T5_finetune',
    ]
    
    success, elapsed = run_command(cmd, 'T5_finetune')
    
    log_path = os.path.join(config.LOG_DIR, 'T5_finetune_log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            history = json.load(f)
            best_bleu = max([h['val_bleu'] for h in history])
            print(f"\nT5 Fine-tuning: BLEU={best_bleu:.2f}, Time={elapsed:.1f}s")
            return {'name': 'T5_finetune', 'best_bleu': best_bleu, 'elapsed_time': elapsed}
    
    return None


def run_all_experiments():
    """Run all experiments and generate summary"""
    print("\n" + "="*80)
    print(f"Starting Full Experiment Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = {}
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run experiments
    all_results['rnn'] = run_rnn_ablation(epochs=15, batch_size=64)
    all_results['transformer'] = run_transformer_ablation(epochs=15, batch_size=64)
    all_results['t5'] = run_t5_experiment(epochs=5, batch_size=16)
    
    # Save all results
    summary_path = os.path.join(config.LOG_DIR, 'full_experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("FULL EXPERIMENT SUMMARY")
    print("="*80)
    
    # Print final comparison
    print("\nBest Models:")
    if all_results.get('rnn'):
        best_rnn = max(all_results['rnn'], key=lambda x: x['best_bleu'])
        print(f"  RNN: {best_rnn['name']} - BLEU: {best_rnn['best_bleu']:.2f}")
    
    if all_results.get('transformer'):
        best_trans = max(all_results['transformer'], key=lambda x: x['best_bleu'])
        print(f"  Transformer: {best_trans['name']} - BLEU: {best_trans['best_bleu']:.2f}")
    
    if all_results.get('t5'):
        print(f"  T5: BLEU: {all_results['t5']['best_bleu']:.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NMT experiments")
    parser.add_argument('--exp_type', type=str, default='all',
                        choices=['rnn', 'transformer', 't5', 'all'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_amp', action='store_true')
    
    args = parser.parse_args()
    
    if args.exp_type == 'rnn':
        run_rnn_ablation(args.epochs, args.batch_size, args.use_amp)
    elif args.exp_type == 'transformer':
        run_transformer_ablation(args.epochs, args.batch_size, args.use_amp)
    elif args.exp_type == 't5':
        run_t5_experiment()
    else:
        run_all_experiments()
