#!/usr/bin/env python
"""
Main entry point for NMT project
Chinese-English Neural Machine Translation

This project implements:
1. RNN-based NMT (GRU/LSTM) with attention mechanisms (dot, multiplicative, additive)
2. Transformer-based NMT from scratch (absolute/relative position encoding, LayerNorm/RMSNorm)
3. T5 fine-tuning for translation

Usage:
    python main.py train --model_type transformer ...
    python main.py eval --model_type transformer --checkpoint checkpoints/... ...
    python main.py translate --model_type transformer --checkpoint checkpoints/... --text "..."
    python main.py experiments --exp_type rnn_ablation ...
"""
import sys
import os
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    parser = argparse.ArgumentParser(
        description="Chinese-English Neural Machine Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train Transformer model
    python main.py train --model_type transformer --epochs 20 --use_amp
    
    # Train RNN with specific attention
    python main.py train --model_type rnn --attention_type additive --cell_type GRU
    
    # Run ablation experiments
    python main.py experiments --exp_type rnn_ablation
    python main.py experiments --exp_type transformer_ablation
    
    # Evaluate a model
    python main.py eval --model_type transformer --checkpoint checkpoints/transformer_best.pt
    
    # Translate text
    python main.py translate --model_type transformer --checkpoint checkpoints/transformer_best.pt --text "今天天气很好"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ========== Train command ==========
    train_parser = subparsers.add_parser('train', help='Train a single model')
    train_parser.add_argument('--model_type', type=str, required=True,
                              choices=['rnn', 'transformer', 't5'])
    train_parser.add_argument('--train_size', type=str, default='10k',
                              choices=['10k', '100k'], help='Training data size')
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--learning_rate', type=float, default=1e-4)
    train_parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
    
    # RNN specific
    train_parser.add_argument('--cell_type', type=str, default='GRU', choices=['GRU', 'LSTM'])
    train_parser.add_argument('--attention_type', type=str, default='additive',
                              choices=['dot', 'multiplicative', 'additive'])
    train_parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    
    # Transformer specific
    train_parser.add_argument('--position_encoding', type=str, default='absolute',
                              choices=['absolute', 'relative'])
    train_parser.add_argument('--norm_type', type=str, default='layernorm',
                              choices=['layernorm', 'rmsnorm'])
    train_parser.add_argument('--d_model', type=int, default=256)
    train_parser.add_argument('--nhead', type=int, default=8)
    train_parser.add_argument('--num_layers', type=int, default=4)
    
    # Decoding
    train_parser.add_argument('--decode_method', type=str, default='greedy',
                              choices=['greedy', 'beam'])
    train_parser.add_argument('--beam_size', type=int, default=5)
    
    # Output
    train_parser.add_argument('--exp_name', type=str, default=None,
                              help='Experiment name for saving')
    
    # ========== Experiments command ==========
    exp_parser = subparsers.add_parser('experiments', help='Run ablation experiments')
    exp_parser.add_argument('--exp_type', type=str, required=True,
                            choices=['rnn_ablation', 'transformer_ablation', 't5', 'all'])
    exp_parser.add_argument('--train_size', type=str, default='10k',
                            choices=['10k', '100k'])
    exp_parser.add_argument('--epochs', type=int, default=15)
    exp_parser.add_argument('--batch_size', type=int, default=64)
    exp_parser.add_argument('--use_amp', action='store_true')
    
    # ========== Eval command ==========
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--model_type', type=str, required=True,
                             choices=['rnn', 'transformer', 't5'])
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--decode_method', type=str, default='beam',
                             choices=['greedy', 'beam'])
    eval_parser.add_argument('--beam_size', type=int, default=5)
    
    # ========== Translate command ==========
    translate_parser = subparsers.add_parser('translate', help='Translate text')
    translate_parser.add_argument('--model_type', type=str, required=True,
                                  choices=['rnn', 'transformer', 't5'])
    translate_parser.add_argument('--checkpoint', type=str, required=True)
    translate_parser.add_argument('--text', type=str, default=None,
                                  help='Text to translate')
    translate_parser.add_argument('--input_file', type=str, default=None,
                                  help='File with texts to translate')
    translate_parser.add_argument('--decode_method', type=str, default='beam',
                                  choices=['greedy', 'beam'])
    translate_parser.add_argument('--beam_size', type=int, default=5)
    
    # ========== Visualize command ==========
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--log_files', type=str, nargs='+', required=True,
                            help='Log files to visualize')
    viz_parser.add_argument('--labels', type=str, nargs='+', default=None)
    viz_parser.add_argument('--output_dir', type=str, default='figures')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        run_train(args)
    elif args.command == 'experiments':
        run_experiments(args)
    elif args.command == 'eval':
        run_eval(args)
    elif args.command == 'translate':
        run_translate(args)
    elif args.command == 'visualize':
        run_visualize(args)


def run_train(args):
    """Run single model training"""
    import train as train_module
    
    # Build experiment name
    if args.exp_name is None:
        if args.model_type == 'rnn':
            args.exp_name = f"rnn_{args.cell_type}_{args.attention_type}"
        elif args.model_type == 'transformer':
            args.exp_name = f"transformer_{args.position_encoding}_{args.norm_type}"
        else:
            args.exp_name = "t5_finetune"
    
    # Prepare training args
    train_args = [
        '--model_type', args.model_type,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.learning_rate),
        '--exp_name', args.exp_name,
        '--decoding_strategy', args.decode_method,
        '--beam_size', str(args.beam_size),
    ]
    
    if args.model_type == 'rnn':
        train_args.extend([
            '--attention_type', args.attention_type,
            '--teacher_forcing_ratio', str(args.teacher_forcing_ratio),
        ])
    elif args.model_type == 'transformer':
        train_args.extend([
            '--position_encoding', args.position_encoding,
            '--norm_type', args.norm_type,
            '--d_model', str(args.d_model),
            '--nhead', str(args.nhead),
            '--num_encoder_layers', str(args.num_layers),
            '--num_decoder_layers', str(args.num_layers),
        ])
    
    sys.argv = ['train.py'] + train_args
    train_module.main()


def run_experiments(args):
    """Run ablation experiments"""
    from run_experiments import run_rnn_ablation, run_transformer_ablation, run_t5_experiment
    
    print(f"\n{'='*60}")
    print(f"Running {args.exp_type} experiments")
    print(f"{'='*60}\n")
    
    if args.exp_type == 'rnn_ablation' or args.exp_type == 'all':
        run_rnn_ablation(args.epochs, args.batch_size, args.use_amp)
    
    if args.exp_type == 'transformer_ablation' or args.exp_type == 'all':
        run_transformer_ablation(args.epochs, args.batch_size, args.use_amp)
    
    if args.exp_type == 't5' or args.exp_type == 'all':
        run_t5_experiment(epochs=5, batch_size=16)


def run_eval(args):
    """Evaluate a trained model"""
    print(f"Evaluating {args.model_type} model from {args.checkpoint}")
    # Implementation would load model and run evaluation
    print("Evaluation functionality - to be implemented with loaded model")


def run_translate(args):
    """Translate text using a trained model"""
    if args.text:
        print(f"Translating: {args.text}")
    elif args.input_file:
        print(f"Translating from file: {args.input_file}")
    else:
        print("Please provide --text or --input_file")
        return
    # Implementation would load model and run translation
    print("Translation functionality - to be implemented with loaded model")


def run_visualize(args):
    """Generate visualizations"""
    import visualize as viz_module
    viz_module.plot_metrics(args.log_files, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
