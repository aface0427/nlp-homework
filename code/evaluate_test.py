#!/usr/bin/env python
"""
Test set evaluation script
Evaluates trained models on the test set with proper vocab handling
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
import json
from tqdm import tqdm
import sacrebleu

import config
from data.dataprocess import (
    load_and_process_data, build_vocab, TranslationDataset, collate_fn,
    BOS_IDX, EOS_IDX, PAD_IDX
)
from torch.utils.data import DataLoader


def load_jsonl(path):
    """Load JSONL file"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def evaluate_transformer(model, test_loader, en_vocab, device, show_examples=5):
    """Evaluate Transformer model on test set"""
    model.eval()
    
    hypotheses = []
    references = []
    examples = []
    
    idx2word = {v: k for k, v in en_vocab.items()}
    sos_idx = BOS_IDX
    eos_idx = EOS_IDX
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src = batch['zh_ids'].to(device)  # Chinese is source
            tgt = batch['en_ids'].to(device)  # English is target
            
            # Decode
            output = model.greedy_decode(src, max_len=100, sos_idx=sos_idx, eos_idx=eos_idx)
            
            # Convert to text
            for i in range(src.size(0)):
                # Hypothesis
                pred_tokens = []
                for idx in output[i]:
                    idx = idx.item()
                    if idx == eos_idx:
                        break
                    if idx not in [PAD_IDX, sos_idx]:
                        pred_tokens.append(idx2word.get(idx, '<unk>'))
                hyp = ' '.join(pred_tokens)
                hypotheses.append(hyp)
                
                # Reference
                ref_tokens = []
                for idx in tgt[i]:
                    idx = idx.item()
                    if idx == eos_idx:
                        break
                    if idx not in [PAD_IDX, sos_idx]:
                        ref_tokens.append(idx2word.get(idx, '<unk>'))
                ref = ' '.join(ref_tokens)
                references.append(ref)
                
                # Store examples
                if len(examples) < show_examples:
                    # Get source
                    src_text = batch['zh_raw'][i] if 'zh_raw' in batch else '(source not available)'
                    examples.append({
                        'source': src_text,
                        'reference': ref,
                        'hypothesis': hyp
                    })
    
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a')
    
    return bleu.score, examples, hypotheses, references


def evaluate_rnn(model, test_loader, en_vocab, device, show_examples=5, beam_size=5):
    """Evaluate RNN model on test set"""
    model.eval()
    
    hypotheses = []
    references = []
    examples = []
    
    idx2word = {v: k for k, v in en_vocab.items()}
    sos_idx = BOS_IDX
    eos_idx = EOS_IDX
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            src = batch['zh_ids'].to(device)  # Chinese is source
            tgt = batch['en_ids'].to(device)  # English is target
            src_mask = (src != PAD_IDX).float().to(device)
            
            # Decode with greedy or beam
            if beam_size > 1:
                output = model.beam_search(src, src_mask, beam_size=beam_size, 
                                          max_len=100, sos_idx=sos_idx, eos_idx=eos_idx)
            else:
                output, _ = model.greedy_decode(src, src_mask, max_len=100, 
                                            sos_idx=sos_idx, eos_idx=eos_idx)
            
            # Convert to text
            for i in range(src.size(0)):
                # Hypothesis
                pred_tokens = []
                for idx in output[i]:
                    idx = idx.item()
                    if idx == eos_idx:
                        break
                    if idx not in [PAD_IDX, sos_idx]:
                        pred_tokens.append(idx2word.get(idx, '<unk>'))
                hyp = ' '.join(pred_tokens)
                hypotheses.append(hyp)
                
                # Reference
                ref_tokens = []
                for idx in tgt[i]:
                    idx = idx.item()
                    if idx == eos_idx:
                        break
                    if idx not in [PAD_IDX, sos_idx]:
                        ref_tokens.append(idx2word.get(idx, '<unk>'))
                ref = ' '.join(ref_tokens)
                references.append(ref)
                
                if len(examples) < show_examples:
                    src_text = batch['zh_raw'][i] if 'zh_raw' in batch else '(source not available)'
                    examples.append({
                        'source': src_text,
                        'reference': ref,
                        'hypothesis': hyp
                    })
    
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a')
    
    return bleu.score, examples, hypotheses, references


def main():
    parser = argparse.ArgumentParser(description='Evaluate on test set')
    parser.add_argument('--model_type', type=str, required=True, choices=['rnn', 'transformer', 't5'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--train_data', type=str, default=None, 
                        help='Training data for vocab (default: auto-detect)')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Test data (default: dataset/test.jsonl)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--show_examples', type=int, default=10)
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size (1 for greedy)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Auto-detect train data from checkpoint name
    if args.train_data is None:
        if '100k' in args.checkpoint:
            args.train_data = config.TRAIN_100K_PATH
            print("Detected 100k model, using train_100k.jsonl for vocab")
        else:
            args.train_data = config.TRAIN_10K_PATH
            print("Detected 10k model, using train_10k.jsonl for vocab")
    
    if args.test_data is None:
        args.test_data = config.TEST_PATH
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Check if vocab is included in checkpoint
    if isinstance(checkpoint, dict) and 'zh_vocab' in checkpoint:
        print("Using vocab from checkpoint")
        zh_vocab = checkpoint['zh_vocab']
        en_vocab = checkpoint['en_vocab']
        state_dict = checkpoint['model_state_dict']
        model_info = checkpoint.get('model_info', {})
        print(f"Vocab sizes from checkpoint: zh={len(zh_vocab)}, en={len(en_vocab)}")
        
        # Load test data
        test_data = load_and_process_data(args.test_data, is_test=True)
    else:
        # Fallback: rebuild vocab
        print("No vocab in checkpoint, rebuilding...")
        state_dict = checkpoint if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
        
        # Infer vocab size from checkpoint
        if args.model_type == 'transformer':
            model_src_vocab = state_dict['src_embedding.weight'].shape[0]
            model_tgt_vocab = state_dict['tgt_embedding.weight'].shape[0]
        elif args.model_type == 'rnn':
            model_src_vocab = state_dict['encoder.embedding.weight'].shape[0]
            model_tgt_vocab = state_dict['decoder.embedding.weight'].shape[0]
        else:
            model_src_vocab = model_tgt_vocab = 10000
        
        print(f"Model vocab sizes: src={model_src_vocab}, tgt={model_tgt_vocab}")
        
        # Temporarily override VOCAB_SIZE in dataprocess
        import data.dataprocess as dp
        original_vocab_size = dp.VOCAB_SIZE
        dp.VOCAB_SIZE = model_src_vocab
        
        # Load and process data for vocab
        print(f"Building vocab from: {args.train_data}")
        train_data = load_and_process_data(args.train_data)
        test_data = load_and_process_data(args.test_data, is_test=True)
        
        zh_vocab = build_vocab(train_data, lang='zh')
        en_vocab = build_vocab(train_data, lang='en')
        
        dp.VOCAB_SIZE = original_vocab_size
        model_info = {}
    
    print(f"Vocab sizes: zh={len(zh_vocab)}, en={len(en_vocab)}")
    
    # Create test dataset
    test_dataset = TranslationDataset(test_data, en_vocab, zh_vocab)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, collate_fn=collate_fn)
    
    # Load model based on type
    if args.model_type == 'transformer':
        from models.transformer_nmt import TransformerNMT
        
        # Infer parameters from checkpoint
        src_vocab_size = state_dict['src_embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embedding.weight'].shape[0]
        d_model = state_dict['src_embedding.weight'].shape[1]
        max_seq_len = state_dict['src_pos_encoding.pe'].shape[1] if 'src_pos_encoding.pe' in state_dict else 512
        num_encoder_layers = sum(1 for k in state_dict.keys() if k.startswith('encoder_layers.') and '.self_attn.W_q.weight' in k)
        num_decoder_layers = sum(1 for k in state_dict.keys() if k.startswith('decoder_layers.') and '.self_attn.W_q.weight' in k)
        dim_feedforward = state_dict['encoder_layers.0.ff.linear1.weight'].shape[0]
        
        # Detect position encoding type
        if 'src_pos_encoding.rel_pos_embedding.weight' in state_dict:
            position_encoding = 'relative'
        else:
            position_encoding = 'absolute'
        
        # Detect norm type
        norm_type = 'layernorm'  # default
        
        print(f"Model: d_model={d_model}, layers={num_encoder_layers}, pos={position_encoding}")
        
        model = TransformerNMT(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            norm_type=norm_type,
            padding_idx=0
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        
        bleu, examples, hyps, refs = evaluate_transformer(
            model, test_loader, en_vocab, device, args.show_examples
        )
        
    elif args.model_type == 'rnn':
        from models.rnn_nmt import Seq2SeqRNN
        
        src_vocab_size = state_dict['encoder.embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['decoder.embedding.weight'].shape[0]
        embedding_dim = state_dict['encoder.embedding.weight'].shape[1]
        hidden_dim = state_dict['encoder.rnn.weight_hh_l0'].shape[1]
        
        # Detect cell type
        cell_type = 'LSTM' if 'encoder.rnn.weight_hh_l0_reverse' in state_dict else 'GRU'
        
        # Detect attention type (simplified)
        attention_type = 'additive'
        
        model = Seq2SeqRNN(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.3,
            cell_type=cell_type,
            attention_type=attention_type,
            padding_idx=0
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        
        bleu, examples, hyps, refs = evaluate_rnn(
            model, test_loader, en_vocab, device, args.show_examples, args.beam_size
        )
    
    else:
        print("T5 evaluation not implemented in this script")
        return
    
    # Print results
    print("\n" + "="*70)
    print(f"TEST SET EVALUATION RESULTS")
    print(f"Model: {os.path.basename(args.checkpoint)}")
    print(f"BLEU Score: {bleu:.2f}")
    print("="*70)
    
    if examples:
        print(f"\nTranslation Examples ({len(examples)} samples):")
        print("-"*70)
        for i, ex in enumerate(examples, 1):
            print(f"[{i}] Source: {ex['source'][:70]}..." if len(ex['source']) > 70 else f"[{i}] Source: {ex['source']}")
            print(f"    Reference: {ex['reference'][:70]}..." if len(ex['reference']) > 70 else f"    Reference: {ex['reference']}")
            print(f"    Hypothesis: {ex['hypothesis'][:70]}..." if len(ex['hypothesis']) > 70 else f"    Hypothesis: {ex['hypothesis']}")
            print("-"*70)


if __name__ == "__main__":
    main()
