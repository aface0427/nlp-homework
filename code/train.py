import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
import time
import math

import config
from data.dataprocess import load_and_process_data, TranslationDataset, collate_fn, PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX, build_vocab, TRAIN_PATH, VALID_PATH, TEST_PATH
from models.rnn_nmt import create_rnn_model
from models.transformer_nmt import create_transformer_model
from models.t5_nmt import create_t5_model

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        if model.__class__.__name__ == 'T5NMT':
            # T5 Training: Zh -> En
            src_text = batch['zh_raw']
            tgt_text = batch['en_raw']
            loss, _ = model(src_text, tgt_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            epoch_loss += loss.item()
            continue

        src = batch['en_ids'].to(device)
        tgt = batch['zh_ids'].to(device)
        
        # src: [batch, src_len], tgt: [batch, tgt_len]
        # For Transformer, we need to handle masks inside the model or here.
        # The models (RNN and Transformer) implemented handle masks or expect raw indices.
        # RNN expects: src, tgt, src_mask, teacher_forcing_ratio
        # Transformer expects: src, tgt (and generates masks internally)
        
        if isinstance(model, nn.Module) and model.__class__.__name__ == 'Seq2SeqRNN':
            src_mask = (src != PAD_IDX).float()
            output = model(src, tgt, src_mask, teacher_forcing_ratio)
        else:
            # Transformer
            # Transformer forward expects src and tgt (input to decoder)
            # Target for training should be tgt[:, :-1] (input) and tgt[:, 1:] (label)
            tgt_input = tgt[:, :-1]
            output = model(src, tgt_input)
            
        # Output: [batch, seq_len, vocab_size]
        # Target: [batch, seq_len]
        
        # For RNN, output length might match tgt length (including EOS) depending on implementation
        # RNN implementation loops range(1, tgt_len), so it produces outputs for t=1 to end.
        # The RNN forward returns outputs of shape [batch, tgt_len, vocab].
        # However, the loop in RNN forward starts filling from index 1. Index 0 is 0s.
        # So we should ignore index 0 when calculating loss.
        
        if model.__class__.__name__ == 'Seq2SeqRNN':
            # RNN output: [batch, tgt_len, vocab]
            # We compare output[:, 1:] with tgt[:, 1:]
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_out = tgt[:, 1:].reshape(-1)
        else:
            # Transformer output: [batch, tgt_len-1, vocab]
            # We compare with tgt[:, 1:]
            output = output.reshape(-1, output.shape[-1])
            tgt_out = tgt[:, 1:].reshape(-1)
            
        loss = criterion(output, tgt_out)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, tokenizer_vocab, decoding_strategy='greedy', beam_size=3):
    model.eval()
    epoch_loss = 0
    hypotheses = []
    references = []
    
    # Reverse vocab for decoding
    idx2word = {v: k for k, v in tokenizer_vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if model.__class__.__name__ == 'T5NMT':
                src_text = batch['zh_raw']
                tgt_text = batch['en_raw']
                
                # Loss
                loss, _ = model(src_text, tgt_text)
                epoch_loss += loss.item()
                
                # Generate
                generated_ids = model(src_text)
                decoded_preds = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                hypotheses.extend(decoded_preds)
                references.extend(tgt_text)
                continue

            src = batch['en_ids'].to(device)
            tgt = batch['zh_ids'].to(device)
            
            # Calculate Loss
            if model.__class__.__name__ == 'Seq2SeqRNN':
                src_mask = (src != PAD_IDX).float()
                output_loss = model(src, tgt, src_mask, teacher_forcing_ratio=1.0)
                
                output_loss = output_loss[:, 1:].reshape(-1, output_loss.shape[-1])
                tgt_out = tgt[:, 1:].reshape(-1)
            else:
                tgt_input = tgt[:, :-1]
                output_loss = model(src, tgt_input)
                output_loss = output_loss.reshape(-1, output_loss.shape[-1])
                tgt_out = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output_loss, tgt_out)
            epoch_loss += loss.item()
            
            # Generate translations for BLEU
            if model.__class__.__name__ == 'Seq2SeqRNN':
                src_mask = (src != PAD_IDX).float()
                if decoding_strategy == 'beam':
                    decoded_ids = model.beam_search_decode(src, src_mask, beam_size=beam_size, max_len=50)
                else:
                    decoded_ids, _ = model.greedy_decode(src, src_mask, max_len=50)
            else:
                if decoding_strategy == 'beam':
                    decoded_ids = model.beam_search_decode(src, beam_size=beam_size, max_len=50)
                else:
                    decoded_ids = model.greedy_decode(src, max_len=50)
            
            # Convert indices to text
            for i in range(len(decoded_ids)):
                # Hypothesis
                ids = decoded_ids[i].cpu().numpy()
                tokens = []
                for idx in ids:
                    if idx == EOS_IDX:
                        break
                    if idx not in [BOS_IDX, PAD_IDX]:
                        tokens.append(idx2word.get(idx, '<unk>'))
                hypotheses.append("".join(tokens)) # Chinese: join without spaces usually, but let's check
                
                # Reference
                # batch['zh_raw'] contains the raw text
                references.append(batch['zh_raw'][i])
                
    # Calculate BLEU
    if model.__class__.__name__ == 'T5NMT':
        # En target
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a')
    else:
        # Zh target
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='zh')
    
    return epoch_loss / len(dataloader), bleu.score

def main():
    parser = argparse.ArgumentParser()
    
    # Experiment args
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer', 't5'])
    parser.add_argument('--cell_type', type=str, default='GRU', choices=['GRU', 'LSTM'])  # RNN cell type
    parser.add_argument('--attention_type', type=str, default='additive', choices=['dot', 'multiplicative', 'additive'])
    parser.add_argument('--position_encoding', type=str, default='absolute', choices=['absolute', 'relative', 'none'])
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    # Optional Transformer scale overrides for ablations
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--decoding_strategy', type=str, default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=3)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)  # 最大epoch，实际由早停控制
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--early_stopping_patience', type=int, default=3)  # 早停patience
    parser.add_argument('--train_data', type=str, default='10k', choices=['10k', '100k'])  # 数据集选择
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure data exists
    # 选择训练数据集
    if args.train_data == '10k':
        train_path = config.TRAIN_10K_PATH
    else:
        train_path = config.TRAIN_100K_PATH
    
    if not os.path.exists('data/en_vocab.json') or not os.path.exists('data/zh_vocab.json'):
        print("Vocab files not found. Building vocab from training data...")
        print("Loading data to build vocab...")
        train_data = load_and_process_data(train_path)
        en_vocab = build_vocab(train_data, lang='en')
        zh_vocab = build_vocab(train_data, lang='zh')
        with open('data/en_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(en_vocab, f, ensure_ascii=False)
        with open('data/zh_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(zh_vocab, f, ensure_ascii=False)
    else:
        en_vocab = load_vocab('data/en_vocab.json')
        zh_vocab = load_vocab('data/zh_vocab.json')
        
    # Load Datasets
    print("Loading datasets...")
    train_data = load_and_process_data(train_path)
    valid_data = load_and_process_data(config.VALID_PATH)
    # test_data = load_and_process_data(TEST_PATH, is_test=True) # Optional for training script
    
    train_dataset = TranslationDataset(train_data, en_vocab, zh_vocab)
    valid_dataset = TranslationDataset(valid_data, en_vocab, zh_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'rnn':
        model_config = config.RNN_CONFIG.copy()
        model_config['attention_type'] = args.attention_type
        model_config['cell_type'] = args.cell_type  # 添加cell_type
        model = create_rnn_model(len(en_vocab), len(zh_vocab), model_config)
    elif args.model_type == 'transformer':
        model_config = config.TRANSFORMER_CONFIG.copy()
        model_config['position_encoding'] = args.position_encoding
        model_config['norm_type'] = args.norm_type
        # Apply CLI overrides when provided for hyperparameter sensitivity studies
        if args.d_model is not None:
            model_config['d_model'] = args.d_model
        if args.nhead is not None:
            model_config['nhead'] = args.nhead
        if args.num_encoder_layers is not None:
            model_config['num_encoder_layers'] = args.num_encoder_layers
        if args.num_decoder_layers is not None:
            model_config['num_decoder_layers'] = args.num_decoder_layers
        if args.dim_feedforward is not None:
            model_config['dim_feedforward'] = args.dim_feedforward
        model = create_transformer_model(len(en_vocab), len(zh_vocab), model_config)
    elif args.model_type == 't5':
        model_config = config.T5_CONFIG.copy()
        model = create_t5_model(model_config['model_name'], device=device)
        # Default T5 finetuning prefers a smaller LR; if user didn't override, adopt config
        if args.lr == parser.get_default('lr'):
            args.lr = model_config.get('learning_rate', args.lr)
        
    model = model.to(device)
    
    # Optimizer & Criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Training Loop with Early Stopping
    best_bleu = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    history = []
    
    print(f"\n{'='*60}")
    print(f"Starting training: {args.exp_name}")
    print(f"Model: {args.model_type} | Epochs: max {args.epochs} | Early stopping patience: {args.early_stopping_patience}")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            config.TRAIN_CONFIG['clip_grad'], args.teacher_forcing_ratio
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss, val_bleu = evaluate(
            model, valid_loader, criterion, device, zh_vocab, 
            decoding_strategy=args.decoding_strategy, beam_size=args.beam_size
        )
        print(f"Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
        
        # Log metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_bleu': val_bleu
        })
        
        log_path = os.path.join(config.LOG_DIR, f"{args.exp_name}_log.json")
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Save best model and check early stopping
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            patience_counter = 0
            save_path = os.path.join(args.output_dir, f"{args.exp_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✓ New best BLEU: {best_bleu:.2f} - Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
            
            if patience_counter >= args.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best BLEU: {best_bleu:.2f}")
                print(f"{'='*60}")
                break
    
    # Save final summary
    summary = {
        'exp_name': args.exp_name,
        'model_type': args.model_type,
        'best_bleu': best_bleu,
        'total_epochs': len(history),
        'config': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'teacher_forcing_ratio': args.teacher_forcing_ratio if args.model_type == 'rnn' else None,
            'attention_type': args.attention_type if args.model_type == 'rnn' else None,
            'position_encoding': args.position_encoding if args.model_type == 'transformer' else None,
            'norm_type': args.norm_type if args.model_type == 'transformer' else None,
        }
    }
    summary_path = os.path.join(config.LOG_DIR, f"{args.exp_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
            
    print(f"\nTraining complete! Best BLEU: {best_bleu:.2f}")

if __name__ == "__main__":
    main()
