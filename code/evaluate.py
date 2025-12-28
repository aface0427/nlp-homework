"""
Evaluation utilities for NMT models
Includes BLEU score calculation and model evaluation
"""
import torch
import sacrebleu
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import os

from data.dataprocess import PAD_IDX, BOS_IDX, EOS_IDX


def calculate_bleu(hypotheses: List[str], references: List[str], tokenize: str = 'zh') -> float:
    """
    Calculate BLEU score using sacrebleu
    
    Args:
        hypotheses: List of predicted translations
        references: List of reference translations
        tokenize: Tokenization method ('zh' for Chinese, '13a' for English)
    
    Returns:
        BLEU score
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    return bleu.score


def decode_sequence(ids: torch.Tensor, idx2word: Dict[int, str], 
                    skip_special: bool = True) -> str:
    """
    Decode a sequence of token IDs back to text
    
    Args:
        ids: Token IDs (1D tensor or list)
        idx2word: Index to word mapping
        skip_special: Whether to skip special tokens
    
    Returns:
        Decoded string
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().numpy()
    
    tokens = []
    for idx in ids:
        if idx == EOS_IDX:
            break
        if skip_special and idx in [PAD_IDX, BOS_IDX]:
            continue
        tokens.append(idx2word.get(int(idx), '<unk>'))
    
    # Join without spaces for Chinese
    return "".join(tokens)


def evaluate_model(
    model,
    dataloader,
    criterion,
    device: str,
    vocab: Dict[str, int],
    decoding_strategy: str = 'greedy',
    beam_size: int = 5,
    is_t5: bool = False
) -> Tuple[float, float, List[str], List[str]]:
    """
    Evaluate a model on a dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss criterion
        device: Device to use
        vocab: Target vocabulary (word2idx)
        decoding_strategy: 'greedy' or 'beam'
        beam_size: Beam size for beam search
        is_t5: Whether this is a T5 model
    
    Returns:
        Tuple of (average_loss, bleu_score, hypotheses, references)
    """
    model.eval()
    total_loss = 0
    hypotheses = []
    references = []
    
    # Create reverse vocab
    idx2word = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if is_t5:
                # T5 evaluation
                src_text = batch['zh_raw']
                tgt_text = batch['en_raw']
                
                loss, _ = model(src_text, tgt_text)
                total_loss += loss.item()
                
                generated_ids = model(src_text)
                decoded_preds = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                hypotheses.extend(decoded_preds)
                references.extend(tgt_text)
            else:
                # RNN/Transformer evaluation
                src = batch['en_ids'].to(device)
                tgt = batch['zh_ids'].to(device)
                src_mask = (src != PAD_IDX).float()
                
                # Calculate loss
                if model.__class__.__name__ == 'Seq2SeqRNN':
                    output_loss = model(src, tgt, src_mask, teacher_forcing_ratio=1.0)
                    output_loss = output_loss[:, 1:].reshape(-1, output_loss.shape[-1])
                    tgt_out = tgt[:, 1:].reshape(-1)
                else:
                    tgt_input = tgt[:, :-1]
                    output_loss = model(src, tgt_input)
                    output_loss = output_loss.reshape(-1, output_loss.shape[-1])
                    tgt_out = tgt[:, 1:].reshape(-1)
                
                loss = criterion(output_loss, tgt_out)
                total_loss += loss.item()
                
                # Generate translations
                if model.__class__.__name__ == 'Seq2SeqRNN':
                    if decoding_strategy == 'beam':
                        decoded_ids = model.beam_search_decode(src, src_mask, beam_size=beam_size, max_len=50)
                    else:
                        decoded_ids, _ = model.greedy_decode(src, src_mask, max_len=50)
                else:
                    if decoding_strategy == 'beam':
                        decoded_ids = model.beam_search_decode(src, beam_size=beam_size, max_len=50)
                    else:
                        decoded_ids = model.greedy_decode(src, max_len=50)
                
                # Decode to text
                for i in range(len(decoded_ids)):
                    hyp = decode_sequence(decoded_ids[i], idx2word)
                    hypotheses.append(hyp)
                    references.append(batch['zh_raw'][i])
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate BLEU score
    if is_t5:
        bleu_score = calculate_bleu(hypotheses, references, tokenize='13a')
    else:
        bleu_score = calculate_bleu(hypotheses, references, tokenize='zh')
    
    return avg_loss, bleu_score, hypotheses, references


def save_results(results: Dict, path: str):
    """Save evaluation results to JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results(path: str) -> Dict:
    """Load evaluation results from JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models(results_list: List[Dict], model_names: List[str]) -> None:
    """
    Print comparison table of multiple models
    
    Args:
        results_list: List of result dictionaries
        model_names: List of model names
    """
    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    print(f"{'Model':<30} {'BLEU':>10} {'Val Loss':>12} {'Params':>15}")
    print("-" * 80)
    
    for name, results in zip(model_names, results_list):
        bleu = results.get('best_bleu', 0)
        loss = results.get('best_val_loss', 0)
        params = results.get('num_params', 'N/A')
        print(f"{name:<30} {bleu:>10.2f} {loss:>12.4f} {params:>15}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test the module
    print("Evaluate module loaded successfully.")
    print(f"BLEU of identical sentences: {calculate_bleu(['hello world'], ['hello world'], '13a')}")
