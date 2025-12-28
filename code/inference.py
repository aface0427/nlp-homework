#!/usr/bin/env python
"""
NMT推理脚本 - 一键翻译
支持RNN、Transformer、T5模型

使用方法:
    # 翻译单句
    python inference.py --model_type transformer --checkpoint checkpoints/transformer_medium_100k_best.pt --text "你好世界"
    
    # 翻译文件
    python inference.py --model_type transformer --checkpoint checkpoints/transformer_medium_100k_best.pt --input test.jsonl --output translations.jsonl
    
    # 在test set上评估BLEU
    python inference.py --model_type transformer --checkpoint checkpoints/transformer_medium_100k_best.pt --input dataset/test.jsonl --evaluate
"""

import argparse
import json
import os
import sys
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 分词函数 (不依赖外部模块)
def tokenize_zh(text):
    """中文分词 - 使用jieba"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 如果jieba不可用，使用简单字符分词
        return list(text)

def tokenize_en(text):
    """英文分词"""
    tokens = re.findall(r"[\w]+|[^\s\w]", text.lower())
    return tokens


def load_jsonl(path):
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_model_and_vocab(model_type, checkpoint_path, device, **kwargs):
    """加载模型和词表（从checkpoint）"""
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 从checkpoint加载vocab（优先使用checkpoint中保存的vocab）
    if 'zh_vocab' in checkpoint and 'en_vocab' in checkpoint:
        zh_vocab = checkpoint['zh_vocab']
        en_vocab = checkpoint['en_vocab']
        print(f"从checkpoint加载词表: zh_vocab={len(zh_vocab)}, en_vocab={len(en_vocab)}")
    else:
        raise ValueError("Checkpoint中没有找到vocab！请使用rebuild_checkpoints.py重建checkpoint。")
    
    # 从checkpoint推断参数
    if model_type == 'transformer':
        # 从embedding层推断vocab size
        src_vocab_size = state_dict['src_embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['tgt_embedding.weight'].shape[0]
        d_model = state_dict['src_embedding.weight'].shape[1]
        
        # 判断position encoding类型
        if 'src_pos_encoding.pe' in state_dict:
            max_seq_len = state_dict['src_pos_encoding.pe'].shape[1]
            position_encoding = 'absolute'
        elif 'src_pos_encoding.rel_pos_embedding.weight' in state_dict:
            rel_pe_size = state_dict['src_pos_encoding.rel_pos_embedding.weight'].shape[0]
            max_seq_len = (rel_pe_size - 1) // 2
            position_encoding = 'relative'
        else:
            max_seq_len = 128
            position_encoding = 'absolute'
        
        # 推断layer数
        num_encoder_layers = sum(1 for k in state_dict.keys() if k.startswith('encoder_layers.') and 'self_attn.W_q.weight' in k)
        num_decoder_layers = sum(1 for k in state_dict.keys() if k.startswith('decoder_layers.') and 'self_attn.W_q.weight' in k)
        
        # 推断nhead和ffn
        if 'encoder_layers.0.self_attn.W_q.weight' in state_dict:
            nhead = state_dict['encoder_layers.0.self_attn.W_q.weight'].shape[0] // (d_model // 8)
            nhead = max(1, min(nhead, 16))  # 合理范围
            if d_model % nhead != 0:
                nhead = 8  # 默认值
        else:
            nhead = kwargs.get('nhead', 8)
            
        if 'encoder_layers.0.ffn.0.weight' in state_dict:
            dim_feedforward = state_dict['encoder_layers.0.ffn.0.weight'].shape[0]
        else:
            dim_feedforward = kwargs.get('dim_feedforward', d_model * 4)
        
        # 判断norm类型
        if 'encoder_layers.0.norm1.weight' in state_dict:
            norm_type = 'layernorm'
        elif 'encoder_layers.0.norm1.scale' in state_dict:
            norm_type = 'rmsnorm'
        else:
            norm_type = 'layernorm'
        
        print(f"推断参数: vocab_size={src_vocab_size}, d_model={d_model}, max_len={max_seq_len}")
        print(f"  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}")
        print(f"  nhead={nhead}, ffn={dim_feedforward}, position_encoding={position_encoding}, norm={norm_type}")
        
        from models.transformer_nmt import TransformerNMT
        
        model = TransformerNMT(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
            norm_type=norm_type,
            padding_idx=0
        )
        
    elif model_type == 'rnn':
        src_vocab_size = state_dict['encoder.embedding.weight'].shape[0]
        tgt_vocab_size = state_dict['decoder.embedding.weight'].shape[0]
        embedding_dim = state_dict['encoder.embedding.weight'].shape[1]
        hidden_dim = state_dict['encoder.rnn.weight_hh_l0'].shape[1]
        
        # 判断cell type - GRU has 3*hidden_dim gates, LSTM has 4*hidden_dim
        weight_shape = state_dict['encoder.rnn.weight_hh_l0'].shape[0]
        cell_type = 'GRU' if weight_shape == hidden_dim * 3 else 'LSTM'
        
        # 判断attention类型
        if 'decoder.attention.W1.weight' in state_dict or 'decoder.attention.W_a.weight' in state_dict:
            attention_type = 'additive'
        elif 'decoder.attention.W.weight' in state_dict:
            attention_type = 'multiplicative'
        else:
            attention_type = 'dot'
        
        print(f"推断参数: vocab_size={src_vocab_size}, embed={embedding_dim}, hidden={hidden_dim}")
        print(f"  cell_type={cell_type}, attention={attention_type}")
        
        from models.rnn_nmt import Seq2SeqRNN
        
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
        
    elif model_type == 't5':
        from models.t5_nmt import T5NMT
        model = T5NMT(model_name='t5-small')
        # T5使用自己的tokenizer
        zh_vocab = None
        en_vocab = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    if model_type != 't5':
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, zh_vocab, en_vocab


def translate_sentence(model, model_type, sentence, zh_vocab, en_vocab, device, max_len=100):
    """翻译单个句子"""
    if model_type == 't5':
        # T5有自己的tokenizer
        with torch.no_grad():
            translation = model.translate(sentence)
        return translation
    
    # 分词
    tokens = tokenize_zh(sentence)
    
    # 转换为索引
    unk_id = zh_vocab.get('<unk>', 1)
    bos_id = zh_vocab.get('<bos>', zh_vocab.get('<sos>', 2))
    eos_id = zh_vocab.get('<eos>', 3)
    
    src_indices = [zh_vocab.get(t, unk_id) for t in tokens]
    src_indices = [bos_id] + src_indices + [eos_id]
    
    # 转为tensor
    src = torch.tensor([src_indices], dtype=torch.long, device=device)
    
    if model_type == 'rnn':
        src_mask = torch.ones_like(src, dtype=torch.float, device=device)
        with torch.no_grad():
            output = model.greedy_decode(src, src_mask, max_len=max_len)
            # greedy_decode返回(decoded, attention_weights)
            if isinstance(output, tuple):
                decoded = output[0]
            else:
                decoded = output
        output_indices = decoded[0].tolist()
    elif model_type == 'transformer':
        with torch.no_grad():
            decoded = model.greedy_decode(src, max_len=max_len)
        output_indices = decoded[0].tolist()
    
    # 转换回文本
    idx_to_word = {v: k for k, v in en_vocab.items()}
    
    eos_id_en = en_vocab.get('<eos>', 3)
    bos_id_en = en_vocab.get('<bos>', en_vocab.get('<sos>', 2))
    pad_id_en = en_vocab.get('<pad>', 0)
    
    result_tokens = []
    for idx in output_indices:
        if idx == eos_id_en:
            break
        if idx == bos_id_en or idx == pad_id_en:
            continue
        word = idx_to_word.get(idx, '<unk>')
        result_tokens.append(word)
    
    return ' '.join(result_tokens)


def evaluate_bleu(model, model_type, test_data, zh_vocab, en_vocab, device, max_len=100):
    """在测试集上计算BLEU"""
    from sacrebleu.metrics import BLEU
    
    hypotheses = []
    references = []
    
    print(f"正在翻译 {len(test_data)} 个句子...")
    
    for item in tqdm(test_data):
        zh_text = item['zh']
        en_ref = item['en']
        
        # 翻译
        translation = translate_sentence(model, model_type, zh_text, zh_vocab, en_vocab, device, max_len=max_len)
        
        hypotheses.append(translation)
        references.append(en_ref)
    
    # 计算BLEU
    bleu = BLEU(tokenize='zh')
    score = bleu.corpus_score(hypotheses, [references])
    
    return score, hypotheses, references


def main():
    parser = argparse.ArgumentParser(description='NMT推理脚本')
    parser.add_argument('--model_type', type=str, choices=['rnn', 'transformer', 't5'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--text', type=str, help='要翻译的中文文本')
    parser.add_argument('--input', type=str, help='输入JSONL文件')
    parser.add_argument('--output', type=str, help='输出JSONL文件')
    parser.add_argument('--evaluate', action='store_true', help='评估BLEU分数')
    parser.add_argument('--show_examples', type=int, default=5, help='显示翻译示例数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_len', type=int, default=100, help='最大生成长度')
    
    args = parser.parse_args()
    
    print(f"使用设备: {args.device}")
    
    # 加载模型和词表
    model, zh_vocab, en_vocab = load_model_and_vocab(
        args.model_type, 
        args.checkpoint, 
        args.device
    )
    
    # 单句翻译
    if args.text:
        translation = translate_sentence(model, args.model_type, args.text, zh_vocab, en_vocab, args.device, max_len=args.max_len)
        print(f"\n{'='*60}")
        print(f"输入: {args.text}")
        print(f"翻译: {translation}")
        print(f"{'='*60}")
    
    # 文件翻译/评估
    if args.input:
        test_data = load_jsonl(args.input)
        
        if args.evaluate:
            score, hypotheses, references = evaluate_bleu(
                model, args.model_type, test_data, zh_vocab, en_vocab, args.device, max_len=args.max_len
            )
            print(f"\n{'='*60}")
            print(f"Test Set BLEU Score: {score.score:.2f}")
            print(f"{'='*60}")
            
            # 显示示例
            if args.show_examples > 0:
                print(f"\n翻译示例 (前{args.show_examples}条):")
                print("-" * 60)
                for i in range(min(args.show_examples, len(test_data))):
                    print(f"[{i+1}] 源文: {test_data[i]['zh']}")
                    print(f"    参考: {references[i]}")
                    print(f"    翻译: {hypotheses[i]}")
                    print()
            
            # 保存结果
            if args.output:
                results = []
                for i, item in enumerate(test_data):
                    results.append({
                        'zh': item['zh'],
                        'en_ref': item['en'],
                        'en_hyp': hypotheses[i]
                    })
                with open(args.output, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                print(f"结果已保存到: {args.output}")
        
        else:
            # 仅翻译
            translations = []
            for item in tqdm(test_data, desc="翻译中"):
                trans = translate_sentence(model, args.model_type, item['zh'], zh_vocab, en_vocab, args.device, max_len=args.max_len)
                translations.append({
                    'zh': item['zh'],
                    'en': trans
                })
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for t in translations:
                        f.write(json.dumps(t, ensure_ascii=False) + '\n')
                print(f"翻译结果已保存到: {args.output}")
    
    # 如果没有指定任何操作，进入交互模式
    if not args.text and not args.input:
        print("\n进入交互翻译模式 (输入 'quit' 退出)")
        print("-" * 60)
        while True:
            try:
                text = input("请输入中文: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                translation = translate_sentence(model, args.model_type, text, zh_vocab, en_vocab, args.device, max_len=args.max_len)
                print(f"翻译: {translation}\n")
            except KeyboardInterrupt:
                break
        print("\n再见!")


if __name__ == '__main__':
    main()
