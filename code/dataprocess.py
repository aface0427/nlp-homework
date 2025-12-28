import json
import re
import jieba
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Constants - 使用config中的路径
TRAIN_PATH = config.TRAIN_100K_PATH
VALID_PATH = config.VALID_PATH
TEST_PATH = config.TEST_PATH

MAX_LEN = 100
MIN_FREQ = 2
VOCAB_SIZE = 15000

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

def clean_text(text):
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    return text.strip()

def tokenize_zh(text):
    # Jieba exact mode
    return list(jieba.cut(text, cut_all=False))

def tokenize_en(text):
    # Simple whitespace and punctuation tokenization
    text = text.lower()
    # Split on whitespace and keep punctuation as separate tokens
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

def load_and_process_data(path, is_test=False):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    
    print(f"Loaded {len(data)} samples from {path}")
    
    processed_data = []
    
    for item in data:
        en_text = clean_text(item['en'])
        zh_text = clean_text(item['zh'])
        
        if not en_text or not zh_text:
            continue
            
        en_tokens = tokenize_en(en_text)
        zh_tokens = tokenize_zh(zh_text)
        
        # Filter by length
        if len(en_tokens) > MAX_LEN or len(zh_tokens) > MAX_LEN:
            # Option: Truncate or Filter. Requirement says "Filter or truncate". 
            # We will filter for training quality, but let's truncate to be safe if we want to keep data.
            # Let's filter for now as it's usually better for alignment unless data is scarce.
            # Actually, let's truncate as per "Filter or truncate". Truncating is easier to implement.
            en_tokens = en_tokens[:MAX_LEN]
            zh_tokens = zh_tokens[:MAX_LEN]
        
        # Filter by ratio (simple heuristic: length difference > 3x)
        # Only apply this for training data, maybe? 
        # Requirement: "Handle incomplete sentence pairs (abnormal length ratio)"
        if not is_test:
            len_en = len(en_tokens)
            len_zh = len(zh_tokens)
            if len_en == 0 or len_zh == 0:
                continue
            ratio = len_en / len_zh
            if ratio > 3 or ratio < 1/3:
                continue
        
        processed_data.append({
            'en_tokens': en_tokens,
            'zh_tokens': zh_tokens,
            'en_raw': item['en'],
            'zh_raw': item['zh']
        })
        
    print(f"Retained {len(processed_data)} samples after cleaning and filtering")
    return processed_data

def build_vocab(data, lang='en'):
    counter = Counter()
    for item in data:
        tokens = item[f'{lang}_tokens']
        counter.update(tokens)
    
    # Filter by frequency
    filtered_words = [word for word, count in counter.items() if count >= MIN_FREQ]
    
    # Sort by frequency (optional but good for reproducibility) and limit size
    # Note: counter.most_common returns (word, count)
    most_common = counter.most_common(VOCAB_SIZE - 4) # Reserve 4 for special tokens
    
    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
        BOS_TOKEN: BOS_IDX,
        EOS_TOKEN: EOS_IDX
    }
    
    for word, _ in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
            
    return vocab

def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

class TranslationDataset(Dataset):
    def __init__(self, data, en_vocab, zh_vocab):
        self.data = data
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        
    def __len__(self):
        return len(self.data)
    
    def text_to_indices(self, tokens, vocab):
        return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        en_tokens = item['en_tokens']
        zh_tokens = item['zh_tokens']
        
        en_ids = [self.en_vocab[BOS_TOKEN]] + self.text_to_indices(en_tokens, self.en_vocab) + [self.en_vocab[EOS_TOKEN]]
        zh_ids = [self.zh_vocab[BOS_TOKEN]] + self.text_to_indices(zh_tokens, self.zh_vocab) + [self.zh_vocab[EOS_TOKEN]]
        
        return {
            'en_ids': torch.tensor(en_ids, dtype=torch.long),
            'zh_ids': torch.tensor(zh_ids, dtype=torch.long),
            'en_raw': item['en_raw'],
            'zh_raw': item['zh_raw']
        }

def collate_fn(batch):
    en_ids = [item['en_ids'] for item in batch]
    zh_ids = [item['zh_ids'] for item in batch]
    en_raw = [item['en_raw'] for item in batch]
    zh_raw = [item['zh_raw'] for item in batch]
    
    en_padded = pad_sequence(en_ids, batch_first=True, padding_value=PAD_IDX)
    zh_padded = pad_sequence(zh_ids, batch_first=True, padding_value=PAD_IDX)
    
    return {
        'en_ids': en_padded,
        'zh_ids': zh_padded,
        'en_raw': en_raw,
        'zh_raw': zh_raw
    }

def main():
    print("Processing Training Data...")
    train_data = load_and_process_data(TRAIN_PATH)
    
    print("Processing Validation Data...")
    valid_data = load_and_process_data(VALID_PATH)
    
    print("Processing Test Data...")
    test_data = load_and_process_data(TEST_PATH, is_test=True)
    
    print("Building Vocabularies...")
    # Build vocab only from training data
    en_vocab = build_vocab(train_data, lang='en')
    zh_vocab = build_vocab(train_data, lang='zh')
    
    print(f"English Vocab Size: {len(en_vocab)}")
    print(f"Chinese Vocab Size: {len(zh_vocab)}")
    
    # Save vocabs
    save_vocab(en_vocab, 'data/en_vocab.json')
    save_vocab(zh_vocab, 'data/zh_vocab.json')
    
    # Create Datasets
    train_dataset = TranslationDataset(train_data, en_vocab, zh_vocab)
    valid_dataset = TranslationDataset(valid_data, en_vocab, zh_vocab)
    test_dataset = TranslationDataset(test_data, en_vocab, zh_vocab)
    
    # Create DataLoaders
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print("Data processing complete. DataLoaders created.")
    
    # Example batch
    for batch in train_loader:
        print("Example Batch Shapes:")
        print("EN:", batch['en_ids'].shape)
        print("ZH:", batch['zh_ids'].shape)
        break

if __name__ == '__main__':
    main()
