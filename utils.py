# utils.py

import spacy
from torchtext.datasets import IWSLT2016
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

# --- 1. 定义特殊标记和它们的索引 ---
# UNK: 未知词, PAD: 填充, SOS: 句子开始, EOS: 句子结束
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# --- 2. 加载分词器 ---
def get_tokenizers():
    """加载SpaCy的德语和英语分词器"""
    print("正在加载分词器...")
    try:
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        print("请先运行: python -m spacy download de_core_news_sm 和 en_core_web_sm")
        exit()
    print("分词器加载完毕。")
    return spacy_de, spacy_en

# --- 3. 构建词汇表 ---
def build_vocab(spacy_de, spacy_en):
    """从IWSLT'14数据集构建德语和英语词汇表"""
    print("正在构建词汇表...")
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def yield_tokens(data_iter, tokenizer, language_index):
        for data_sample in data_iter:
            yield tokenizer(data_sample[language_index])

    # 加载训练数据来构建词汇表
    train_iter = IWSLT2016(split='train', language_pair=('de', 'en'))
    vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, tokenize_de, 0),
                                          min_freq=2,
                                          specials=SPECIAL_TOKENS,
                                          special_first=True)
    vocab_de.set_default_index(UNK_IDX)

    train_iter = IWSLT2016(split='train', language_pair=('de', 'en'))
    vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenize_en, 1),
                                          min_freq=2,
                                          specials=SPECIAL_TOKENS,
                                          special_first=True)
    vocab_en.set_default_index(UNK_IDX)
    
    print(f"德语词汇表大小: {len(vocab_de)}")
    print(f"英语词汇表大小: {len(vocab_en)}")
    print("词汇表构建完毕。")
    return vocab_de, vocab_en

# --- 4. 定义自定义数据集类 ---
class TranslationDataset(Dataset):
    """一个用于包装文本数据的自定义PyTorch数据集"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 5. 数据处理和加载器创建函数 ---
def get_dataloaders(config, vocab_de, vocab_en, spacy_de, spacy_en):
    """创建训练、验证、测试数据加载器"""
    print("正在创建数据加载器...")
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def process_data(data_iter, vocab_de, vocab_en, tokenize_de, tokenize_en):
        processed_data = []
        for de_text, en_text in data_iter:
            de_tensor = torch.tensor([SOS_IDX] + [vocab_de[token] for token in tokenize_de(de_text)] + [EOS_IDX], dtype=torch.long)
            en_tensor = torch.tensor([SOS_IDX] + [vocab_en[token] for token in tokenize_en(en_text)] + [EOS_IDX], dtype=torch.long)
            processed_data.append((de_tensor, en_tensor))
        return processed_data

    train_iter, valid_iter, test_iter = IWSLT2016(language_pair=('de', 'en'))
    
    train_data = process_data(train_iter, vocab_de, vocab_en, tokenize_de, tokenize_en)
    valid_data = process_data(valid_iter, vocab_de, vocab_en, tokenize_de, tokenize_en)
    test_data = process_data(test_iter, vocab_de, vocab_en, tokenize_de, tokenize_en)

    train_dataset = TranslationDataset(train_data)
    valid_dataset = TranslationDataset(valid_data)
    test_dataset = TranslationDataset(test_data)

    def collate_fn(batch):
        """
        自定义的collate_fn来处理批次数据，特别是进行填充
        """
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
        
        # 使用pad_sequence来填充批次中的所有句子到相同的长度
        src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_padded, tgt_padded

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print("数据加载器创建完毕。")
    return train_loader, valid_loader, test_loader