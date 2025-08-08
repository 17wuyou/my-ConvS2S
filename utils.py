# utils.py (自包含数据集，不再有任何外部依赖)

import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

# --- 1. 定义特殊标记和它们的索引 ---
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# --- 2. 加载分词器 ---
def get_tokenizers():
    print("正在加载分词器...")
    try:
        import de_core_news_sm
        import en_core_web_sm
        spacy_de = de_core_news_sm.load()
        spacy_en = en_core_web_sm.load()
    except ImportError:
        print("请先确保已运行 'python -m spacy download de_core_news_sm' 和 'en_core_web_sm' 并重启了会话。")
        exit()
    print("分词器加载完毕。")
    return spacy_de, spacy_en

# --- 3. 定义一个微型数据集 ---
# 我们不再从网上下载，而是直接在代码里定义数据
DUMMY_DATA = [
    ("Ein Mann in einem roten Hemd.", "A man in a red shirt."),
    ("Eine Frau isst einen Apfel.", "A woman is eating an apple."),
    ("Zwei Hunde spielen im Park.", "Two dogs are playing in the park."),
    ("Ein Junge liest ein Buch.", "A boy is reading a book."),
    ("Das Mädchen springt.", "The girl is jumping."),
    ("Ein Mann mit Brille.", "A man with glasses."),
    ("Eine Frau mit einem Hut.", "A woman with a hat."),
    ("Ein Hund bellt.", "A dog is barking."),
    ("Eine Katze schläft.", "A cat is sleeping."),
    ("Kinder lachen.", "Children are laughing."),
]
# 为了让训练有意义，我们把数据复制几百次
TRAIN_DATA = DUMMY_DATA * 200
VALID_DATA = DUMMY_DATA
TEST_DATA = DUMMY_DATA


# --- 4. 构建词汇表 ---
def build_vocab(spacy_de, spacy_en):
    print("正在从内置数据构建词汇表...")

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def yield_tokens(data, tokenizer, lang_index):
        for sample in data:
            yield tokenizer(sample[lang_index])

    print("正在构建德语词汇表...")
    vocab_de = build_vocab_from_iterator(yield_tokens(TRAIN_DATA, tokenize_de, 0),
                                          min_freq=1, # 频率设为1，因为数据量小
                                          specials=SPECIAL_TOKENS,
                                          special_first=True)
    vocab_de.set_default_index(UNK_IDX)

    print("正在构建英语词汇表...")
    vocab_en = build_vocab_from_iterator(yield_tokens(TRAIN_DATA, tokenize_en, 1),
                                          min_freq=1,
                                          specials=SPECIAL_TOKENS,
                                          special_first=True)
    vocab_en.set_default_index(UNK_IDX)

    print(f"德语词汇表大小: {len(vocab_de)}")
    print(f"英语词汇表大小: {len(vocab_en)}")
    print("词汇表构建完毕。")
    return vocab_de, vocab_en

# --- 5. 自定义PyTorch数据集类 ---
class PyTorchTranslationDataset(Dataset):
    def __init__(self, data, vocab_de, vocab_en, tokenize_de, tokenize_en):
        self.data = data
        self.vocab_de = vocab_de
        self.vocab_en = vocab_en
        self.tokenize_de = tokenize_de
        self.tokenize_en = tokenize_en

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de_text, en_text = self.data[idx]
        
        de_tensor = torch.tensor([SOS_IDX] + [self.vocab_de[token] for token in self.tokenize_de(de_text)] + [EOS_IDX], dtype=torch.long)
        en_tensor = torch.tensor([SOS_IDX] + [self.vocab_en[token] for token in self.tokenize_en(en_text)] + [EOS_IDX], dtype=torch.long)
        return de_tensor, en_tensor

# --- 6. 数据加载器创建函数 ---
def get_dataloaders(config, vocab_de, vocab_en, spacy_de, spacy_en):
    print("正在创建数据加载器...")
    
    def tokenize_de(text): return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenize_en(text): return [tok.text for tok in spacy_en.tokenizer(text)]

    train_dataset = PyTorchTranslationDataset(TRAIN_DATA, vocab_de, vocab_en, tokenize_de, tokenize_en)
    valid_dataset = PyTorchTranslationDataset(VALID_DATA, vocab_de, vocab_en, tokenize_de, tokenize_en)
    test_dataset = PyTorchTranslationDataset(TEST_DATA, vocab_de, vocab_en, tokenize_de, tokenize_en)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
        
        src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_padded, tgt_padded

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print("数据加载器创建完毕。")
    return train_loader, valid_loader, test_loader