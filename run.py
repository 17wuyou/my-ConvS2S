# run.py

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
from conv_s2s.model import ConvS2S
from utils import get_tokenizers, build_vocab, get_dataloaders

def train(model, dataloader, optimizer, criterion, clip, device):
    """一个epoch的训练函数"""
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # 解码器输入是 <sos> + target_sentence
        decoder_input = tgt[:, :-1]
        # 目标是 target_sentence + <eos>
        ground_truth = tgt[:, 1:]

        output_logits = model(src, decoder_input)
        
        output_dim = output_logits.shape[-1]
        output_flat = output_logits.reshape(-1, output_dim)
        ground_truth_flat = ground_truth.reshape(-1)
        
        loss = criterion(output_flat, ground_truth_flat)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """评估函数"""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            decoder_input = tgt[:, :-1]
            ground_truth = tgt[:, 1:]
            
            output_logits = model(src, decoder_input)
            
            output_dim = output_logits.shape[-1]
            output_flat = output_logits.reshape(-1, output_dim)
            ground_truth_flat = ground_truth.reshape(-1)
            
            loss = criterion(output_flat, ground_truth_flat)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def main():
    # 1. 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. 数据预处理
    spacy_de, spacy_en = get_tokenizers()
    vocab_de, vocab_en = build_vocab(spacy_de, spacy_en)
    
    # 将动态获取的词汇表大小更新到配置中
    config['model']['vocab_size_src'] = len(vocab_de)
    config['model']['vocab_size_tgt'] = len(vocab_en)
    
    train_loader, valid_loader, test_loader = get_dataloaders(config, vocab_de, vocab_en, spacy_de, spacy_en)

    # 3. 设置设备
    device = torch.device(config['training']['device'] if config['training']['device'] != 'auto' else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n使用设备: {device}")

    # 4. 初始化模型
    model = ConvS2S(config).to(device)
    print(f'模型共有 {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M 个可训练参数')

    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=config['special_tokens']['pad_idx'])

    # 6. 训练循环
    best_valid_loss = float('inf')
    num_epochs = config['training']['num_epochs']
    
    print("\n--- 开始训练 ---")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, config['training']['clip_grad'], device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # 如果当前验证损失是最好的，就保存模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'convs2s-best-model.pt')
        
        print(f'Epoch: {epoch+1:02} | 时间: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\t训练损失: {train_loss:.3f} | 训练 PPL: {math.exp(train_loss):7.3f}')
        print(f'\t验证损失: {valid_loss:.3f} | 验证 PPL: {math.exp(valid_loss):7.3f}')

    print("\n--- 训练结束 ---")
    
    # 7. 在测试集上评估最终模型
    model.load_state_dict(torch.load('convs2s-best-model.pt'))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'| 测试损失: {test_loss:.3f} | 测试 PPL: {math.exp(test_loss):7.3f} |')

if __name__ == '__main__':
    main()