# run.py

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import math
from conv_s2s.model import ConvS2S

def main():
    # 1. 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    train_config = config['training']

    # 2. 设置设备
    if train_config['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_config['device'])
    
    print(f"Using device: {device}")

    # 3. 初始化模型
    model = ConvS2S(config).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    # 4. 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    # `ignore_index` 用于忽略padding部分的损失
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    # 5. 模拟数据和训练循环
    print("\nStarting training with dummy data...")
    model.train()
    
    for epoch in range(train_config['num_epochs']):
        start_time = time.time()

        # 创建模拟数据批次
        src_tokens = torch.randint(1, model_config['vocab_size_src'], 
                                   (train_config['batch_size'], model_config['max_seq_len']-10)).to(device)
        tgt_tokens = torch.randint(1, model_config['vocab_size_tgt'], 
                                   (train_config['batch_size'], model_config['max_seq_len']-15)).to(device)
        
        # 解码器输入是目标序列的前n-1个词
        decoder_input = tgt_tokens[:, :-1]
        # 目标是目标序列的后n-1个词
        ground_truth = tgt_tokens[:, 1:]

        optimizer.zero_grad()

        # 前向传播
        output_logits = model(src_tokens, decoder_input)

        # 计算损失
        # CrossEntropyLoss 期望 (N, C, ...) 和 (N, ...)
        # output_logits: (B, T, V) -> (B*T, V)
        # ground_truth: (B, T) -> (B*T)
        loss = criterion(output_logits.view(-1, output_logits.size(-1)), ground_truth.view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | Loss: {loss.item():.3f}')

    print("\nTraining finished.")


if __name__ == '__main__':
    main()