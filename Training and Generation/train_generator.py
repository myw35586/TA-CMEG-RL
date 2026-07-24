import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

# 引用刚才定义的模型
from generate_model import ScaffoldGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 数据集定义 ===
class GenDataset(Dataset):
    def __init__(self, pt_file):
        print(f"📦 正在加载数据: {pt_file}")
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"找不到数据文件: {pt_file}，请检查路径！")
            
        data_pkg = torch.load(pt_file)
        self.data = data_pkg['data']
        self.char_to_idx = data_pkg['char_to_idx']
        self.vocab = data_pkg['vocab']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        smi = item['target_smi']
        
        # 给句子加上“开始”和“结束”标记
        # 例如: "C(=O)O" -> "<SOS> C ( = O ) O <EOS>"
        seq = [self.char_to_idx['<SOS>']] + \
              [self.char_to_idx[c] for c in smi] + \
              [self.char_to_idx['<EOS>']]
              
        return {
            'core_pos': item['core_pos'],
            'core_z': item['core_z'],
            'seq': torch.tensor(seq, dtype=torch.long)
        }

# === 2. 打包函数 (处理不同长度的分子) ===
def collate_fn(batch):
    core_pos_list = []
    core_z_list = []
    core_batch_list = []
    seq_list = []
    
    for i, item in enumerate(batch):
        n_atoms = item['core_z'].size(0)
        core_pos_list.append(item['core_pos'])
        core_z_list.append(item['core_z'])
        # 告诉 PyG 哪些原子属于第 i 个分子
        core_batch_list.append(torch.full((n_atoms,), i, dtype=torch.long))
        seq_list.append(item['seq'])
        
    # 拼接成大 Tensor
    batch_core_pos = torch.cat(core_pos_list, dim=0)
    batch_core_z = torch.cat(core_z_list, dim=0)
    batch_core_batch = torch.cat(core_batch_list, dim=0)
    
    # 对序列进行 Padding (补零)，保证长度一致
    batch_seq = pad_sequence(seq_list, batch_first=True, padding_value=0)
    
    return batch_core_pos, batch_core_z, batch_core_batch, batch_seq

# === 3. 训练主流程 ===
def train():
    # A. 配置路径
    DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt'
    # 之前训练好的 EGNN 筛选器权重
    PRETRAINED_PATH = '/home/myw/drugvae/3D_model/3D_model/egnn_screener.pth' 
    # 生成模型保存位置
    SAVE_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'

    # 如果路径不对，尝试自动修正 (兼容性处理)
    if not os.path.exists(DATA_PATH) and os.path.exists('data/generation_data.pt'):
        DATA_PATH = 'data/generation_data.pt'

    # B. 加载数据
    try:
        dataset = GenDataset(DATA_PATH)
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print(f"✅ 数据加载成功! 样本数: {len(dataset)}, 词表大小: {len(dataset.vocab)}")
    
    # C. 初始化模型
    model = ScaffoldGenerator(vocab_size=len(dataset.vocab), 
                              hidden_dim=64, 
                              pretrained_egnn_path=PRETRAINED_PATH).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 计算 Loss 时忽略 <PAD> (index=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    print(f"🚀 开始在 {DEVICE} 上训练生成模型...")
    model.train()
    
    # D. 训练循环
    EPOCHS = 51
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (pos, z, core_batch, seq) in enumerate(loader):
            pos, z, core_batch, seq = pos.to(DEVICE), z.to(DEVICE), core_batch.to(DEVICE), seq.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Teacher Forcing 策略
            input_seq = seq[:, :-1]
            target_seq = seq[:, 1:]
            
            logits = model(pos, z, core_batch, input_seq)
            
            # 计算 Loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch:<3} | Loss: {avg_loss:.4f}")
            
    # E. 保存
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 训练完成！生成模型已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    train()