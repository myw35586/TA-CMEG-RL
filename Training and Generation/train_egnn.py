import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

# ✅ 引入原生 DataLoader
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool

# ================= 配置 =================
if os.path.exists('data/processed_antibiotics_3d.pt'):
    DATA_PATH = 'data/processed_antibiotics_3d.pt'
elif os.path.exists('3D_model/processed_antibiotics_3d.pt'):
    DATA_PATH = '3D_model/processed_antibiotics_3d.pt'
else:
    DATA_PATH = 'processed_antibiotics_3d.pt'

BATCH_SIZE = 32  # 稍微调小 Batch
HIDDEN_DIM = 64
LR = 5e-4        # 降低学习率，防止梯度爆炸
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =======================================

# ----------------- 核心修复：更稳健的 KNN -----------------
def simple_knn_graph(x, k, batch):
    edge_index_list = []
    for i in batch.unique():
        mask = (batch == i)
        local_x = x[mask]
        local_ids = torch.where(mask)[0]
        num_nodes = local_x.size(0)
        curr_k = min(k, num_nodes - 1) if num_nodes > 1 else 0
        
        if curr_k > 0:
            dist = torch.cdist(local_x, local_x)
            # 添加一个小 epsilon 防止距离为 0 导致的数值不稳定
            dist = dist + 1e-6 
            _, indices = dist.topk(curr_k + 1, dim=1, largest=False)
            neighbor_indices = indices[:, 1:]
            sources = local_ids.repeat_interleave(curr_k)
            targets = local_ids[neighbor_indices.flatten()]
            edge_index_list.append(torch.stack([sources, targets], dim=0))
            
    if len(edge_index_list) > 0:
        return torch.cat(edge_index_list, dim=1)
    else:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

# ----------------- 自定义 EGNN -----------------
class SimpleEGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels + 1, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        # 限制距离的最大值，防止数值溢出
        dist_sq = (pos_i - pos_j).pow(2).sum(dim=-1, keepdim=True)
        dist_sq = torch.clamp(dist_sq, min=0, max=100) 
        inputs = torch.cat([x_i, x_j, dist_sq], dim=-1)
        return self.message_net(inputs)

    def update(self, aggr_out, x):
        inputs = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(inputs)

# ----------------- 数据集修复：处理 NaN 和坐标归一化 -----------------
class Antibiotic3DDataset(Dataset):
    def __init__(self, pt_file):
        super().__init__()
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"❌ 找不到数据文件: {pt_file}")
        
        print(f"📂 正在加载并清洗数据: {pt_file} ...")
        raw_data = torch.load(pt_file)
        self.data_list = []
        
        nan_count = 0
        
        for item in raw_data:
            # 1. 检查标签是否为 NaN
            label = item['y']
            if torch.isnan(label).any():
                # 策略：如果有 NaN，直接设为 0.0，或者你可以选择跳过
                label = torch.zeros_like(label)
                nan_count += 1
            
            # 2. 坐标归一化 (Centering) - 非常重要！
            # 把分子的质心移动到 (0,0,0)，防止坐标数值过大导致梯度爆炸
            core_center = item['core_pos'].mean(dim=0, keepdim=True)
            item['core_pos'] = item['core_pos'] - core_center
            
            # 侧链也要相对于骨架移动
            # (通常侧链和骨架是同一个坐标系，所以都减去骨架中心比较合理)
            item['sidechain_pos'] = item['sidechain_pos'] - core_center

            item['y'] = label
            self.data_list.append(item)
            
        print(f"✅ 数据清洗完成。修复了 {nan_count} 个 NaN 标签。有效样本: {len(self.data_list)}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    core_data_list = []
    side_data_list = []
    labels = []
    
    for item in batch:
        core_data = Data(pos=item['core_pos'], x=item['core_z'].unsqueeze(1))
        core_data_list.append(core_data)
        side_data = Data(pos=item['sidechain_pos'], x=item['sidechain_z'].unsqueeze(1))
        side_data_list.append(side_data)
        labels.append(item['y'])
    
    core_batch = Batch.from_data_list(core_data_list)
    side_batch = Batch.from_data_list(side_data_list)
    labels = torch.cat(labels)
    return core_batch, side_batch, labels

# ----------------- 模型定义 -----------------
class DualTowerEGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_dim)
        
        self.core_egnn1 = SimpleEGNNConv(hidden_dim, hidden_dim)
        self.core_egnn2 = SimpleEGNNConv(hidden_dim, hidden_dim)
        
        self.side_egnn1 = SimpleEGNNConv(hidden_dim, hidden_dim)
        self.side_egnn2 = SimpleEGNNConv(hidden_dim, hidden_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_tower(self, batch_data, egnn1, egnn2):
        x, pos, batch = batch_data.x, batch_data.pos, batch_data.batch
        h = self.embedding(x.squeeze()) 
        edge_index = simple_knn_graph(pos, k=5, batch=batch)
        
        h = egnn1(h, pos, edge_index)
        h = F.silu(h)
        h = egnn2(h, pos, edge_index)
        return global_mean_pool(h, batch)

    def forward(self, core_batch, side_batch):
        core_vec = self.forward_tower(core_batch, self.core_egnn1, self.core_egnn2)
        side_vec = self.forward_tower(side_batch, self.side_egnn1, self.side_egnn2)
        combined = torch.cat([core_vec, side_vec], dim=1)
        return self.predictor(combined)

# ----------------- 训练流程 -----------------
def train():
    print(f"⚙️  配置: LR={LR}, Batch={BATCH_SIZE}, Device={DEVICE}")
    
    try:
        dataset = Antibiotic3DDataset(DATA_PATH)
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = DualTowerEGNN(HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() 
    
    print("🔥 开始训练...")
    train_losses = []
    test_maes = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for core_batch, side_batch, labels in train_loader:
            core_batch = core_batch.to(DEVICE)
            side_batch = side_batch.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(core_batch, side_batch)
            loss = criterion(preds, labels)
            
            # 🔍 检查 Loss 是否为 NaN
            if torch.isnan(loss):
                print("⚠️ Warning: Loss is NaN! 跳过此 Batch")
                continue

            loss.backward()
            
            # ✅ 关键：梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            mae_sum = 0
            count = 0
            with torch.no_grad():
                for core_batch, side_batch, labels in test_loader:
                    core_batch = core_batch.to(DEVICE)
                    side_batch = side_batch.to(DEVICE)
                    labels = labels.to(DEVICE).unsqueeze(1)
                    preds = model(core_batch, side_batch)
                    mae_sum += torch.abs(preds - labels).sum().item()
                    count += len(labels)
            
            avg_mae = mae_sum / count if count > 0 else 0
            test_maes.append(avg_mae)
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train MSE: {avg_loss:.4f} | Test MAE: {avg_mae:.4f}")

    if not os.path.exists('3D_model'):
        os.makedirs('3D_model')
    torch.save(model.state_dict(), '3D_model/egnn_screener.pth')
    print("✅ 模型已保存")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(range(4, EPOCHS, 5), test_maes, label='Test MAE', marker='o')
    plt.title('EGNN Training Curve')
    plt.legend()
    plt.savefig('3D_model/training_curve.png')

if __name__ == "__main__":
    train()