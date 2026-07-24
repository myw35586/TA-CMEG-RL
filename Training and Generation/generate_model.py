import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
# 确保 train_egnn.py 在同一目录下
from train_egnn import DualTowerEGNN 

class ScaffoldGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, pretrained_egnn_path=None):
        super().__init__()
        
        # 1. 编码器 (Encoder): 骨架理解模块
        self.egnn_full = DualTowerEGNN(hidden_dim=hidden_dim)
        
        # 加载预训练权重
        if pretrained_egnn_path:
            try:
                state = torch.load(pretrained_egnn_path, map_location='cpu')
                self.egnn_full.load_state_dict(state, strict=False)
                print("🤖 成功加载预训练 EGNN 权重 (迁移学习开启)")
            except Exception as e:
                print(f"⚠️ 预训练权重加载失败: {e}")

        # 2. 解码器 (Decoder)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def build_graph_pure_torch(self, pos, batch_idx, threshold=6.0):
        """ 纯 PyTorch 实现建图 """
        diff = pos.unsqueeze(1) - pos.unsqueeze(0) 
        dist_sq = torch.sum(diff**2, dim=-1) 
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
        mask = (dist_sq < threshold**2) & batch_mask
        mask.fill_diagonal_(False)
        return mask.nonzero(as_tuple=False).t()

    def safe_run_layer(self, layer, h, pos, edge_index):
        """
        安全运行 EGNN 层
        """
        # 强制检查输入维度
        if h.dim() == 1: h = h.unsqueeze(-1)
        if pos.dim() == 1: pos = pos.unsqueeze(-1)
        
        # 运行层
        out = layer(h, pos, edge_index=edge_index)
        
        # 提取结果并处理可能的解包错误
        if isinstance(out, (tuple, list)):
            new_h, new_pos = out[0], out[1]
        else:
            new_h, new_pos = out, pos 
            
        return new_h, new_pos

    def encode_core(self, pos, z, batch_idx):
        """ 提取骨架特征 """
        # 初始 Embedding
        h = self.egnn_full.embedding(z)
        
        # 初始建图
        edge_index = self.build_graph_pure_torch(pos, batch_idx)
        
        # ⚠️ 关键修改：只运行前两层 (Layer 1 & 2)
        # Layer 1
        h, pos = self.safe_run_layer(self.egnn_full.core_egnn1, h, pos, edge_index)
        # Layer 2
        h, pos = self.safe_run_layer(self.egnn_full.core_egnn2, h, pos, edge_index)
        
        # (已删除 Layer 3 的调用，因为你的模型没有这一层)
        
        # 确保 h 的维度是 [N, Hidden]
        if h.dim() == 3: h = h.squeeze(0)
        
        # Global Pooling
        h_vec = global_mean_pool(h, batch_idx)
        return h_vec

    def forward(self, core_pos, core_z, core_batch, target_seq):
        """ 前向传播 """
        # 1. Encode
        context = self.encode_core(core_pos, core_z, core_batch) 
        
        # 2. Decoder
        hidden = context.unsqueeze(0) 
        
        # 3. RNN
        embed = self.embedding(target_seq)
        output, _ = self.rnn(embed, hidden)
        
        # 4. Predict
        logits = self.fc_out(output)
        return logits