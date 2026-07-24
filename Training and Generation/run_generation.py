import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import os

# 引用模型
from generate_model import ScaffoldGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. 核心生成逻辑 (Autoregressive) =================
def generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, max_len=50, temperature=1.0):
    """
    输入: 3D 骨架数据
    输出: 生成的侧链 SMILES 字符串
    """
    model.eval()
    with torch.no_grad():
        # 1. 构造各种 batch 索引 (因为推理时通常是一个个来，batch_size=1)
        # core_z.size(0) 是原子数量
        core_batch = torch.zeros(core_z.size(0), dtype=torch.long).to(DEVICE)
        core_pos = core_pos.to(DEVICE)
        core_z = core_z.to(DEVICE)
        
        # 2. Encode: AI 观察骨架，提取灵感
        # context: [1, Hidden]
        context = model.encode_core(core_pos, core_z, core_batch)
        
        # 3. 准备开始生成
        # 初始输入是 <SOS>
        current_token = torch.tensor([[char_to_idx['<SOS>']]], dtype=torch.long).to(DEVICE)
        # 初始隐状态是骨架向量
        hidden = context.unsqueeze(0) # [1, 1, Hidden]
        
        generated_indices = []
        
        # 4. 循环生成字符 (直到遇到 <EOS> 或太长)
        for _ in range(max_len):
            # Embedding
            embed = model.embedding(current_token)
            
            # RNN
            output, hidden = model.rnn(embed, hidden)
            
            # 预测下一个字
            logits = model.fc_out(output) # [1, 1, Vocab]
            logits = logits.squeeze(0).squeeze(0)
            
            # --- 采样策略 ---
            if temperature == 0:
                # 贪婪搜索 (每次都选概率最大的)
                next_token_idx = torch.argmax(logits).item()
            else:
                # 随机采样 (增加多样性)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
            
            # 检查是否结束
            if idx_to_char[next_token_idx] == '<EOS>':
                break
                
            generated_indices.append(next_token_idx)
            
            # 把生成的这个字作为下一步的输入
            current_token = torch.tensor([[next_token_idx]], dtype=torch.long).to(DEVICE)
            
    # 解码成字符串
    gen_str = "".join([idx_to_char[idx] for idx in generated_indices])
    return gen_str

# ================= 2. 拼接与验证工具 =================
def attach_generated_sidechain(core_smi, side_smi):
    """
    把生成的侧链拼接到骨架上
    """
    try:
        # 必须确保生成的侧链里有连接点 *
        if '*' not in side_smi:
            # 如果 AI 忘了生成 *，我们强制在开头加一个 (简单的修复策略)
            side_smi = '*' + side_smi
            
        core_mol = Chem.MolFromSmiles(core_smi)
        side_mol = Chem.MolFromSmiles(side_smi)
        
        if not core_mol or not side_mol: return None
        
        # 标准反应式拼接
        rxn = AllChem.ReactionFromSmarts('[*:1]-[#0].[*:2]-[#0]>>[*:1]-[*:2]')
        ps = rxn.RunReactants((core_mol, side_mol))
        
        if ps:
            product = ps[0][0]
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        return None
    except:
        return None

# ================= 3. 主程序 =================
def run():
    print("🚀 加载模型和词表...")
    # 路径配置 (确保和你训练时一致)
    DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt' # 为了读词表
    MODEL_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'
    
    # 加载词表
    data_pkg = torch.load(DATA_PATH)
    vocab = data_pkg['vocab']
    char_to_idx = data_pkg['char_to_idx']
    idx_to_char = data_pkg['idx_to_char']
    
    # 加载模型
    model = ScaffoldGenerator(vocab_size=len(vocab), hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("🧪 准备骨架...")
    # 我们从数据集中随便挑几个骨架来测试
    # 你也可以在这里手写一个新的骨架 SMILES
    test_data = data_pkg['data'][:5] # 取前5个做测试
    
    print("-" * 60)
    print(f"{'Core Skeleton':<30} | {'AI Generated Sidechain':<25}")
    print("-" * 60)
    
    for i, item in enumerate(test_data):
        core_pos = item['core_pos']
        core_z = item['core_z']
        
        # 尝试生成 3 次，看看能不能搞出不同的花样
        print(f"🧬 Case {i+1}:")
        
        for t in range(3):
            # temperature=0.8 表示稍微有点随机性，如果是 0 就是完全固定
            gen_smi = generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, temperature=0.8)
            
            # 尝试拼接（这里我们需要原始骨架SMILES，数据里没存，为了演示我们假设一个）
            # 在实际使用中，你应该知道输入的骨架 SMILES 是什么
            # 这里我们只展示生成的片段
            print(f"   Attempt {t+1}: {gen_smi}")
            
    print("-" * 60)
    print("✅ 生成演示完毕！")

if __name__ == "__main__":
    run()