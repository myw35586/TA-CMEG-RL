import pandas as pd
import torch
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# 1. 配置
CSV_FILE = 'data/cleaned_antibiotics_for_training.csv' 
# 为了保险，尝试两个路径
try:
    pd.read_csv(CSV_FILE)
except FileNotFoundError:
    CSV_FILE = 'data/cleaned_antibiotics_for_training.csv'

OUTPUT_FILE = 'data/generation_data.pt'

# 2. 读取 CSV
df = pd.read_csv(CSV_FILE)
print(f"📄 读取到 {len(df)} 条原始数据")

data_list = []
all_chars = set()

print("⚗️ 开始处理数据 (生成3D骨架 + 提取侧链序列)...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    full_smi = row['SMILES']
    core_smart = row['core_smarts']
    
    # --- A. 提取侧链字符串 ---
    mol = Chem.MolFromSmiles(full_smi)
    # 注意：这里 core 我们依然用 MolFromSmarts 来做结构匹配
    core_query = Chem.MolFromSmarts(core_smart)
    
    if not mol or not core_query or not mol.HasSubstructMatch(core_query):
        continue
        
    try:
        # 切割得到侧链
        sidechains = Chem.ReplaceCore(mol, core_query)
        if not sidechains: continue
        
        # 得到 SMILES 串
        target_smi = Chem.MolToSmiles(sidechains, isomericSmiles=False)
        # 简化一下：把 [1*] 这种带编号的连接点统一变成 *
        target_smi = re.sub(r'\[\d+\*\]', '*', target_smi)
        
        # 记录字符用于构建词表
        all_chars.update(list(target_smi))
    except:
        continue

    # --- B. 生成骨架的 3D 坐标 (关键修复部分) ---
    
    # 【修复逻辑】
    # 1. 优先尝试用 MolFromSmiles 读取。
    #    因为我们需要生成 3D 坐标，这需要一个"真实"的分子对象，而不仅仅是匹配模式。
    #    你的 core_smarts 列虽然叫 SMARTS，但其实是合法的 SMILES 格式。
    core_mol = Chem.MolFromSmiles(core_smart)
    
    # 2. 如果 SMILES 读取失败（极少数情况），回退到 SMARTS 并手动修复属性
    if not core_mol:
        core_mol = Chem.MolFromSmarts(core_smart)
        if core_mol:
            # 关键修复：手动计算化合价，防止 AddHs 报错
            core_mol.UpdatePropertyCache(strict=False)
    
    if not core_mol: continue
    
    # 3. 补氢，生成 3D
    try:
        core_mol = Chem.AddHs(core_mol) # 现在这里不会报错了
        
        # 生成 3D 构象
        # useRandomCoords=True 可以提高生成成功率
        res = AllChem.EmbedMolecule(core_mol, AllChem.ETKDG())
        if res == -1: # 生成失败
            # 备用方案：使用随机坐标初始化再优化
            params = AllChem.ETKDG()
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(core_mol, params)
            
        if res == -1: continue # 还是失败则跳过
        
        AllChem.MMFFOptimizeMolecule(core_mol)
    except Exception as e:
        # print(f"3D Error: {e}")
        continue 
        
    # 提取坐标和原子序数
    conf = core_mol.GetConformer()
    pos_list = []
    z_list = []
    for i in range(core_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        pos_list.append([pos.x, pos.y, pos.z])
        z_list.append(core_mol.GetAtomWithIdx(i).GetAtomicNum())
        
    # 转 Tensor
    core_pos = torch.tensor(pos_list, dtype=torch.float32)
    core_z = torch.tensor(z_list, dtype=torch.long)
    
    # 归一化：中心化
    if core_pos.shape[0] > 0:
        core_pos = core_pos - core_pos.mean(dim=0, keepdim=True)
    else:
        continue

    # --- C. 存入列表 ---
    data_list.append({
        'core_pos': core_pos,
        'core_z': core_z,
        'target_smi': target_smi
    })

# 3. 构建词表 (Vocab)
# 加入特殊符号: <PAD>, <SOS>, <EOS>
special_tokens = ['<PAD>', '<SOS>', '<EOS>']
chars = sorted(list(all_chars))
vocab = special_tokens + chars

# 映射字典
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for i, c in enumerate(vocab)}

print(f"\n✅ 处理完成！有效数据: {len(data_list)} 条")
print(f"📚 词表大小: {len(vocab)}")
print(f"💾 保存至 {OUTPUT_FILE} ...")

torch.save({
    'data': data_list,
    'vocab': vocab,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
}, OUTPUT_FILE)