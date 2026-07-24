import pandas as pd
from rdkit import Chem
from collections import Counter

# 1. 读取数据
df = pd.read_csv('data/cleaned_antibiotics_for_training.csv')
print(f"原始数据: {len(df)} 条")

sidechain_smiles_list = []
vocab = Counter()

print("🧪 正在提取侧链并构建词表...")

for idx, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    core = Chem.MolFromSmarts(row['core_smarts'])
    
    if mol and core and mol.HasSubstructMatch(core):
        try:
            # 使用 ReplaceCore 切掉骨架，得到侧链
            # 结果可能是 "C.CN.O" 这种用点号分隔的多个片段
            # 我们这里简单处理：只取最大的那个片段作为主侧链来生成
            sidechains = Chem.ReplaceCore(mol, core)
            if sidechains:
                smi = Chem.MolToSmiles(sidechains, isomericSmiles=False)
                # 简单清洗：去掉连接标记如 [1*]
                # 实际项目中可能需要保留连接位点信息，这里先简化
                import re
                smi_clean = re.sub(r'\[\d+\*\]', '*', smi) 
                
                sidechain_smiles_list.append(smi_clean)
                vocab.update(list(smi_clean))
        except:
            pass

print(f"✅ 成功提取侧链: {len(sidechain_smiles_list)} 条")
print(f"📚 词表大小: {len(vocab)}")
print(f"🔍 词表预览: {list(vocab.keys())[:10]}...")
print(f"🧪 样本预览: {sidechain_smiles_list[:3]}")

# 保存一下处理好的数据，后面训练要用
# (这里只是简单打印，实际你可以存成 .txt 或 .pt)