import pandas as pd
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ================= 配置 =================
INPUT_FILE = 'data/cleaned_antibiotics_for_training.csv'
OUTPUT_FILE = 'data/processed_antibiotics_3d.pt'
# =======================================

def generate_final_data():
    # 修正：必须放在函数的第一行！
    global INPUT_FILE 
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        # 尝试在当前目录下找
        if os.path.exists('cleaned_antibiotics_for_training.csv'):
            print("💡 在当前目录找到了文件，将使用: cleaned_antibiotics_for_training.csv")
            INPUT_FILE = 'cleaned_antibiotics_for_training.csv'
        else:
            return

    df = pd.read_csv(INPUT_FILE)
    data_list = []
    
    print(f"🚀 开始生成最终数据，共 {len(df)} 条...")
    print("⏳ 注意：正常的 3D 生成速度约为每秒 1-2 条，如果跑得太快(>100it/s)说明出错了！")
    
    success_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 移除了 try...except 以便查看报错
        smiles = str(row['SMILES']).strip()
        core_smarts = str(row['core_smarts']).strip()
        
        # 1. 加载分子
        mol = Chem.MolFromSmiles(smiles)
        core = Chem.MolFromSmarts(core_smarts)
        
        if mol is None:
            print(f"❌ 分子加载失败 Row {index}: {smiles}")
            continue
        if core is None:
            print(f"❌ 骨架加载失败 Row {index}: {core_smarts}")
            continue

        # 2. 生成 3D (兼容性写法)
        mol = Chem.AddHs(mol)
        
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        params.maxIterations = 500
        
        res = AllChem.EmbedMolecule(mol, params)
        
        if res != 0:
            # 重试机制
            params.useRandomCoords = True
            params.randomSeed = 0xf00d
            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                continue
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass

        # 3. 匹配骨架
        matches = mol.GetSubstructMatches(core)
        if not matches:
            continue
        
        # 4. 拆分索引
        core_indices = set(matches[0])
        all_indices = set(range(mol.GetNumAtoms()))
        sidechain_indices = all_indices - core_indices
        
        # 5. 提取数据
        conf = mol.GetConformer()
        
        core_pos = []
        core_z = []
        for idx in core_indices:
            pos = conf.GetAtomPosition(idx)
            core_pos.append([pos.x, pos.y, pos.z])
            core_z.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
        
        sidechain_pos = []
        sidechain_z = []
        for idx in sidechain_indices:
            pos = conf.GetAtomPosition(idx)
            sidechain_pos.append([pos.x, pos.y, pos.z])
            sidechain_z.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
        
        label = float(row['XLogP']) if 'XLogP' in row else 0.0

        if len(core_pos) > 0:
            data_list.append({
                'core_pos': torch.tensor(core_pos, dtype=torch.float32),
                'core_z': torch.tensor(core_z, dtype=torch.long),
                'sidechain_pos': torch.tensor(sidechain_pos, dtype=torch.float32),
                'sidechain_z': torch.tensor(sidechain_z, dtype=torch.long),
                'y': torch.tensor([label], dtype=torch.float32)
            })
            success_count += 1

    if success_count > 0:
        torch.save(data_list, OUTPUT_FILE)
        print(f"\n✅ 成功！已保存 {len(data_list)} 条数据到 {OUTPUT_FILE}")
    else:
        print("\n❌ 没有生成数据，请检查报错。")

if __name__ == "__main__":
    generate_final_data()