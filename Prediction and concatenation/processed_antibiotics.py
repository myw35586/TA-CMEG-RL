import pandas as pd
import torch
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ================= 配置路径 =================
INPUT_FILE = 'data/cleaned_antibiotics_for_training.csv'
OUTPUT_FILE = 'data/processed_antibiotics_3d.pt'
# ===========================================

def generate_3d_data(input_path, output_path):
    # 1. 检查文件
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    print(f"正在读取数据: {input_path} ...")
    df = pd.read_csv(input_path)
    
    data_list = []
    success_count = 0
    fail_count = 0
    
    print(f"开始处理 {len(df)} 个分子 (生成 3D 构象 + 拆分骨架)...")

    # 使用 tqdm 显示进度条
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            smiles = row['SMILES']
            core_smarts = row['core_smarts']
            # 这里默认使用 XLogP 作为训练的目标标签，如果没有这一列，请改为其他数值列
            label = float(row['XLogP']) 

            # A. 构建分子对象
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol) # 3D 必须加氢
            
            core = Chem.MolFromSmarts(core_smarts)
            
            if mol is None or core is None:
                fail_count += 1
                continue

            # B. 生成 3D 构象 (Embedding)
            # 使用 ETKDG 算法生成 3D 坐标
            embed_res = AllChem.EmbedMolecule(mol, AllChem.ETKDG(randomSeed=42))
            if embed_res != 0:
                # 如果生成失败，尝试使用随机坐标再优化 (备用方案)
                embed_res = AllChem.EmbedMolecule(mol, AllChem.ETKDG(useRandomCoords=True))
                if embed_res != 0:
                    fail_count += 1
                    continue
            
            # 力场优化，让结构更合理
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass # 如果力场优化失败，勉强用未优化的

            # C. 拆分骨架与侧链
            # 找到骨架在分子中的原子索引
            matches = mol.GetSubstructMatches(core)
            if not matches:
                fail_count += 1
                continue
            
            # 取第一个匹配到的骨架 (通常只有一个)
            core_indices = set(matches[0])
            all_indices = set(range(mol.GetNumAtoms()))
            sidechain_indices = all_indices - core_indices
            
            # 如果没有侧链 (全是骨架)，对于连接任务来说可能意义不大，但也可以保留
            if len(sidechain_indices) == 0:
                # 视情况过滤，这里先跳过纯骨架分子
                fail_count += 1 
                continue

            # D. 提取 PyTorch Tensor 数据
            conf = mol.GetConformer()
            
            # --- 提取骨架数据 (Context) ---
            core_pos = []
            core_atom_nums = []
            for idx in core_indices:
                pos = conf.GetAtomPosition(idx)
                core_pos.append([pos.x, pos.y, pos.z])
                core_atom_nums.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
            
            # --- 提取侧链数据 (Fragment) ---
            sidechain_pos = []
            sidechain_atom_nums = []
            for idx in sidechain_indices:
                pos = conf.GetAtomPosition(idx)
                sidechain_pos.append([pos.x, pos.y, pos.z])
                sidechain_atom_nums.append(mol.GetAtomWithIdx(idx).GetAtomicNum())

            # E. 封装成字典
            data_item = {
                'smiles': smiles,
                # 骨架特征
                'core_pos': torch.tensor(core_pos, dtype=torch.float32),
                'core_z': torch.tensor(core_atom_nums, dtype=torch.long), # 原子序数
                # 侧链特征
                'sidechain_pos': torch.tensor(sidechain_pos, dtype=torch.float32),
                'sidechain_z': torch.tensor(sidechain_atom_nums, dtype=torch.long), # 原子序数
                # 标签
                'y': torch.tensor([label], dtype=torch.float32)
            }
            
            data_list.append(data_item)
            success_count += 1

        except Exception as e:
            # print(f"Error: {e}") # 调试时可打开
            fail_count += 1
            continue

    # 4. 保存文件
    print("-" * 30)
    print(f"处理完成！")
    print(f"成功: {success_count} 条")
    print(f"失败/跳过: {fail_count} 条")
    
    if success_count > 0:
        torch.save(data_list, output_path)
        print(f"数据已保存至: {output_path}")
        print("可以直接用于 PyG 模型训练了。")
    else:
        print("没有生成有效数据，请检查输入 CSV。")

if __name__ == "__main__":
    generate_3d_data(INPUT_FILE, OUTPUT_FILE)