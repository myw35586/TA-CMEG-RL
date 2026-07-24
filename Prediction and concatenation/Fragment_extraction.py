import pandas as pd
import torch
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# ================= 配置路径 =================
INPUT_FILE = 'data/cleaned_antibiotics_for_training.csv'
OUTPUT_FILE = 'data/processed_antibiotics_3d.pt'
# ===========================================

def debug_generate_3d_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"📊 数据预览 (前 2 行):")
    print(df[['SMILES', 'core_smarts']].head(2))
    print("-" * 40)
    
    data_list = []
    success_count = 0
    
    # 错误计数器
    errors = {
        "mol_load_fail": 0,
        "core_load_fail": 0,
        "match_fail_before_Hs": 0,  # 加氢前就匹配不上
        "embed_fail": 0,
        "match_fail_after_Hs": 0,   # 加氢后匹配不上
        "no_sidechain": 0
    }

    print(f"🚀 开始诊断处理 {len(df)} 条数据...")

    # 我们只打印前 10 个详细错误，避免刷屏
    print_limit = 10
    printed_errors = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            smiles = str(row['SMILES']).strip()
            core_smarts = str(row['core_smarts']).strip()
            
            # 1. 检查 SMILES 加载
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                errors["mol_load_fail"] += 1
                continue

            # 2. 检查 SMARTS 加载
            core = Chem.MolFromSmarts(core_smarts)
            if core is None:
                errors["core_load_fail"] += 1
                if printed_errors < print_limit:
                    print(f"\n[Error Row {index}] SMARTS 无效: {core_smarts}")
                    printed_errors += 1
                continue

            # 3. 关键诊断：加氢前匹配测试
            # 很多时候是因为加氢导致匹配失败，先测一下不加氢能不能匹配
            if not mol.HasSubstructMatch(core):
                errors["match_fail_before_Hs"] += 1
                if printed_errors < print_limit:
                    print(f"\n[Error Row {index}] 结构不匹配 (Pre-3D):")
                    print(f"  SMILES: {smiles}")
                    print(f"  SMARTS: {core_smarts}")
                    printed_errors += 1
                continue

            # 4. 加氢与 3D 生成
            mol_3d = Chem.AddHs(mol)
            
            # 生成 3D
            embed_params = AllChem.ETKDG()
            embed_params.useRandomCoords = True # 增加成功率
            embed_params.maxIterations = 500    # 减少迭代次数提速
            
            res = AllChem.EmbedMolecule(mol_3d, embed_params)
            if res != 0:
                errors["embed_fail"] += 1
                continue
                
            # 5. 加氢后匹配与拆分
            matches = mol_3d.GetSubstructMatches(core)
            if not matches:
                # 这种情况极其罕见：加氢前能匹配，加氢后不能
                # 通常是因为 SMARTS 里写死了隐式氢
                errors["match_fail_after_Hs"] += 1
                continue
            
            # 逻辑成功，开始提取数据 (略过具体 tensor 提取，只验证逻辑)
            core_indices = set(matches[0])
            all_indices = set(range(mol_3d.GetNumAtoms()))
            sidechain_indices = all_indices - core_indices
            
            if len(sidechain_indices) == 0:
                errors["no_sidechain"] += 1
                continue

            # 成功！
            # (这里省略 Tensor 封装代码，只为了跑通逻辑)
            success_count += 1

        except Exception as e:
            print(f"未知异常 Row {index}: {e}")
            continue

    print("\n" + "=" * 40)
    print(f"✅ 诊断报告:")
    print(f"总数据量: {len(df)}")
    print(f"成功逻辑跑通: {success_count}")
    print("-" * 20)
    print(f"❌ 失败原因统计:")
    for k, v in errors.items():
        print(f"  {k}: {v}")
    print("=" * 40)

    if errors["match_fail_before_Hs"] > 0:
        print("💡 建议：大部分失败是 'match_fail'。")
        print("   原因：CSV 中的 core_smarts 写法与 RDKit 解析的 SMILES 结构不一致。")
        print("   解决：请尝试使用 Murcko 自动生成的骨架，或者检查 SMARTS 是否包含特殊的原子属性限制。")

if __name__ == "__main__":
    debug_generate_3d_data(INPUT_FILE, OUTPUT_FILE)
