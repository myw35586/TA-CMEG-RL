import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit import RDLogger
import os

# 引用模型
from generate_model import ScaffoldGenerator
from run_generation import generate_sidechain

# 关闭 RDKit 的警报
RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. 定义评分函数 =================
def calculate_metrics(mol):
    """
    给分子做全套体检
    """
    if not mol: return None
    
    # A. 基础属性
    mw = Descriptors.MolWt(mol)        # 分子量
    logp = Descriptors.MolLogP(mol)    # 脂溶性
    hbd = Descriptors.NumHDonors(mol)  # 氢键供体
    hba = Descriptors.NumHAcceptors(mol) # 氢键受体
    tpsa = Descriptors.TPSA(mol)       # 极性表面积
    
    # B. QED 成药性评分 (0~1)
    try:
        qed_score = QED.qed(mol)
    except:
        qed_score = 0.0
        
    # C. Lipinski Rule of 5 判断
    # 1. 分子量 <= 500
    # 2. LogP <= 5
    # 3. 氢键供体 <= 5
    # 4. 氢键受体 <= 10
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    is_lipinski_pass = (violations <= 1) # 允许违反 1 条
    
    return {
        'MW': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'TPSA': tpsa,
        'QED': qed_score,
        'Lipinski': is_lipinski_pass
    }

def clean_smiles(smi):
    if '.' in smi:
        fragments = smi.split('.')
        smi = max(fragments, key=len)
    return smi

# ================= 2. 主程序：海选与排名 =================
def run_evaluation():
    print("🚀 加载模型...")
    DATA_PATH = '/home/myw/drugvae/3D_model/data/generation_data.pt' # 为了读词表
    MODEL_PATH = '/home/myw/drugvae/3D_model/data/generator.pth'
    
    data_pkg = torch.load(DATA_PATH)
    vocab = data_pkg['vocab']
    char_to_idx = data_pkg['char_to_idx']
    idx_to_char = data_pkg['idx_to_char']
    
    model = ScaffoldGenerator(vocab_size=len(vocab), hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 我们用前 50 个骨架来做大规模筛选
    test_data = data_pkg['data'][:50]
    
    results = []
    print(f"🧪 开始大规模海选 (筛选 {len(test_data)} 个骨架)...")
    
    for i, item in enumerate(test_data):
        core_pos = item['core_pos']
        core_z = item['core_z']
        
        # 每个骨架尝试生成 5 次，取最好的那个
        best_qed_for_this_core = -1
        best_mol_info = None
        
        for attempt in range(5):
            raw_smi = generate_sidechain(model, core_pos, core_z, idx_to_char, char_to_idx, 
                                         max_len=80, temperature=0.7)
            clean_smi = clean_smiles(raw_smi)
            
            # 补全连接点
            if not clean_smi.startswith('*'): display_smi = '*' + clean_smi
            else: display_smi = clean_smi
            
            mol = Chem.MolFromSmiles(display_smi)
            if mol:
                metrics = calculate_metrics(mol)
                if metrics:
                    # 如果这个生成的分子 QED 更高，就暂存它
                    if metrics['QED'] > best_qed_for_this_core:
                        best_qed_for_this_core = metrics['QED']
                        metrics['SMILES'] = display_smi
                        metrics['Case_ID'] = i
                        best_mol_info = metrics
        
        # 如果这个骨架生成出了有效分子，加入总榜单
        if best_mol_info:
            results.append(best_mol_info)
            # 简单的进度打印
            if (i+1) % 10 == 0:
                print(f"   ...已处理 {i+1} 个骨架")

    # ================= 3. 生成排行榜 =================
    if not results:
        print("⚠️ 没有生成有效分子。")
        return

    df = pd.DataFrame(results)
    
    # 按 QED 分数从高到低排序
    df_sorted = df.sort_values(by='QED', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("🏆 AI 药物生成排行榜 (Top 10 By QED)")
    print("="*80)
    
    # 打印漂亮的表格
    columns_to_show = ['Case_ID', 'SMILES', 'QED', 'LogP', 'MW', 'Lipinski']
    print(df_sorted[columns_to_show].head(10).to_string(index=False))
    
    # 保存结果
    df_sorted.to_csv('/home/myw/drugvae/3D_model/drug_candidates_ranked.csv', index=False)
    print(f"\n💾 完整榜单已保存至: 3D_model/drug_candidates_ranked.csv")
    
    # 统计数据
    avg_qed = df['QED'].mean()
    pass_rate = df['Lipinski'].mean() * 100
    print("-" * 50)
    print(f"📊 统计概览:")
    print(f"   平均 QED 得分: {avg_qed:.4f} (越接近1越好)")
    print(f"   五规则通过率: {pass_rate:.1f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_evaluation()
