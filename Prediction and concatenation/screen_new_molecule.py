import torch
import pandas as pd
import io
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from rdkit import RDLogger

# 引用模型
from train_egnn import DualTowerEGNN, collate_fn, DEVICE

# 关闭 RDKit 干扰日志
RDLogger.DisableLog('rdApp.*')

# ================= 1. 定义官能团库 =================
csv_data = """ID,SMILES,Description
0,*O,Hydroxyl
1,*C(=O)O,Carboxyl
2,*N,Amino
3,*S(=O)(=O)O,Sulfonyl
4,*NC(=O),Amide
5,*C(=O)N,Carbamoyl
6,*c1ccccc1,Phenyl/Aryl
7,*C(=O)OC,Ester
8,*C,Methyl
9,*CC(C)C,Isobutyl
10,*c1ccc2ccccc2c1,Naphthyl
11,*c1sccn1,Thiazolyl
12,*[H],Hydrogen
13,*F,Fluoro
14,*Cl,Chloro
15,*Nc1scnc1,Aminothiazole
16,*[n+]1ccccc1,Pyridinium
17,*CON=C,Oxime Group
18,*Oc1c(O)cccc1,Catechol or Hydroxypyridone
19,*C(F)(F)F,Trifluoromethyl
20,*Sc1nnc(=O)[nH]n1C,Cephalosporin Core (Side)
21,*Sc1nnnn1C,Methyl Tetrazole Sulfur
22,*C(C)(C)C,Tert-butyl
23,*OC,Methoxy
24,*C#N,Cyanide"""

# ================= 2. 定义核心骨架 =================
# 你可以随意替换这里！只要带 * 即可。
# 例如：
# 青霉素母核: "CC1(C)S[C@@H]2[C@H](NC(=O)*)C(=O)N2[C@H]1C(=O)O"
# 简单苯环: "c1ccccc1*"
# 四元内酰胺环: "O=C1NCC1*"
CORE_SMILES = "O=C1NCC1*"

# ================= 3. 核心工具函数 =================

def attach_fragment(core_smi, frag_smi):
    """ 
    化学反应拼接 (标准修复版) 
    """
    try:
        core_mol = Chem.MolFromSmiles(core_smi)
        frag_mol = Chem.MolFromSmiles(frag_smi)
        
        if core_mol is None or frag_mol is None: return None
        
        # 标准反应式：
        # [*:1]-[*:2] 表示原子1连着原子2
        # [#0] 表示虚拟原子(Dummy Atom *)
        # 意思：找到连着*的原子1，和连着*的原子2，把*扔掉，把1和2连起来
        rxn = AllChem.ReactionFromSmarts('[*:1]-[#0].[*:2]-[#0]>>[*:1]-[*:2]')
        
        ps = rxn.RunReactants((core_mol, frag_mol))
        
        if not ps: return None

        product = ps[0][0]
        
        # 清洗产物：移除所有原子编号和同位素标记
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)

        try:
            Chem.SanitizeMol(product)
        except:
            return None
            
        return Chem.MolToSmiles(product)
    except Exception as e:
        # print(f"拼接报错: {e}") # 调试用
        return None

def find_core_via_mcs(mol, original_core_smi):
    """ MCS 骨架识别 """
    # 准备模板 (去掉 *)
    clean_core_smi = original_core_smi.replace('*', '')
    core_template = Chem.MolFromSmiles(clean_core_smi)
    
    if not core_template:
        # 兜底：如果模板解析失败，尝试手动创建一个简单的环
        # 这里假设用户用的是内酰胺，如果换了骨架，这一行其实不会触发
        core_template = Chem.MolFromSmarts("C1(=O)NCC1")

    # 计算最大公共子结构
    # ringMatchesRingOnly=True 非常重要，防止侧链的碳链被误认为是骨架环的一部分
    mcs = rdFMCS.FindMCS([mol, core_template], 
                         matchValences=False, 
                         ringMatchesRingOnly=True, 
                         completeRingsOnly=True)
    
    if not mcs.smartsString: return None
        
    mcs_query = Chem.MolFromSmarts(mcs.smartsString)
    if mol.HasSubstructMatch(mcs_query):
        return mol.GetSubstructMatch(mcs_query)
    
    return None

def process_molecule_for_ai(full_smiles):
    """ 生成 3D 并提取特征 """
    mol = Chem.MolFromSmiles(full_smiles)
    if not mol: return None
    mol = Chem.AddHs(mol)
    
    # 1. 生成 3D
    embedded = False
    try:
        params = AllChem.ETKDG()
        params.useRandomCoords = True
        params.maxIterations = 500
        if AllChem.EmbedMolecule(mol, params) == 0:
            embedded = True
    except:
        pass
    
    if not embedded: return None
        
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass

    # 2. 识别骨架 (MCS)
    core_tuple = find_core_via_mcs(mol, CORE_SMILES)
    
    if not core_tuple:
        return None

    core_indices = set(core_tuple)
    all_indices = set(range(mol.GetNumAtoms()))
    side_indices = all_indices - core_indices
    
    # 3. 提取特征
    conf = mol.GetConformer()
    
    core_pos = []
    core_z = []
    for idx in core_indices:
        pos = conf.GetAtomPosition(idx)
        core_pos.append([pos.x, pos.y, pos.z])
        core_z.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
        
    side_pos = []
    side_z = []
    for idx in side_indices:
        pos = conf.GetAtomPosition(idx)
        side_pos.append([pos.x, pos.y, pos.z])
        side_z.append(mol.GetAtomWithIdx(idx).GetAtomicNum())
    
    # 4. 归一化
    if len(core_pos) > 0:
        core_t = torch.tensor(core_pos, dtype=torch.float32)
        center = core_t.mean(dim=0, keepdim=True)
        
        if len(side_pos) == 0:
            side_t = torch.zeros((1, 3), dtype=torch.float32)
            side_z = [1]
        else:
            side_t = torch.tensor(side_pos, dtype=torch.float32) - center

        return {
            'core_pos': core_t - center,
            'core_z': torch.tensor(core_z, dtype=torch.long),
            'sidechain_pos': side_t,
            'sidechain_z': torch.tensor(side_z, dtype=torch.long),
            'y': torch.tensor([0.0])
        }
    return None

# ================= 4. 主程序 =================
def run_screening():
    print("🚀 初始化官能团库...")
    df = pd.read_csv(io.StringIO(csv_data))
    print(f"📦 加载了 {len(df)} 个官能团。")
    print(f"🧬 核心骨架: {CORE_SMILES}")
    
    try:
        model = DualTowerEGNN(hidden_dim=64).to(DEVICE)
        model.load_state_dict(torch.load('3D_model/egnn_screener.pth', map_location=DEVICE))
        model.eval()
        print("🤖 AI 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    results = []
    print("-" * 80)
    print(f"{'ID':<4} | {'Description':<25} | {'Score':<8} | {'Status'}")
    print("-" * 80)

    for index, row in df.iterrows():
        frag_smi = row['SMILES']
        desc = row['Description']
        frag_id = row['ID']
        
        # A. 拼接
        full_smi = attach_fragment(CORE_SMILES, frag_smi)
        if not full_smi:
            print(f"{frag_id:<4} | {desc:<25} | {'---':<8} | ❌ 拼接失败")
            continue
            
        # B. 3D 处理
        data_item = process_molecule_for_ai(full_smi)
        
        if data_item:
            # C. AI 预测
            batch_core, batch_side, _ = collate_fn([data_item])
            with torch.no_grad():
                pred = model(batch_core.to(DEVICE), batch_side.to(DEVICE))
                score = pred.item()
            
            print(f"{frag_id:<4} | {desc:<25} | {score:.4f}   | ✅")
            results.append({'Description': desc, 'SMILES': full_smi, 'Score': score})
        else:
            print(f"{frag_id:<4} | {desc:<25} | {'---':<8} | ⚠️ 3D/匹配失败")
            if '*' in full_smi:
                 print(f"      [DEBUG] 产物含 *: {full_smi}")

    if results:
        print("-" * 80)
        print("\n🏆 推荐的 Top 3 高脂溶性改性方案:")
        results.sort(key=lambda x: x['Score'], reverse=True) 
        for i, res in enumerate(results[:3]):
            print(f"{i+1}. {res['Description']} (Score: {res['Score']:.4f})")
            print(f"   SMILES: {res['SMILES']}")
    else:
        print("\n⚠️ 依然没有结果。")

if __name__ == "__main__":
    run_screening()