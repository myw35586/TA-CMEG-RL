import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# 1. 读取你刚刚生成的 CSV
csv_path = "/home/myw/drugvae/3D_model/data/multi_point_results.csv"
df = pd.read_csv(csv_path)

# 2. 取前 9 个最好的分子
top_mols_df = df.head(9)

mols = []
legends = []

for index, row in top_mols_df.iterrows():
    smi = row['Full_Molecule']
    qed = row['QED']
    mol = Chem.MolFromSmiles(smi)
    
    # 生成 2D 坐标以便画图
    AllChem.Compute2DCoords(mol)
    mols.append(mol)
    
    # 图片下方的文字说明
    legends.append(f"Rank {index+1}\nQED: {qed:.3f}")

# 3. 画网格图
img = Draw.MolsToGridImage(
    mols, 
    molsPerRow=3, 
    subImgSize=(300, 300), 
    legends=legends,
    returnPNG=False
)

# 4. 保存图片
img.save("/home/myw/drugvae/3D_model/data/best_molecules.png")
print("🎉 图片已保存为 3D_model/data/best_molecules.png，快去看看！")