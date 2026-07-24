import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem


csv_path = "/home/myw/drugvae/3D_model/data/multi_point_results.csv"
df = pd.read_csv(csv_path)


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
    

    legends.append(f"Rank {index+1}\nQED: {qed:.3f}")


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
