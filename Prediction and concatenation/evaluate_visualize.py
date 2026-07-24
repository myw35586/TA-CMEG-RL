import torch
import matplotlib.pyplot as plt
import numpy as np
from train_egnn import DualTowerEGNN, Antibiotic3DDataset, collate_fn, DEVICE, BATCH_SIZE, DATA_PATH
from torch.utils.data import DataLoader

def evaluate_and_plot():
    # 1. 加载数据和模型
    dataset = Antibiotic3DDataset(DATA_PATH)
    # 只取测试集 (后 20%)
    test_size = int(0.2 * len(dataset))
    _, test_set = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = DualTowerEGNN(hidden_dim=64).to(DEVICE)
    model.load_state_dict(torch.load('3D_model/egnn_screener.pth', map_location=DEVICE))
    model.eval()

    print("🧪 正在测试集上进行推理...")
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for core_batch, side_batch, labels in loader:
            core_batch = core_batch.to(DEVICE)
            side_batch = side_batch.to(DEVICE)
            
            preds = model(core_batch, side_batch).cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            
            y_true.extend(labels)
            y_pred.extend(preds)

    # 2. 画图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', s=10)
    
    # 画对角线 (完美预测线)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True XLogP')
    plt.ylabel('Predicted XLogP')
    plt.title(f'Model Evaluation (MAE: {np.mean(np.abs(np.array(y_true) - np.array(y_pred))):.4f})')
    plt.legend()
    plt.grid(True)
    
    save_path = '3D_model/prediction_scatter.png'
    plt.savefig(save_path)
    print(f"✅ 评估完成！散点图已保存至: {save_path}")

if __name__ == "__main__":
    evaluate_and_plot()