"""
CTNet推理使用示例

這個文件展示了如何使用inference.py進行推理的幾種方式

重要提示：
- 對於LOSO交叉驗證，應該使用 CTNetEnsembleInference 來加載所有模型並進行平均預測
- 單個模型推理（CTNetInference）只適用於單個受試者的模型
- Ensemble方法通常能提供更穩定和準確的預測結果
"""

import numpy as np
from inference import CTNetInference, CTNetEnsembleInference


# ========== 方式4: 實時分類分析（使用Ensemble） ==========
def example_realtime_classification():
    """實時分類分析示例 - 使用滑動窗口處理連續數據流"""
    print("\n" + "=" * 50)
    print("示例4: 實時分類分析（滑動窗口）")
    print("=" * 50)
    
    # 使用Ensemble推理器
    model_dir = "Loso_C_heads_2_depth_8_0"
    inferencer = CTNetEnsembleInference(
        model_dir=model_dir,
        dataset_type='C',
        heads=2, emb_size=16, depth=8,
        eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
        eeg1_pooling_size1=8, eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.25, flatten_eeg1=240
    )
    
    # 方式1: 從文件實時讀取（模擬實時數據流）
    print("\n--- 方式1: 從文件實時讀取 ---")
    txt_file = "bci_dataset_113-2/S01/2.txt"
    
    try:
        # 讀取完整數據
        full_data = np.loadtxt(txt_file, dtype=np.float32)
        print(f"數據文件: {txt_file}")
        print(f"數據長度: {len(full_data)} 個樣本")
        
        # 定義數據流生成器（模擬實時接收數據）
        def data_stream_generator(data, chunk_size=200):
            """模擬實時數據流，每次返回一小塊數據"""
            for i in range(0, len(data), chunk_size):
                yield data[i:i+chunk_size]
        
        # 定義回調函數（每次預測後調用）
        class_names = ['放鬆', '專注']  # 根據你的數據集調整
        
        def prediction_callback(result):
            """實時顯示預測結果"""
            pred = result['prediction']
            prob = result['probability']
            window_idx = result['window_idx']
            
            # 格式化輸出
            status = "🟢" if pred == 1 else "🔵"
            print(f"窗口 {window_idx:4d} | {status} 預測: {class_names[pred]:4s} | "
                  f"概率: [{prob[0]:.3f}, {prob[1]:.3f}] | "
                  f"置信度: {max(prob)*100:.1f}%")
        
        # 進行實時推理
        print("\n開始實時分析...")
        print("-" * 70)
        
        results_list = []
        for result in inferencer.predict_realtime(
            data_stream_generator(full_data, chunk_size=200),
            window_size=1000,      # 窗口大小
            stride=100,            # 滑動步長（越小越實時）
            smoothing_window=5,     # 平滑窗口（減少跳動）
            callback=prediction_callback
        ):
            results_list.append(result)
        
        print("-" * 70)
        print(f"\n分析完成！共處理 {len(results_list)} 個窗口")
        
        # 統計結果
        predictions = [r['prediction'] for r in results_list]
        from collections import Counter
        pred_counts = Counter(predictions)
        print(f"\n預測統計:")
        for cls_idx, count in pred_counts.items():
            percentage = count / len(predictions) * 100
            print(f"  {class_names[cls_idx]}: {count} 次 ({percentage:.1f}%)")
        
        return results_list
        
    except FileNotFoundError:
        print(f"文件不存在: {txt_file}")
        print("請確保數據文件存在")
        return None


# ========== 方式4b: 實時分類分析（從數組） ==========
def example_realtime_from_array():
    """從數組進行實時分類分析"""
    print("\n" + "=" * 50)
    print("示例4b: 實時分類分析（從數組）")
    print("=" * 50)
    
    # 使用Ensemble推理器
    model_dir = "Loso_C_heads_2_depth_8_0"
    inferencer = CTNetEnsembleInference(
        model_dir=model_dir,
        dataset_type='C',
        heads=2, emb_size=16, depth=8,
        eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
        eeg1_pooling_size1=8, eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.25, flatten_eeg1=240
    )
    
    # 創建模擬數據（實際使用時替換為真實數據流）
    print("使用模擬數據進行實時分析...")
    simulated_data = np.random.randn(5000).astype(np.float32)  # 5000個樣本
    
    class_names = ['放鬆', '專注']
    results_list = []
    
    print("\n實時預測結果:")
    print("-" * 70)
    
    for result in inferencer.predict_realtime(
        simulated_data,
        window_size=1000,
        stride=200,           # 每200個樣本滑動一次
        smoothing_window=3,   # 使用3個窗口的移動平均
        callback=None          # 不使用回調，直接處理結果
    ):
        pred = result['prediction']
        prob = result['probability']
        window_idx = result['window_idx']
        
        status = "🟢" if pred == 1 else "🔵"
        print(f"窗口 {window_idx:3d} | {status} {class_names[pred]:4s} | "
              f"概率: [{prob[0]:.3f}, {prob[1]:.3f}]")
        
        results_list.append(result)
        
        # 只顯示前20個窗口，避免輸出過多
        if window_idx >= 19:
            print("... (省略後續結果)")
            break
    
    print("-" * 70)
    print(f"\n處理完成！共 {len(results_list)} 個窗口")
    
    return results_list


# ========== 方式5: 使用真實數據進行推理（使用Ensemble） ==========
def example_real_data_inference():
    """使用真實測試數據進行推理 - 使用Ensemble"""
    print("\n" + "=" * 50)
    print("示例4: 使用真實測試數據推理（Ensemble）")
    print("=" * 50)
    
    from utils import load_data_evaluate
    
    # 使用Ensemble推理器
    model_dir = "Loso_C_heads_2_depth_8_0"
    inferencer = CTNetEnsembleInference(
        model_dir=model_dir,
        dataset_type='C',
        heads=2, emb_size=16, depth=8,
        eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
        eeg1_pooling_size1=8, eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.25, flatten_eeg1=240
    )
    
    # 加載測試數據
    data_dir = "./bci_dataset_113-2/"
    train_data, train_label, test_data, test_label = load_data_evaluate(
        data_dir, 'C', 1, mode_evaluate='LOSO'
    )
    
    # 準備測試數據（需要添加channel維度）
    test_data = np.expand_dims(test_data, axis=1)  # (n_trials, 1, 1000)
    
    # 計算標準化參數（使用訓練數據）
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    inferencer.set_normalization_params(train_mean, train_std)
    
    # 進行預測
    predictions, probs = inferencer.predict(test_data[:10], return_probs=True)  # 只測試前10個樣本
    
    # 處理標籤形狀（確保是1D陣列）
    true_labels = test_label[:10]
    if true_labels.ndim > 1:
        true_labels = true_labels.flatten()  # 將 (n, 1) 轉換為 (n,)
    true_labels = true_labels - 1  # 轉換為0-indexed
    
    print(f"測試數據前10個樣本:")
    print(f"真實標籤: {true_labels}")
    print(f"預測標籤: {predictions}")
    
    # 計算準確率
    correct = (predictions == true_labels).sum()
    accuracy = correct / len(predictions)
    print(f"準確率: {accuracy:.4f} ({correct}/{len(predictions)})")
    
    # 顯示每個樣本的詳細信息
    print("\n詳細結果:")
    for i in range(len(predictions)):
        print(f"  樣本 {i+1}: 真實={true_labels[i]}, 預測={predictions[i]}, "
              f"概率=[類別0={probs[i][0]:.4f}, 類別1={probs[i][1]:.4f}]")


if __name__ == "__main__":
    # 運行示例
    print("CTNet推理示例\n")
    
    # 示例1: 單個樣本
   # example_single_sample()
    
    # 示例2: 批量推理
   # example_batch_inference()
    
    # 示例3: 從文件推理
    # example_file_inference()
    
    # 示例3b: 推理並同時評估（推薦）
    # example_inference_with_evaluation()
    
    # 示例4: 實時分類分析（推薦用於實時應用）
    example_realtime_classification()
    
    # 示例4b: 實時分類分析（從數組）
    # example_realtime_from_array()
    
    # 示例5: 真實數據推理
   # example_real_data_inference()
    
    print("\n" + "=" * 50)
    print("所有示例運行完成！")
    print("=" * 50)
