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
            stride=500,            # 滑動步長（越小越實時）
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

if __name__ == "__main__":
    # 運行示例
    print("CTNet推理示例\n")
    example_realtime_classification()
    
    print("\n" + "=" * 50)
    print("所有示例運行完成！")
    print("=" * 50)
