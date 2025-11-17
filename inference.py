"""
CTNet Inference Script
用於對EEG數據進行推理/預測

使用方法：
1. 單個樣本推理：
   python inference.py --model_path Loso_C_heads_2_depth_8_0/model_1.pth --data_path your_data.npy

2. 批量推理（從文件夾）：
   python inference.py --model_path Loso_C_heads_2_depth_8_0/model_1.pth --data_dir ./test_data/

3. 從txt文件推理（單通道）：
   python inference.py --model_path Loso_C_heads_2_depth_8_0/model_1.pth --txt_file ./data.txt
"""

import os
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
import argparse
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import math
from utils import numberClassChannel

# 設置CUDA
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 導入模型架構（從loso.py）
# 注意：需要確保 loso.py 在同一個目錄下
from loso import EEGTransformer


class CTNetInference:
    """CTNet推理類"""
    
    def predict_realtime(self, data_stream, window_size=1000, stride=100, 
                         smoothing_window=5, callback=None):
        """
        實時推理 - 使用滑動窗口處理連續數據流
        
        Parameters
        ----------
        data_stream : iterable or np.ndarray
            數據流（可以是數組、生成器或文件讀取器）
        window_size : int
            滑動窗口大小（默認1000）
        stride : int
            滑動步長（默認100，越小越實時但計算量越大）
        smoothing_window : int
            平滑窗口大小（用於平滑預測結果，0表示不使用平滑）
        callback : callable, optional
            回調函數，每次預測後調用 callback(prediction, probability, window_idx)
        
        Yields
        ------
        dict
            包含預測結果的字典
        """
        import time
        
        # 如果輸入是數組，轉換為迭代器
        if isinstance(data_stream, np.ndarray):
            data_stream = iter([data_stream])
        
        buffer = np.array([], dtype=np.float32)
        predictions_history = []
        
        window_idx = 0
        for chunk in data_stream:
            # 將新數據添加到緩衝區
            chunk = np.array(chunk, dtype=np.float32).flatten()
            buffer = np.concatenate([buffer, chunk])
            
            # 處理所有完整的窗口
            while len(buffer) >= window_size:
                # 提取一個窗口
                window_data = buffer[:window_size]
                
                # 進行預測
                try:
                    pred, prob = self.predict(window_data.reshape(1, -1), return_probs=True)
                    prediction = pred[0]
                    probability = prob[0]
                    
                    # 平滑處理（移動平均）
                    if smoothing_window > 0:
                        predictions_history.append(prediction)
                        if len(predictions_history) > smoothing_window:
                            predictions_history.pop(0)
                        # 使用最常見的預測作為平滑結果
                        from collections import Counter
                        smoothed_pred = Counter(predictions_history).most_common(1)[0][0]
                    else:
                        smoothed_pred = prediction
                    
                    result = {
                        'prediction': smoothed_pred,
                        'raw_prediction': prediction,
                        'probability': probability,
                        'window_idx': window_idx,
                        'timestamp': time.time()
                    }
                    
                    # 調用回調函數
                    if callback is not None:
                        callback(result)
                    
                    yield result
                    
                except Exception as e:
                    print(f"預測窗口 {window_idx} 時出錯: {e}")
                
                # 移動窗口（根據stride）
                buffer = buffer[stride:]
                window_idx += 1
    
    def __init__(self, model_path, dataset_type='C', 
                 heads=2, emb_size=16, depth=8,
                 eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.25, flatten_eeg1=240):
        """
        初始化推理器
        
        Parameters
        ----------
        model_path : str
            模型文件路徑 (.pth)
        dataset_type : str
            數據集類型 'A', 'B', 或 'C'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type
        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        
        # 加載模型權重
        print(f"正在加載模型: {model_path}")
        # 解決 pickle 載入問題：模型保存時記錄的是 __main__.EEGTransformer
        # 需要在載入前將 loso.EEGTransformer 註冊到 __main__ 模組
        import sys
        
        # 確保 loso 模組已導入
        import loso
        
        # 將 loso 模組中的類註冊到 __main__ 模組（解決 pickle 載入問題）
        # 這樣 torch.load 就能找到 __main__.EEGTransformer
        if not hasattr(sys.modules['__main__'], 'EEGTransformer'):
            sys.modules['__main__'].EEGTransformer = loso.EEGTransformer
            # 也註冊其他相關類（以防萬一）
            related_classes = [
                'PatchEmbeddingCNN', 'MultiHeadAttention', 'FeedForwardBlock',
                'ClassificationHead', 'ResidualAdd', 'TransformerEncoderBlock',
                'TransformerEncoder', 'BranchEEGNetTransformer', 'PositioinalEncoding'
            ]
            for attr_name in related_classes:
                if hasattr(loso, attr_name) and not hasattr(sys.modules['__main__'], attr_name):
                    setattr(sys.modules['__main__'], attr_name, getattr(loso, attr_name))
        
        # 載入模型
        try:
            loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(loaded_model, torch.nn.Module):
                # 載入的是完整模型，直接使用
                self.model = loaded_model.to(self.device)
            else:
                # 載入的是 state_dict，需要先創建模型再載入
                self.model = EEGTransformer(
                    heads=heads,
                    emb_size=emb_size,
                    depth=depth,
                    database_type=dataset_type,
                    eeg1_f1=eeg1_f1,
                    eeg1_kernel_size=eeg1_kernel_size,
                    eeg1_D=eeg1_D,
                    eeg1_pooling_size1=eeg1_pooling_size1,
                    eeg1_pooling_size2=eeg1_pooling_size2,
                    eeg1_dropout_rate=eeg1_dropout_rate,
                    eeg1_number_channel=self.number_channel,
                    flatten_eeg1=flatten_eeg1,
                ).to(self.device)
                self.model.load_state_dict(loaded_model)
        except Exception as e:
            print(f"載入模型時出錯: {e}")
            raise Exception(f"無法載入模型。請確保模型文件格式正確且 loso 模組可用。錯誤: {e}")
        
        self.model.eval()
        print("模型加載完成！")
        
        # 數據標準化參數（需要從訓練數據計算，這裡提供默認值）
        self.mean = None
        self.std = None
    
    def set_normalization_params(self, mean, std):
        """設置數據標準化參數"""
        self.mean = mean
        self.std = std
    
    def normalize_data(self, data):
        """標準化數據"""
        if self.mean is not None and self.std is not None:
            return (data - self.mean) / self.std
        else:
            # 使用數據本身的均值和標準差
            data_mean = np.mean(data)
            data_std = np.std(data)
            return (data - data_mean) / data_std
    
    def prepare_data(self, data):
        """
        準備數據進行推理
        
        Parameters
        ----------
        data : np.ndarray
            輸入數據，形狀可以是：
            - (1000,) - 單個時間序列
            - (n_trials, 1000) - 多個時間序列
            - (n_trials, 1, 1000) - 已添加channel維度
            - (n_trials, 1, n_channels, 1000) - 完整格式
        
        Returns
        -------
        torch.Tensor
            準備好的數據，形狀 (batch, 1, n_channels, 1000)
        """
        data = np.array(data, dtype=np.float32)
        
        # 處理不同輸入形狀
        if data.ndim == 1:
            # (1000,) -> (1, 1, 1, 1000)
            data = data.reshape(1, 1, self.number_channel, -1)
        elif data.ndim == 2:
            # (n_trials, 1000) -> (n_trials, 1, 1, 1000)
            data = data.reshape(data.shape[0], 1, self.number_channel, -1)
        elif data.ndim == 3:
            # (n_trials, 1, 1000) -> (n_trials, 1, 1, 1000)
            if data.shape[1] != 1:
                data = data.reshape(data.shape[0], 1, self.number_channel, -1)
            else:
                data = data.reshape(data.shape[0], 1, self.number_channel, -1)
        elif data.ndim == 4:
            # 已經是正確格式 (n_trials, 1, n_channels, 1000)
            pass
        else:
            raise ValueError(f"不支持的數據維度: {data.ndim}")
        
        # 確保時間維度是1000
        if data.shape[-1] != 1000:
            raise ValueError(f"時間維度必須是1000，但得到: {data.shape[-1]}")
        
        # 標準化
        data = self.normalize_data(data)
        
        # 轉換為torch tensor
        return torch.from_numpy(data).float()
    
    def predict(self, data, return_probs=False, return_features=False):
        """
        進行推理
        
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            輸入數據
        return_probs : bool
            是否返回概率分布
        return_features : bool
            是否返回特徵向量
        
        Returns
        -------
        predictions : np.ndarray
            預測的類別 (0-indexed)
        probs : np.ndarray (optional)
            類別概率分布
        features : np.ndarray (optional)
            特徵向量
        """
        # 準備數據
        if isinstance(data, np.ndarray):
            data_tensor = self.prepare_data(data)
        else:
            data_tensor = data
        
        data_tensor = data_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            features, outputs = self.model(data_tensor)
            
            # 獲取預測類別
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            probs_np = probs.cpu().numpy()
            features_np = features.cpu().numpy()
        
        result = [predictions]
        if return_probs:
            result.append(probs_np)
        if return_features:
            result.append(features_np)
        
        if len(result) == 1:
            return result[0]
        return tuple(result)
    
    def predict_from_file(self, file_path, window_size=1000):
        """
        從文件加載數據並進行預測
        
        Parameters
        ----------
        file_path : str
            文件路徑（.npy, .txt, 或 .npz）
        window_size : int
            時間窗口大小（默認1000）
        
        Returns
        -------
        predictions : np.ndarray
            預測結果
        """
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.txt'):
            data = np.loadtxt(file_path, dtype=np.float32)
            # 如果是長序列，需要切片
            if len(data) > window_size:
                n_windows = len(data) // window_size
                data = data[:n_windows * window_size].reshape(n_windows, window_size)
        elif file_path.endswith('.npz'):
            npz = np.load(file_path)
            # 嘗試常見的鍵名
            keys = list(npz.keys())
            if keys:
                data = npz[keys[0]]
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        return self.predict(data, return_probs=True)
    
    def predict_batch_from_dir(self, data_dir, file_pattern='*.npy'):
        """
        從目錄批量加載數據並進行預測
        
        Parameters
        ----------
        data_dir : str
            數據目錄
        file_pattern : str
            文件匹配模式
        
        Returns
        -------
        results : dict
            每個文件的預測結果
        """
        import glob
        results = {}
        
        files = glob.glob(os.path.join(data_dir, file_pattern))
        for file_path in files:
            print(f"處理文件: {file_path}")
            predictions, probs = self.predict_from_file(file_path)
            results[file_path] = {
                'predictions': predictions,
                'probabilities': probs
            }
        
        return results
    
    def evaluate(self, data, true_labels, verbose=True):
        """
        進行推理並同時計算評估指標
        
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            輸入數據
        true_labels : np.ndarray
            真實標籤（0-indexed）
        verbose : bool
            是否打印評估結果
        
        Returns
        -------
        dict
            包含預測結果和評估指標的字典
        """
        from utils import calMetrics
        from sklearn.metrics import confusion_matrix
        
        # 進行預測
        predictions, probs = self.predict(data, return_probs=True)
        
        # 確保標籤格式正確
        true_labels = np.array(true_labels).flatten()
        if len(true_labels) != len(predictions):
            raise ValueError(f"標籤數量 ({len(true_labels)}) 與預測數量 ({len(predictions)}) 不匹配")
        
        # 計算評估指標
        accuracy, precision, recall, f1, kappa = calMetrics(true_labels, predictions)
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictions)
        
        # 構建結果字典
        results = {
            'predictions': predictions,
            'probabilities': probs,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'kappa': kappa,
            'confusion_matrix': cm
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("評估結果")
            print("=" * 50)
            print(f"準確率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"精確率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
            print(f"召回率 (Recall): {recall:.4f} ({recall*100:.2f}%)")
            print(f"F1分數 (F1-Score): {f1:.4f} ({f1*100:.2f}%)")
            print(f"Kappa係數: {kappa:.4f}")
            print(f"\n混淆矩陣:")
            print(cm)
            print("=" * 50)
        
        return results


class CTNetEnsembleInference:
    """CTNet Ensemble推理類 - 用於LOSO交叉驗證的多模型平均預測"""
    
    def __init__(self, model_dir, dataset_type='C', 
                 heads=2, emb_size=16, depth=8,
                 eeg1_f1=8, eeg1_kernel_size=64, eeg1_D=2,
                 eeg1_pooling_size1=8, eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.25, flatten_eeg1=240,
                 n_models=None):
        """
        初始化Ensemble推理器
        
        Parameters
        ----------
        model_dir : str
            模型文件夾路徑（包含 model_1.pth, model_2.pth, ...）
        dataset_type : str
            數據集類型 'A', 'B', 或 'C'
        n_models : int, optional
            模型數量，如果為None則自動檢測
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type
        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        self.model_dir = model_dir
        
        # 解決 pickle 載入問題
        import sys
        import loso
        
        if not hasattr(sys.modules['__main__'], 'EEGTransformer'):
            sys.modules['__main__'].EEGTransformer = loso.EEGTransformer
            related_classes = [
                'PatchEmbeddingCNN', 'MultiHeadAttention', 'FeedForwardBlock',
                'ClassificationHead', 'ResidualAdd', 'TransformerEncoderBlock',
                'TransformerEncoder', 'BranchEEGNetTransformer', 'PositioinalEncoding'
            ]
            for attr_name in related_classes:
                if hasattr(loso, attr_name) and not hasattr(sys.modules['__main__'], attr_name):
                    setattr(sys.modules['__main__'], attr_name, getattr(loso, attr_name))
        
        # 自動檢測模型數量
        import glob
        model_files = sorted(glob.glob(os.path.join(model_dir, 'model_*.pth')))
        if n_models is None:
            n_models = len(model_files)
        
        if n_models == 0:
            raise ValueError(f"在 {model_dir} 中找不到模型文件")
        
        print(f"正在加載 {n_models} 個模型進行Ensemble推理...")
        self.models = []
        
        for i in range(1, n_models + 1):
            model_path = os.path.join(model_dir, f'model_{i}.pth')
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}，跳過")
                continue
            
            try:
                loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded_model, torch.nn.Module):
                    model = loaded_model.to(self.device)
                else:
                    model = EEGTransformer(
                        heads=heads,
                        emb_size=emb_size,
                        depth=depth,
                        database_type=dataset_type,
                        eeg1_f1=eeg1_f1,
                        eeg1_kernel_size=eeg1_kernel_size,
                        eeg1_D=eeg1_D,
                        eeg1_pooling_size1=eeg1_pooling_size1,
                        eeg1_pooling_size2=eeg1_pooling_size2,
                        eeg1_dropout_rate=eeg1_dropout_rate,
                        eeg1_number_channel=self.number_channel,
                        flatten_eeg1=flatten_eeg1,
                    ).to(self.device)
                    model.load_state_dict(loaded_model)
                
                model.eval()
                self.models.append(model)
                if (i) % 5 == 0 or i == n_models:
                    print(f"  已加載 {i}/{n_models} 個模型")
            except Exception as e:
                print(f"警告: 載入模型 {model_path} 時出錯: {e}，跳過")
                continue
        
        if len(self.models) == 0:
            raise Exception("沒有成功載入任何模型")
        
        print(f"Ensemble推理器初始化完成！共 {len(self.models)} 個模型")
        
        # 數據標準化參數
        self.mean = None
        self.std = None
    
    def set_normalization_params(self, mean, std):
        """設置數據標準化參數"""
        self.mean = mean
        self.std = std
    
    def normalize_data(self, data):
        """標準化數據"""
        if self.mean is not None and self.std is not None:
            return (data - self.mean) / self.std
        else:
            data_mean = np.mean(data)
            data_std = np.std(data)
            return (data - data_mean) / data_std
    
    def prepare_data(self, data):
        """準備數據進行推理（與CTNetInference相同）"""
        data = np.array(data, dtype=np.float32)
        
        if data.ndim == 1:
            data = data.reshape(1, 1, self.number_channel, -1)
        elif data.ndim == 2:
            data = data.reshape(data.shape[0], 1, self.number_channel, -1)
        elif data.ndim == 3:
            if data.shape[1] != 1:
                data = data.reshape(data.shape[0], 1, self.number_channel, -1)
            else:
                data = data.reshape(data.shape[0], 1, self.number_channel, -1)
        elif data.ndim == 4:
            pass
        else:
            raise ValueError(f"不支持的數據維度: {data.ndim}")
        
        if data.shape[-1] != 1000:
            raise ValueError(f"時間維度必須是1000，但得到: {data.shape[-1]}")
        
        data = self.normalize_data(data)
        return torch.from_numpy(data).float()
    
    def predict(self, data, return_probs=False, return_features=False):
        """
        使用Ensemble進行推理（所有模型的平均預測）
        
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            輸入數據
        return_probs : bool
            是否返回概率分布
        return_features : bool
            是否返回特徵向量（返回所有模型特徵的平均）
        
        Returns
        -------
        predictions : np.ndarray
            預測的類別 (0-indexed)
        probs : np.ndarray (optional)
            平均類別概率分布
        features : np.ndarray (optional)
            平均特徵向量
        """
        if isinstance(data, np.ndarray):
            data_tensor = self.prepare_data(data)
        else:
            data_tensor = data
        
        data_tensor = data_tensor.to(self.device)
        
        # 收集所有模型的輸出
        all_outputs = []
        all_probs = []
        all_features = []
        
        with torch.no_grad():
            for model in self.models:
                features, outputs = model(data_tensor)
                probs = F.softmax(outputs, dim=1)
                
                all_outputs.append(outputs)
                all_probs.append(probs)
                all_features.append(features)
        
        # 平均所有模型的輸出
        avg_outputs = torch.stack(all_outputs).mean(dim=0)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        avg_features = torch.stack(all_features).mean(dim=0)
        
        # 獲取預測類別
        predictions = torch.argmax(avg_outputs, dim=1).cpu().numpy()
        probs_np = avg_probs.cpu().numpy()
        features_np = avg_features.cpu().numpy()
        
        result = [predictions]
        if return_probs:
            result.append(probs_np)
        if return_features:
            result.append(features_np)
        
        if len(result) == 1:
            return result[0]
        return tuple(result)
    
    def predict_from_file(self, file_path, window_size=1000):
        """從文件加載數據並進行Ensemble預測"""
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.txt'):
            data = np.loadtxt(file_path, dtype=np.float32)
            if len(data) > window_size:
                n_windows = len(data) // window_size
                data = data[:n_windows * window_size].reshape(n_windows, window_size)
        elif file_path.endswith('.npz'):
            npz = np.load(file_path)
            keys = list(npz.keys())
            if keys:
                data = npz[keys[0]]
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        return self.predict(data, return_probs=True)
    
    def predict_batch_from_dir(self, data_dir, file_pattern='*.npy'):
        """從目錄批量加載數據並進行Ensemble預測"""
        import glob
        results = {}
        
        files = glob.glob(os.path.join(data_dir, file_pattern))
        for file_path in files:
            print(f"處理文件: {file_path}")
            predictions, probs = self.predict_from_file(file_path)
            results[file_path] = {
                'predictions': predictions,
                'probabilities': probs
            }
        
        return results
    
    def evaluate(self, data, true_labels, verbose=True):
        """
        進行Ensemble推理並同時計算評估指標
        
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            輸入數據
        true_labels : np.ndarray
            真實標籤（0-indexed）
        verbose : bool
            是否打印評估結果
        
        Returns
        -------
        dict
            包含預測結果和評估指標的字典
        """
        from utils import calMetrics
        from sklearn.metrics import confusion_matrix
        
        # 進行Ensemble預測
        predictions, probs = self.predict(data, return_probs=True)
        
        # 確保標籤格式正確
        true_labels = np.array(true_labels).flatten()
        if len(true_labels) != len(predictions):
            raise ValueError(f"標籤數量 ({len(true_labels)}) 與預測數量 ({len(predictions)}) 不匹配")
        
        # 計算評估指標
        accuracy, precision, recall, f1, kappa = calMetrics(true_labels, predictions)
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_labels, predictions)
        
        # 構建結果字典
        results = {
            'predictions': predictions,
            'probabilities': probs,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'kappa': kappa,
            'confusion_matrix': cm,
            'n_models': len(self.models)  # Ensemble使用的模型數量
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print(f"Ensemble評估結果 (使用 {len(self.models)} 個模型)")
            print("=" * 50)
            print(f"準確率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"精確率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
            print(f"召回率 (Recall): {recall:.4f} ({recall*100:.2f}%)")
            print(f"F1分數 (F1-Score): {f1:.4f} ({f1*100:.2f}%)")
            print(f"Kappa係數: {kappa:.4f}")
            print(f"\n混淆矩陣:")
            print(cm)
            print("=" * 50)
        
        return results
    
    def predict_realtime(self, data_stream, window_size=1000, stride=100, 
                         smoothing_window=5, callback=None):
        """
        實時推理 - 使用滑動窗口處理連續數據流
        
        Parameters
        ----------
        data_stream : iterable or np.ndarray
            數據流（可以是數組、生成器或文件讀取器）
        window_size : int
            滑動窗口大小（默認1000）
        stride : int
            滑動步長（默認100，越小越實時但計算量越大）
        smoothing_window : int
            平滑窗口大小（用於平滑預測結果，0表示不使用平滑）
        callback : callable, optional
            回調函數，每次預測後調用 callback(prediction, probability, window_idx)
        
        Yields
        ------
        dict
            包含預測結果的字典：
            - 'prediction': 預測類別
            - 'probability': 類別概率
            - 'window_idx': 窗口索引
            - 'timestamp': 時間戳（如果數據流提供）
        """
        import time
        
        # 如果輸入是數組，轉換為迭代器
        if isinstance(data_stream, np.ndarray):
            data_stream = iter([data_stream])
        
        buffer = np.array([], dtype=np.float32)
        predictions_history = []
        
        window_idx = 0
        for chunk in data_stream:
            # 將新數據添加到緩衝區
            chunk = np.array(chunk, dtype=np.float32).flatten()
            buffer = np.concatenate([buffer, chunk])
            
            # 處理所有完整的窗口
            while len(buffer) >= window_size:
                # 提取一個窗口
                window_data = buffer[:window_size]
                
                # 進行預測
                try:
                    pred, prob = self.predict(window_data.reshape(1, -1), return_probs=True)
                    prediction = pred[0]
                    probability = prob[0]
                    
                    # 平滑處理（移動平均）
                    if smoothing_window > 0:
                        predictions_history.append(prediction)
                        if len(predictions_history) > smoothing_window:
                            predictions_history.pop(0)
                        # 使用最常見的預測作為平滑結果
                        from collections import Counter
                        smoothed_pred = Counter(predictions_history).most_common(1)[0][0]
                    else:
                        smoothed_pred = prediction
                    
                    result = {
                        'prediction': smoothed_pred,
                        'raw_prediction': prediction,
                        'probability': probability,
                        'window_idx': window_idx,
                        'timestamp': time.time()
                    }
                    
                    # 調用回調函數
                    if callback is not None:
                        callback(result)
                    
                    yield result
                    
                except Exception as e:
                    print(f"預測窗口 {window_idx} 時出錯: {e}")
                
                # 移動窗口（根據stride）
                buffer = buffer[stride:]
                window_idx += 1


def main():
    parser = argparse.ArgumentParser(description='CTNet Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路徑 (.pth)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='單個數據文件路徑')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='數據目錄（批量推理）')
    parser.add_argument('--txt_file', type=str, default=None,
                        help='txt文件路徑（單通道數據）')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出結果文件路徑')
    parser.add_argument('--dataset_type', type=str, default='C',
                        choices=['A', 'B', 'C'],
                        help='數據集類型 (A: BCI IV2a, B: BCI IV2b, C: 自定義)')
    parser.add_argument('--heads', type=int, default=2,
                        help='Transformer heads數量')
    parser.add_argument('--emb_size', type=int, default=16,
                        help='嵌入維度')
    parser.add_argument('--depth', type=int, default=8,
                        help='Transformer深度')
    parser.add_argument('--eeg1_f1', type=int, default=8,
                        help='EEG1 f1參數')
    parser.add_argument('--eeg1_kernel_size', type=int, default=64,
                        help='EEG1 kernel size')
    parser.add_argument('--eeg1_D', type=int, default=2,
                        help='EEG1 D參數')
    parser.add_argument('--eeg1_pooling_size1', type=int, default=8,
                        help='EEG1 pooling size 1')
    parser.add_argument('--eeg1_pooling_size2', type=int, default=8,
                        help='EEG1 pooling size 2')
    parser.add_argument('--eeg1_dropout_rate', type=float, default=0.25,
                        help='EEG1 dropout rate')
    parser.add_argument('--flatten_eeg1', type=int, default=240,
                        help='Flatten EEG1大小')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = CTNetInference(
        model_path=args.model_path,
        dataset_type=args.dataset_type,
        heads=args.heads,
        emb_size=args.emb_size,
        depth=args.depth,
        eeg1_f1=args.eeg1_f1,
        eeg1_kernel_size=args.eeg1_kernel_size,
        eeg1_D=args.eeg1_D,
        eeg1_pooling_size1=args.eeg1_pooling_size1,
        eeg1_pooling_size2=args.eeg1_pooling_size2,
        eeg1_dropout_rate=args.eeg1_dropout_rate,
        flatten_eeg1=args.flatten_eeg1
    )
    
    # 進行推理
    if args.txt_file:
        print(f"從txt文件推理: {args.txt_file}")
        predictions, probs = inferencer.predict_from_file(args.txt_file)
        print(f"預測結果: {predictions}")
        print(f"類別概率: {probs}")
        
    elif args.data_path:
        print(f"從文件推理: {args.data_path}")
        predictions, probs = inferencer.predict_from_file(args.data_path)
        print(f"預測結果: {predictions}")
        print(f"類別概率: {probs}")
        
    elif args.data_dir:
        print(f"從目錄批量推理: {args.data_dir}")
        results = inferencer.predict_batch_from_dir(args.data_dir)
        for file_path, result in results.items():
            print(f"\n文件: {file_path}")
            print(f"預測結果: {result['predictions']}")
            print(f"類別概率: {result['probabilities']}")
    else:
        # 示例：使用隨機數據
        print("使用隨機數據進行示例推理...")
        sample_data = np.random.randn(10, 1000).astype(np.float32)
        predictions, probs = inferencer.predict(sample_data, return_probs=True)
        print(f"示例預測結果: {predictions}")
        print(f"類別概率形狀: {probs.shape}")
    
    # 保存結果
    if args.output:
        import json
        results_dict = {
            'predictions': predictions.tolist() if 'predictions' in locals() else None,
            'probabilities': probs.tolist() if 'probs' in locals() else None
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"\n結果已保存到: {args.output}")


if __name__ == "__main__":
    main()
