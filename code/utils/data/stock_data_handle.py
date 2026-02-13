import os
import sys
import numpy as np
import pandas as pd
import gc
import datetime

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.preprocess import FeatureEngineer
from utils import config

class Stock_Data():
    def __init__(self, full_stock_path, temporal_len, attr = config.TECHNICAL_INDICATORS_LIST, temporal_feature = config.TEMPORAL_FEATURE, scale=True, prediction_len=[2,5], exp_type='mae', a=1.2):
        self.scale = scale
        self.a = a
        self.attr = attr
        self.temporal_feature = temporal_feature
        self.full_stock_dir = full_stock_path
        self.ticker_list = config.USE_CSI_300_TICKET
        self.border_dates = config.CSI_date_trans
        self.prediction_len = prediction_len
        self.exp_type = exp_type

        self.seq_len = temporal_len
        self.type_map = {'train':0, 'valid':1, 'test':2}
        self.pred_type_map = {'label_short_term':0, 'label_long_term':1}

        self.__read_data__()

    def __read_data__(self):
        stock_num = len(self.ticker_list)
        df_list = []
        
        print(f"Loading data for {self.exp_type} mode...")
        for ticket in self.ticker_list:
            file_path = os.path.join(self.full_stock_dir, ticket + '.csv')
            if not os.path.exists(file_path): continue
            
            temp_df = pd.read_csv(file_path)
            # 修改 1：日期规范化
            temp_df['date'] = pd.to_datetime(temp_df['date']).dt.normalize()
            
            numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price']
            for col in numeric_cols:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').interpolate().ffill().bfill().fillna(0)
            
            temp_df['label_short_term'] = temp_df['close'].pct_change(periods=self.prediction_len[0]).shift(-self.prediction_len[0])
            temp_df['label_long_term'] = temp_df['close'].pct_change(periods=self.prediction_len[1]).shift(-self.prediction_len[1])
            temp_df['tic'] = ticket

            # 恢复安全检查：检查每只股票的原始数据列是否全空
            raw_cols_to_check = ['open', 'close', 'high', 'low', 'volume']
            for col in raw_cols_to_check:
                if temp_df[col].isna().all():
                    print(f"Error: Column '{col}' for ticker '{ticket}' is entirely empty in CSV. Program exiting.")
                    sys.exit(1)

            df_list.append(temp_df)
            
        df = pd.concat(df_list, ignore_index=True)
        del df_list

        # 特征工程
        fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=config.TECHNICAL_INDICATORS_LIST)
        df = fe.preprocess_data(df)

        # 严格日期对齐
        all_dates = sorted(df['date'].unique())
        full_index = pd.MultiIndex.from_product([all_dates, self.ticker_list], names=['date', 'tic'])
        df = df.set_index(['date', 'tic']).reindex(full_index).reset_index()
        df = df.sort_values(['date','tic'], ignore_index=True)
        
        # 恢复安全检查：在补全前检查是否存在全空列
        fill_cols = [col for col in df.columns if col not in ['date', 'tic']]
        for tic, tic_df in df.groupby('tic'):
            for col in fill_cols:
                if tic_df[col].isna().all():
                    print(f"Error: Column '{col}' for ticker '{tic}' is entirely empty after alignment but before filling. Program exiting.")
                    sys.exit(1)

        df[fill_cols] = df.groupby('tic')[fill_cols].transform(lambda x: x.interpolate().ffill().bfill().fillna(0))
        df['date_str'] = df['date'].dt.strftime('%Y%m%d')

        # 恢复日期特征
        df['month_day'] = df['date'].dt.month + df['date'].dt.day / 100.0
        df['weekday'] = df['date'].dt.dayofweek

        # 恢复：处理技术指标中的无穷大值，防止梯度爆炸
        df[self.attr] = df[self.attr].replace([np.inf], config.INF)
        df[self.attr] = df[self.attr].replace([-np.inf], config.INF * (-1))

        # 修改 1：按需生成协方差
        lookback = 252
        unique_date = sorted(df['date'].unique())
        
        if self.exp_type == 'mae':
            print("MAE Mode: Generating covariance matrix...")
            price_pivot = df.pivot_table(index='date', columns='tic', values='close')
            return_pivot = price_pivot.pct_change().fillna(0)
            cov_list = []
            for i in range(lookback, len(unique_date)):
                window = return_pivot[(return_pivot.index < unique_date[i]) & (return_pivot.index >= unique_date[i-lookback+1])]
                cov_list.append(window.cov().values)
            
            df_cov = pd.DataFrame({'date': unique_date[lookback:], 'cov_list': cov_list})
            df = df.merge(df_cov, on='date')
            del cov_list, df_cov
        else:
            print("Pred Mode: Skipping expensive covariance matrices...")
            df = df[df['date'] >= unique_date[lookback]]

        # 恢复精细化缩放逻辑
        if self.scale:
            # 标准化只在训练集上 fit，避免测试集信息泄露
            scaler = MinMaxScaler()
            # 获取训练集的范围
            train_mask = (df['date_str'] >= self.border_dates[0]) & (df['date_str'] <= self.border_dates[1])
            
            # 修改：找到训练集中最新的 250 组日期进行 fit
            train_dates = sorted(df.loc[train_mask, 'date'].unique())
            fit_dates = train_dates[-min(len(train_dates), 250):]
            fit_mask = df['date'].isin(fit_dates)

            # 1. 归一化技术指标 (Tech Indicators)
            train_data_for_scaler = df.loc[fit_mask, self.attr]
            scaler.fit(train_data_for_scaler.values)
            # 修改：将获取的最大值乘以系数 a
            scaler.data_max_ *= self.a
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_ + 1e-8)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            df[self.attr] = scaler.transform(df[self.attr].values)

            # 2. 归一化时序特征 (Temporal Features) - 分组比例缩放
            price_abs_cols = ['open', 'close', 'high', 'low']
            price_diff_cols = ['dopen', 'dclose', 'dhigh', 'dlow']
            vol_abs_cols = ['volume']
            vol_diff_cols = ['dvolume']

            # 筛选当前存在的列
            valid_price_abs = [c for c in price_abs_cols if c in self.temporal_feature]
            valid_price_diff = [c for c in price_diff_cols if c in self.temporal_feature]
            valid_vol_abs = [c for c in vol_abs_cols if c in self.temporal_feature]
            valid_vol_diff = [c for c in vol_diff_cols if c in self.temporal_feature]

            # --- Group A: 价格组 (保持价格间比例) ---
            if valid_price_abs:
                # 修改：使用 fit_mask 并乘以系数 a
                train_price_vals = df.loc[fit_mask, valid_price_abs].values.flatten()
                max_price = np.max(train_price_vals) * self.a + 1e-8
                df[valid_price_abs] = df[valid_price_abs] / max_price
                if valid_price_diff:
                    df[valid_price_diff] = df[valid_price_diff] / max_price

            # --- Group B: 成交量组 (保持成交量比例) ---
            if valid_vol_abs:
                # 修改：使用 fit_mask 并乘以系数 a
                train_vol_vals = df.loc[fit_mask, valid_vol_abs].values.flatten()
                max_vol = np.max(train_vol_vals) * self.a + 1e-8
                df[valid_vol_abs] = df[valid_vol_abs] / max_vol
                if valid_vol_diff:
                    df[valid_vol_diff] = df[valid_vol_diff] / max_vol

            # --- Group C: 时间特征组 (恢复离散化逻辑) ---
            df['month_day'] = (df['month_day'] / 2).astype(int) / 7.0
            df['weekday'] = df['weekday'] / 7.0

        # 优化 2：预转置内存布局 (Stocks, Days, Feats)
        dates_final = df['date_str'].unique()
        num_days = len(dates_final)
        data_tech = df[self.attr].values.reshape(num_days, stock_num, -1).astype(np.float32)
        data_temp = df[self.temporal_feature].values.reshape(num_days, stock_num, -1).astype(np.float32)
        
        # 增加日期特征到 data_all，保持与 Env 原始顺序一致 [weekday, month_day]
        data_date = df[['weekday', 'month_day']].values.reshape(num_days, stock_num, -1).astype(np.float32)

        if self.exp_type == 'mae':
            cov_data = np.array(df['cov_list'].values.tolist()).reshape(num_days, stock_num, stock_num, stock_num)
            self.data_all = np.concatenate((cov_data[:, 0, :, :].astype(np.float32), data_tech, data_temp, data_date), axis=-1)
            del cov_data
        else:
            # 预转置布局
            self.data_all = data_temp.transpose(1, 0, 2) 

        self.label_all = np.stack([
            df['label_short_term'].values.reshape(num_days, stock_num),
            df['label_long_term'].values.reshape(num_days, stock_num)
        ], axis=0).astype(np.float32)

        self.dates = dates_final
        self.data_close = df['price'].values.reshape(num_days, stock_num)
        self.boarder_start = [dates_final.tolist().index(self.border_dates[i*2]) for i in range(3)]
        self.boarder_end = [dates_final.tolist().index(self.border_dates[i*2+1]) for i in range(3)]

        # 修改 1：彻底销毁庞大的 DataFrame
        self.full_df = df.copy() # 为了兼容性暂时保留，但稍后销毁局部 df
        del df
        gc.collect()
        print(f"Data initialization complete. Shape: {self.data_all.shape}")

    def get_split_df(self, type_str):
        if type_str not in self.type_map:
            raise ValueError(f"Invalid type: {type_str}")
        
        pos = self.type_map[type_str]
        start_date = self.border_dates[pos*2]
        end_date = self.border_dates[pos*2+1]

        # 筛选数据
        temp_df = self.full_df[(self.full_df['date_str'] >= start_date) & (self.full_df['date_str'] <= end_date)].copy()

        # 排序并设置索引，确保 Env 类可以通过 .loc[day] 获取当天所有股票数据
        temp_df = temp_df.sort_values(['date', 'tic'], ignore_index=True)
        temp_df.index = temp_df.date.factorize()[0]

        return temp_df

    def get_split_data(self, type_str):
        if type_str not in self.type_map:
            raise ValueError(f"Invalid type: {type_str}")
        pos = self.type_map[type_str]
        return self.data_all[self.boarder_start[pos] : self.boarder_end[pos]+1]

class DatasetStock_PRED(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE, pred_type='label_short_term'):
        super().__init__()
        pos = stock.type_map[type]
        self.start_pos = max(stock.boarder_start[pos], stock.seq_len - 1)
        self.end_pos = stock.boarder_end[pos] + 1
        
        self.seq_len = stock.seq_len
        self.data = stock.data_all # (Stocks, Days, Feats)
        
        label_idx = stock.pred_type_map[pred_type]
        self.label = stock.label_all[label_idx, self.start_pos : self.end_pos]
        
        # 修正切片位置：temporal 特征在倒数第 12 到 倒数第 2 位
        self.feat_len = len(feature)

    def __getitem__(self, index):
        p = self.start_pos + index
        # 排除最后的 2 列日期特征，取 10 列时序特征
        seq_x = self.data[:, p - self.seq_len + 1 : p + 1, -(self.feat_len + 2) : -2]
        return seq_x, seq_x[:, -1:, :], self.label[index, :]

    def __len__(self):
        return self.end_pos - self.start_pos

class DatasetStock_MAE(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE):
        super().__init__()
        pos = stock.type_map[type]
        self.data = stock.data_all[stock.boarder_start[pos] : stock.boarder_end[pos]+1]
        self.exclude_len = len(feature) + 2 # 排除时序特征和日期特征

    def __getitem__(self, index):
        # 仅返回 Cov + Technical 部分
        return self.data[index, :, :-self.exclude_len]

    def __len__(self):
        return len(self.data)
