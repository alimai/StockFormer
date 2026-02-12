import os
import sys
import numpy as np
import pandas as pd

import pdb
import torch
from doctest import testfile
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from utils.preprocess import FeatureEngineer, data_split
from utils import config
import datetime



class Stock_Data():
    def __init__(self, full_stock_path, size, attr = config.TECHNICAL_INDICATORS_LIST, temporal_feature = config.TEMPORAL_FEATURE, scale=True, prediction_len=[2,5]):
        # size [seq_len, label_len, pred_len]
        self.scale = scale
        self.attr = attr
        self.temporal_feature = temporal_feature
        self.full_stock_dir = full_stock_path
        self.ticker_list = config.USE_CSI_300_TICKET
        self.border_dates = config.CSI_date_trans
        self.prediction_len = prediction_len

        self.seq_len = size[0] # seq_len
        self.type_map = {'train':0, 'valid':1, 'test':2}
        self.pred_type_map = {'label_short_term':0, 'label_long_term':1}


        self.__read_data__()

    def __read_data__(self):
        scaler = MinMaxScaler()
        stock_num = len(self.ticker_list)

        df = pd.DataFrame([], columns=['date','open','close','high','low','volume','dopen','dclose','dhigh','dlow','dvolume', 'price', 'tic'])
        # Track which stocks were successfully loaded
        successful_tickers = []
        for ticket in self.ticker_list:
            temp_df = pd.read_csv(os.path.join(self.full_stock_dir,ticket+'.csv'), usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])

            # 【关键修复】确保所有可能归一化的数值列在数据加载后立即转换为浮点类型
            numeric_cols_to_float = ['open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price']
            for col in numeric_cols_to_float:
                if col in temp_df.columns:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').astype(float)
                    # 统一逻辑：线性插值 -> 外推 -> 赋零
                    temp_df[col] = temp_df[col].interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0)

            temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df['label_short_term'] = temp_df['close'].pct_change(periods=self.prediction_len[0]).shift(periods=(-1*self.prediction_len[0]))
            temp_df['label_long_term'] = temp_df['close'].pct_change(periods=self.prediction_len[1]).shift(periods=(-1*self.prediction_len[1]))
            temp_df['tic'] = ticket
            
            # 【核心新增】检查每只股票的原始数据列是否全空
            raw_cols_to_check = ['open', 'close', 'high', 'low', 'volume']
            for col in raw_cols_to_check:
                if temp_df[col].isna().all():
                    print(f"Error: Column '{col}' for ticker '{ticket}' is entirely empty in CSV. Program exiting.")
                    sys.exit(1)

            df = pd.concat((df, temp_df))
            successful_tickers.append(ticket)
        df = df.sort_values(by=['date','tic'])
        df['date'] = pd.to_datetime(df['date'])

        # Update stock_num to reflect the actual number of stocks loaded
        stock_num = len(successful_tickers)
        print(f"Successfully loaded {stock_num} stocks: {successful_tickers[:5]}{'...' if len(successful_tickers) > 5 else ''}")

        # Add time features
        # month_day: 12.15 for Dec 15th
        df['month_day'] = df['date'].dt.month + df['date'].dt.day / 100.0
        # weekday: 0 for Monday, ..., 6 for Sunday
        df['weekday'] = df['date'].dt.dayofweek

        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

        print("generate technical indicator...")
        df = fe.preprocess_data(df)

        # 【核心修复】强制对齐所有日期和股票，确保每个日期都有 stock_num 行
        # 1. 提取所有唯一的日期和股票代码
        all_dates = sorted(df['date'].unique())
        all_tics = successful_tickers
        
        # 2. 创建一个完整的 MultiIndex (date, tic)
        full_index = pd.MultiIndex.from_product([all_dates, all_tics], names=['date', 'tic'])
        
        # 3. 重新索引 df
        df = df.set_index(['date', 'tic']).reindex(full_index).reset_index()
        
        # 【核心新增】在补全前检查是否存在全空列
        fill_cols = [col for col in df.columns if col not in ['date', 'tic']]
        for tic, tic_df in df.groupby('tic'):
            for col in fill_cols:
                if tic_df[col].isna().all():
                    print(f"Error: Column '{col}' for ticker '{tic}' is entirely empty after alignment but before filling. Program exiting.")
                    sys.exit(1)

        # 4. 填充缺失值
        # 统一逻辑：线性插值 -> 外推 -> 赋零
        fill_cols = [col for col in df.columns if col not in ['date', 'tic']]
        # 按股票分组填充，防止股票间数据污染
        df[fill_cols] = df.groupby('tic')[fill_cols].transform(lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0))
        
        # # 【关键修复】确保所有技术指标和时序特征都是数值类型，防止产生 object 数组
        # for col in self.attr + self.temporal_feature:
        #     if col in df.columns:
        #         df[col] = pd.to_numeric(df[col], errors='coerce')

        # add covariance matrix as states
        df=df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []
        return_list = []

        # look back is one year
        print("generate convariate matrix...")
        lookback=252

        # 参照 preprocess.py 的方式：先 pivot 整个数据，再使用日期索引筛选
        price_pivot = df.pivot_table(index='date', columns='tic', values='close')
        # 【修复】使用 fillna(0) 代替 dropna()，防止因为一只股票缺失数据导致全场日期被删
        return_pivot = price_pivot.pct_change().fillna(0)

        unique_date = df.date.unique()
        for i in range(lookback, len(unique_date)):
            # 使用日期索引筛选，排除当天数据（只使用历史数据计算协方差）
            # 注意：由于 pct_change().dropna() 删除了第一行，return_pivot.index 的第一个日期是 unique_date[1]
            # 所以窗口起始日期需要调整为 unique_date[i-lookback+1] 以匹配 dropna() 后的索引
            # 窗口范围：[unique_date[i-lookback+1], unique_date[i-1]]，共 252 个数据点
            return_lookback = return_pivot[
                (return_pivot.index < unique_date[i])
                & (return_pivot.index >= unique_date[i - lookback + 1])  # +1 以匹配 dropna() 后的索引
            ]
            return_list.append(return_lookback)

            covs = return_lookback.cov().values
            cov_list.append(covs)

        df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)
        df['date_str'] = df['date'].apply(lambda x: datetime.datetime.strftime(x,'%Y%m%d'))

        dates = df['date_str'].unique().tolist()
        boarder1_ = dates.index(self.border_dates[0])
        boarder1 = dates.index(self.border_dates[1])

        boarder2_ = dates.index(self.border_dates[2])
        boarder2 = dates.index(self.border_dates[3])

        boarder3_ = dates.index(self.border_dates[4])
        boarder3 = dates.index(self.border_dates[5])

        self.boarder_end = [boarder1, boarder2, boarder3]
        self.boarder_start = [boarder1_, boarder2_, boarder3_]

        # 检查并过滤掉不存在的列，但为了模型维度，必须确保 self.attr 的列数与 config 一致
        original_attr_list = config.TECHNICAL_INDICATORS_LIST
        for col in original_attr_list:
            if col not in df.columns:
                print(f"Warning: Technical indicator column '{col}' not found in data. Filling with zeros.")
                sys.exit(1)#df[col] = 0.0  # 缺失列
        
        self.attr = original_attr_list
        print(f"Attributes used (aligned to config): {self.attr}")

        # 【关键修复】统一逻辑：填充所有 NaN 为 0，作为最后一道防线
        #df[self.attr] = df[self.attr].fillna(0)
        #df[self.temporal_feature] = df[self.temporal_feature].fillna(0)

        # 处理技术指标中的无穷值
        df[self.attr] = df[self.attr].replace([np.inf], config.INF)
        df[self.attr] = df[self.attr].replace([-np.inf], config.INF*(-1))

        # After trimming the df to match the cov matrix dates, we need to ensure all data arrays have the same number of days
        # Extract data after trimming to ensure consistent dimensions
        # Process data after df has been trimmed to match cov matrix dates
        if self.scale:
            # 【统一修改】标准化只在训练集上 fit，避免测试集信息泄露
            scaler = MinMaxScaler()

            # 获取训练集的范围
            train_mask = (df['date_str'] >= self.border_dates[0]) & (df['date_str'] <= self.border_dates[1])
            
            # 1. 归一化技术指标 (Tech Indicators) - 保持原有逻辑
            train_data_for_scaler = df.loc[train_mask, self.attr]
            scaler.fit(train_data_for_scaler.values)
            data = scaler.transform(df[self.attr].values)

            # 2. 归一化时序特征 (Temporal Features) - 方案 C
            # 定义分组
            price_abs_cols = ['open', 'close', 'high', 'low']
            price_diff_cols = ['dopen', 'dclose', 'dhigh', 'dlow']
            vol_abs_cols = ['volume']
            vol_diff_cols = ['dvolume']

            # 筛选当前存在的列
            valid_price_abs = [c for c in price_abs_cols if c in self.temporal_feature]
            valid_price_diff = [c for c in price_diff_cols if c in self.temporal_feature]
            valid_vol_abs = [c for c in vol_abs_cols if c in self.temporal_feature]
            valid_vol_diff = [c for c in vol_diff_cols if c in self.temporal_feature]

            # --- Group A: 价格组 ---
            if valid_price_abs:
                # 仅使用训练集数据的绝对价格计算参数 (只缩放，不平移)
                train_price_vals = df.loc[train_mask, valid_price_abs].values.flatten()
                max_price = np.max(train_price_vals) + 1e-8
                
                # 1. 绝对价格：X / Max -> 保持比例
                df.loc[:, valid_price_abs] = df[valid_price_abs] / max_price
                
                # 2. 差分价格：也使用相同的 Max 缩放
                if valid_price_diff:
                    df.loc[:, valid_price_diff] = df[valid_price_diff] / max_price

            # --- Group B: 成交量组 ---
            if valid_vol_abs:
                # 仅使用训练集数据的绝对成交量计算参数
                train_vol_vals = df.loc[train_mask, valid_vol_abs].values.flatten()
                max_vol = np.max(train_vol_vals) + 1e-8
                
                # 1. 绝对成交量：X / Max
                df.loc[:, valid_vol_abs] = df[valid_vol_abs] / max_vol
                
                # 2. 差分成交量：也使用相同的 Max 缩放
                if valid_vol_diff:
                    df.loc[:, valid_vol_diff] = df[valid_vol_diff] / max_vol

            # --- Group C: 时间特征组 (缩放到7个台阶) ---
            df['month_day'] = (df['month_day']/2).astype(int) / 7.0
            df['weekday'] = df['weekday'] / 7.0

            # 提取最终归一化后的特征矩阵
            feature_list = df[self.temporal_feature].values
            
        else:
            data = df[self.attr].values
            feature_list = np.array(df[self.temporal_feature].values.tolist())

        cov_list = np.array(df['cov_list'].values.tolist()) # [stock_num*len, stock_num]
        # feature_list 已经在上面处理过了
        close_list = np.array(df['price'].values.tolist())
        month_day_list = np.array(df['month_day'].values.tolist())
        weekday_list = np.array(df['weekday'].values.tolist())

        # pdb.set_trace()
        # Check the actual shape of the cov_list to determine how to reshape it properly
        print(f"Shape of cov_list: {cov_list.shape}")
        print(f"Expected stock_num: {stock_num}")
        
        data_cov = cov_list.reshape(-1, stock_num, cov_list.shape[1], cov_list.shape[2]) # [day, num_stocks, num_stocks, num_stocks]
        data_technical = data.reshape(-1, stock_num, len(self.attr)) # [day, stock_num, technical_len]
        data_feature = feature_list.reshape(-1, stock_num, len(self.temporal_feature)) # [day, stock_num, temporal_feature_len=10]
        data_close = close_list.reshape(-1, stock_num)
        data_month_day = month_day_list.reshape(-1, stock_num)
        data_weekday = weekday_list.reshape(-1, stock_num)

        label_short_term = np.array(df['label_short_term'].values.tolist()).reshape(-1, stock_num)
        label_long_term = np.array(df['label_long_term'].values.tolist()).reshape(-1, stock_num)

        # 显式转换为 float32 确保类型统一
        data_cov_part = data_cov[:, 0, :, :].astype(np.float32)
        data_technical_part = data_technical.astype(np.float32)
        data_feature_part = data_feature.astype(np.float32)

        self.data_all = np.concatenate((data_cov_part, data_technical_part, data_feature_part), axis=-1) # [days, num_stocks, cov+technical_len+feature_len]
        self.label_all = np.stack((label_short_term, label_long_term), axis=0).astype(np.float32) # [2, days, num_stocks, 1]
        self.dates = np.array(dates)
        self.data_close = data_close
        self.data_month_day = data_month_day
        self.data_weekday = data_weekday
        self.full_df = df # 【新增】保存处理好的全量 DataFrame

        print("data shape: ",self.data_all.shape)
        print("label shape: ",self.label_all.shape)

    def get_split_df(self, type='train'):
        # 【新增】获取对应阶段的 DataFrame 并重置索引
        pos = self.type_map[type]
        start_date = self.border_dates[pos*2]
        end_date = self.border_dates[pos*2+1]
        
        # 筛选数据
        temp_df = self.full_df[(self.full_df['date_str'] >= start_date) & (self.full_df['date_str'] <= end_date)]
        
        # 【关键修复】重置索引，确保每个数据集（Train/Eval/Test）的索引都从 0 开始
        # 这样环境中的 time_window_start 才能正确匹配
        temp_df = temp_df.sort_values(['date', 'tic'], ignore_index=True)
        temp_df.index = temp_df.date.factorize()[0]
        
        return temp_df



class DatasetStock_MAE(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE, pred_type=None):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        pos = stock.type_map[type]
        self.feature_len = len(feature)

        self.data = stock.data_all[stock.boarder_start[pos]: stock.boarder_end[pos]+1]
        self.label = stock.label_all[:, stock.boarder_start[pos]: stock.boarder_end[pos]+1]

        # pdb.set_trace()

    def __getitem__(self, index):
        seq_x = self.data[index, :, :-self.feature_len]
        return seq_x

    def __len__(self):
        return len(self.data)


class DatasetStock_PRED(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE, pred_type='label_short_term'):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        assert pred_type in ['label_short_term', 'label_long_term']
        print(pred_type)
        pos = stock.type_map[type]

        self.label_type = stock.pred_type_map[pred_type]
        self.start_pos = stock.boarder_start[pos]
        self.end_pos = stock.boarder_end[pos]+1
        print(self.start_pos, self.end_pos)

        self.feature_len = len(feature)
        self.feature_day_len = stock.seq_len
        self.data = stock.data_all
        self.label = stock.label_all

        self.dates = stock.dates[self.start_pos: self.end_pos]
        self.data_close = stock.data_close[self.start_pos: self.end_pos]

        # pdb.set_trace()

    def __getitem__(self, index):
        position = self.start_pos+index
        # Extract the temporal features (last self.feature_len columns) for the sequence length
        seq_x = self.data[position-self.feature_day_len+1:position+1, :, -self.feature_len:]
        # Extract the cov + technical features (first 96 columns) for the sequence length
        #seq_x = self.data[position-self.feature_day_len+1:position+1, :, :-self.feature_len]
        
        # Transpose to (Stocks, Time, Feats)
        seq_x = seq_x.transpose(1, 0, 2)
        seq_x_dec = seq_x[:, -1:, :]  # Take the last time step for decoder input

        seq_y = self.label[self.label_type, index, :]
        return seq_x, seq_x_dec, seq_y

    def __len__(self):
        return self.end_pos-self.start_pos#len(self.data)




class DatasetStock(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        pos = stock.type_map[type]

        self.start_pos = stock.boarder_start[pos]
        self.end_pos = stock.boarder_end[pos]+1
        print(self.start_pos, self.end_pos)

        self.feature_len = len(feature)
        self.feature_day_len = stock.seq_len
        self.data = stock.data_all
        self.label = stock.label_all

        # pdb.set_trace()

    def __getitem__(self, index):
        position = self.start_pos+index
        data1 = self.data[position, :, :-self.feature_len] #[num_stocks, cov+technical]
        data2 = self.data[position-self.feature_day_len+1:position+1, :, -self.feature_len:].transpose(1,0,2) #[days, num_stocks, feature]-> [num_stocks, days, feature]

        label1 = self.label[0, index, :]
        label2 = self.label[1, index, :]
        return data1, data2, label1, label2

    def __len__(self):
        return self.end_pos-self.start_pos