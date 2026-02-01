import os
import sys
import numpy as np
import pandas as pd

import pdb
import torch
from doctest import testfile
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from utils.preprocess import FeatureEngineer, data_split
from utils import config
import datetime



class Stock_Data():
    def __init__(self, root_path, dataset_name, full_stock_path, size, attr = config.TECHNICAL_INDICATORS_LIST, temporal_feature = config.TEMPORAL_FEATURE, scale=True, prediction_len=[2,5]):
        # size [seq_len, label_len, pred_len]
        self.scale = scale
        self.attr = attr
        self.temporal_feature = temporal_feature
        self.root_path = root_path
        self.full_stock = full_stock_path
        self.ticker_list = config.use_ticker_dict[dataset_name]
        self.border_dates = config.date_dict[dataset_name]
        self.prediction_len = prediction_len

        self.seq_len = size[0] # seq_len
        self.type_map = {'train':0, 'valid':1, 'test':2}
        self.pred_type_map = {'label_short_term':0, 'label_long_term':1}


        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        stock_num = len(self.ticker_list)

        full_stock_dir = os.path.join(self.root_path, self.full_stock)

        df = pd.DataFrame([], columns=['date','open','close','high','low','volume','dopen','dclose','dhigh','dlow','dvolume', 'price', 'tic'])
        for ticket in self.ticker_list:
            temp_df = pd.read_csv(os.path.join(full_stock_dir,ticket+'.csv'), usecols=['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price'])

            temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df['label_short_term'] = temp_df['close'].pct_change(periods=self.prediction_len[0]).shift(periods=(-1*self.prediction_len[0]))
            temp_df['label_long_term'] = temp_df['close'].pct_change(periods=self.prediction_len[1]).shift(periods=(-1*self.prediction_len[1]))
            temp_df['tic'] = ticket
            df = pd.concat((df, temp_df))
        df = df.sort_values(by=['date','tic'])

        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

        print("generate technical indicator...")
        df = fe.preprocess_data(df)

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
        return_pivot = price_pivot.pct_change().dropna()

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

        # 处理技术指标中的无穷值
        df[self.attr] = df[self.attr].replace([np.inf], config.INF)
        df[self.attr] = df[self.attr].replace([-np.inf], config.INF*(-1))

        if self.scale:
            # 【统一修改】标准化只在训练集上 fit，避免测试集信息泄露
            scaler = StandardScaler()
            scaler_temporal = StandardScaler() # 新增：用于 temporal_feature 的归一化器

            # 获取训练集的范围
            train_mask = (df['date_str'] >= self.border_dates[0]) & (df['date_str'] <= self.border_dates[1])
            
            # 1. 归一化技术指标
            train_data_for_scaler = df.loc[train_mask, self.attr]
            scaler.fit(train_data_for_scaler.values)
            data = scaler.transform(df[self.attr].values)

            # 2. 【关键修复】归一化时序特征 (Open, Close, High, Low, Volume ...)
            # 必须归一化，否则不同时间窗口的价格绝对值差异会导致分布漂移
            train_temporal_for_scaler = df.loc[train_mask, self.temporal_feature]
            scaler_temporal.fit(train_temporal_for_scaler.values)
            feature_list = scaler_temporal.transform(df[self.temporal_feature].values)
            
        else:
            data = df[self.attr].values
            feature_list = np.array(df[self.temporal_feature].values.tolist())

        cov_list = np.array(df['cov_list'].values.tolist()) # [stock_num*len, stock_num]
        # feature_list 已经在上面处理过了
        close_list = np.array(df['price'].values.tolist())

        # pdb.set_trace()
        data_cov = cov_list.reshape(-1, stock_num, cov_list.shape[1], cov_list.shape[2]) # [day, num_stocks, num_stocks, num_stocks]
        data_technical = data.reshape(-1, stock_num, len(self.attr)) # [day, stock_num, technical_len]
        data_feature = feature_list.reshape(-1, stock_num, len(self.temporal_feature)) # [day, stock_num, temporal_feature_len=10]
        data_close = close_list.reshape(-1, stock_num)

        label_short_term = np.array(df['label_short_term'].values.tolist()).reshape(-1, stock_num)
        label_long_term = np.array(df['label_long_term'].values.tolist()).reshape(-1, stock_num)

        self.data_all = np.concatenate((data_cov[:, 0, :, :], data_technical, data_feature), axis=-1) # [days, num_stocks, cov+technical_len+feature_len]
        self.label_all = np.stack((label_short_term, label_long_term), axis=0) # [2, days, num_stocks, 1]
        self.dates = np.array(dates)
        self.data_close = data_close
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