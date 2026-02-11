import os
import sys
import pandas as pd
import numpy as np

# 将当前目录添加到路径以便导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils.yahoodownloader import YahooDownloader
from utils import config

def main():
    # 1. 确定目标目录 (项目根目录/data/CSI_new)
    project_root = os.path.dirname(current_dir)
    target_dir = os.path.join(project_root, "data", "CSI_new")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目录: {target_dir}")

    # 2. 读取配置
    ticker_list = config.USE_CSI_300_TICKET
    start_date = config.START_DATE
    # 将end_date设置为当天日期
    end_date = config.END_DATE

    print(f"准备下载 {len(ticker_list)} 只股票的数据...")
    print(f"时间范围: {start_date} 至 {end_date}")

    # 初始化计数器
    success_count = 0
    failure_count = 0

    for tic in ticker_list:
        try:
            # 实例化下载器（逐个下载以便更好地处理错误）
            print(f"正在处理{success_count+failure_count+1}: {tic}")
            downloader = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[tic])
            df = downloader.fetch_data()

            if df is None or df.empty:
                print(f"警告: 无法获取 {tic} 的数据，跳过。")
                failure_count += 1
                continue

            # 3. 计算模型所需的额外列 (保持与 data/CSI 老文件格式一致)
            # price 列在老文件中存的是 Adjusted Close
            df['price'] = df['close']

            # 计算每日差分 (X_t - X_{t-1})
            # 包含: dopen, dclose, dhigh, dlow, dvolume
            diff_cols = ['open', 'close', 'high', 'low', 'volume']
            for col in diff_cols:
                d_col = 'd' + col
                df[d_col] = df[col].diff()
                # 第一行差分设为原值 (参考老文件 data/CSI/000001.SZ.csv 的做法)
                if len(df) > 0:
                    df.loc[0, d_col] = df.loc[0, col]

            # 4. 整理列顺序
            # 格式: ,date,open,close,high,low,volume,dopen,dclose,dhigh,dlow,dvolume,price
            final_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price']
            df_final = df[final_cols]

            # 5. 保存到 CSV
            output_path = os.path.join(target_dir, f"{tic}.csv")
            # index=True 会生成第一列无名索引，匹配老文件格式
            df_final.to_csv(output_path, index=True)
            
            success_count += 1

        except Exception as e:
            print(f"处理 {tic} 时出错: {str(e)}")
            failure_count += 1
    
    # 输出统计结果
    print(f"\n下载完成！成功: {success_count} 只股票，失败: {failure_count} 只股票")

if __name__ == "__main__":
    main()
