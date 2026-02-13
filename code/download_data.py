import os
import sys
import pandas as pd
import numpy as np

from utils.yahoodownloader import YahooDownloader
from utils import config

# 将当前目录添加到路径以便导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    # 1. 确定目标目录 (项目根目录/data/CSI)
    project_root = os.path.dirname(current_dir)
    target_dir = os.path.join(project_root, "data", config.version_name)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目录: {target_dir}")

    # 2. 读取配置
    ticker_list = config.CSI_300_TICKET_download
    start_date = config.START_DATE
    end_date = config.END_DATE

    print(f"准备下载 {len(ticker_list)} 只股票的数据...")
    print(f"时间范围: {start_date} 至 {end_date}")

    # 初始化计数器和记录列表
    success_count = 0
    failure_count = 0
    failed_tickers = []  # 记录下载失败的股票代码
    success_data_info = {}  # 记录成功下载的股票数据信息 {'ticker': {'length': int, 'start_date': str, 'end_date': str}}

    for tic in ticker_list:
        try:
            # 实例化下载器（逐个下载以便更好地处理错误）
            print(f"正在处理{success_count+failure_count+1}: {tic}")
            downloader = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[tic])
            df = downloader.fetch_data()

            if df is None or df.empty:
                print(f"警告: 无法获取 {tic} 的数据，跳过。")
                failure_count += 1
                failed_tickers.append(tic)
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

            # 记录成功下载的股票数据信息
            if not df_final.empty:
                date_col = df_final['date']
                success_data_info[tic] = {
                    'length': len(df_final),
                    'start_date': date_col.min(),
                    'end_date': date_col.max()
                }

            success_count += 1

        except Exception as e:
            print(f"处理 {tic} 时出错: {str(e)}")
            failure_count += 1
            failed_tickers.append(tic)

    # 输出统计结果
    print(f"\n下载完成！成功: {success_count} 只股票，失败: {failure_count} 只股票")

    # 输出下载异常股票代码列表
    if failed_tickers:
        print(f"\n下载异常股票代码列表 ({len(failed_tickers)} 只):")
        for i, ticker in enumerate(failed_tickers, 1):
            print(f"  {i}. {ticker}")
    else:
        print("\n所有股票均下载成功！")

    # 输出正常下载的股票数据中日期数目最长和最短的股票代码和起止日期
    if success_data_info:
        # 找到日期数目最短和最长的股票
        shortest_ticker = min(success_data_info, key=lambda k: success_data_info[k]['length'])
        longest_ticker = max(success_data_info, key=lambda k: success_data_info[k]['length'])
        shortest_info = success_data_info[shortest_ticker]
        longest_info = success_data_info[longest_ticker]
        
        print(f"\n日期数目最短的股票: {shortest_ticker}")
        print(f"  日期数目: {shortest_info['length']}")
        print(f"  起始日期: {shortest_info['start_date']}")
        print(f"  结束日期: {shortest_info['end_date']}")
        
        print(f"\n日期数目最长的股票: {longest_ticker}")
        print(f"  日期数目: {longest_info['length']}")
        print(f"  起始日期: {longest_info['start_date']}")
        print(f"  结束日期: {longest_info['end_date']}")
    else:
        print("\n没有成功下载任何股票数据。")

if __name__ == "__main__":
    main()
