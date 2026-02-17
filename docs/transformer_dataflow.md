根据工程中的 train_tf.py 及相关代码（exp_mae.py, exp_pred.py, stock_data_handle.py, config.py），exp_type="pred" 和 exp_type="mae"

✦ 一. 两种模式对应的训练数据内容：

  1. exp_type="mae" (Masked Auto-Encoder 任务)

  该模式主要用于训练模型学习股票间的关联和技术面特征，数据主要包含协方差信息和技术指标。

   * 输入数据特征 (`enc_in` ≈ 96维):
       * 协方差特征 (Covariance Matrix, 约88维): 当前股票与其自身及其他所有股票在过去一年（252天）收益率的协方差。这部分特征捕获了股票间的相关性。
       * 技术指标 (Technical Indicators, 8维): 包含以下计算好的技术指标：
           1. macd
           2. boll_ub (布林带上轨)
           3. boll_lb (布林带下轨)
           4. rsi_30 (30日相对强弱指标)
           5. cci_30 (30日顺势指标)
           6. dx_30 (30日动向指标)
           7. close_30_sma (30日收盘均价)
           8. close_60_sma (60日收盘均价)
   * 注意: 该模式明确排除了开盘价、收盘价等基础量价时序特征（即排除 pred 模式用的那些特征）。

  2. exp_type="pred" (Prediction 预测任务)

  该模式用于股价趋势预测（短期或长期），主要使用基础的量价时序特征。

   * 输入数据特征 (`enc_in` = 10维):
      仅包含 TEMPORAL_FEATURE 定义的10个基础量价特征：
       1. open (开盘价)
       2. close (收盘价)
       3. high (最高价)
       4. low (最低价)
       5. volume (成交量)
       6. dopen (开盘价变化率)
       7. dclose (收盘价变化率)
       8. dhigh (最高价变化率)
       9. dlow (最低价变化率)
       10. dvolume (成交量变化率)

   * 预测目标 (Labels):
       * 短期: label_short_term (通常为未来 1 天的收益率，short_term_len=1)
       * 长期: label_long_term (通常为未来 5 天的收益率，long_term_len=5)
       * (具体取决于配置中的 pred_type 参数)

  总结：
   * MAE 侧重于关系与状态（协方差+技术面）。
   * PRED 侧重于量价趋势（OHLCV+变化率）。


✦ 二. 两种模式在“时间维度”上的输入逻辑：

  1. exp_type="mae" (Masked Auto-Encoder)
  输入：只有“当天”的数据。
   * 逻辑：它不是在时间上做滑动窗口，而是在股票维度上做建模。
   * 输入形状：每一批次输入的是 [Batch_Size, Stock_Num, Features]。
   * 做法：它会随机遮掩（Mask）一部分股票的特征，然后让模型利用其他股票在同一天的特征（以及它们之间的协方差关系）来预测被遮掩股票的特征。
   * 目的：学习股票之间的相关性（Cross-sectional relationship）。

  2. exp_type="pred" (Prediction)
  输入：过去 `seq_len` 天的数据（默认 60 天）。
   * 逻辑：这是标准的时间序列预测模式。
   * 输入形状：对于每一支股票，它会取过去连续的 seq_len 天数据。
   * 计算过程：
       1. 从 Stock_Data 中提取 [seq_len, Stock_Num, 10] 的数据块。
       2. 转置为 [Stock_Num, seq_len, 10]。
       3. 在训练时进一步展平，让模型处理 [Batch_Size * Stock_Num, seq_len, 10]。
   * 目的：学习单支股票自身的历史价格走势（Temporal patterns）来预测未来的收益率。

  ---

  简而言之：
   * MAE 模式是“横截面”视角：看今天 A 股票和 B 股票的关系。
   * PRED 模式是“时序”视角：看 A 股票过去 60 天的走势。