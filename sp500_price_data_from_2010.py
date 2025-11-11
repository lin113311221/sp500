#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取2010年1月1日开始的S&P 500股票价格数据

从2010年1月1日的S&P 500股票池开始，获取每个股票从2010年1月1日到现在的日线价格数据（close和adjusted close）。
如果股票中途被移除S&P 500，则只获取到它仍在S&P 500期间的数据。
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import os

# ============================================================================
# 配置参数
# ============================================================================

# 设置起始日期
START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')  # 当前日期

# ============================================================================
# 辅助函数
# ============================================================================

def get_table(filename):
    """读取CSV文件"""
    df = pd.read_csv(filename)
    df.index = pd.to_datetime(df.index)
    return df

def convert_tickers_to_list(tickers_str):
    """将逗号分隔的ticker字符串转换为列表"""
    if pd.isna(tickers_str) or not isinstance(tickers_str, str):
        return []
    return [t.strip() for t in tickers_str.split(',') if t.strip()]

def get_stock_price(ticker, start_date, end_date):
    """
    获取股票价格数据

    参数:
    ticker: 股票代码
    start_date: 开始日期
    end_date: 结束日期（如果为None，则获取到当前）

    返回:
    DataFrame包含Close和Adj Close列
    """
    # 处理yfinance的特殊股票代码（如BRK.B需要转换为BRK-B）
    yf_ticker = ticker.replace('.', '-')

    # 确定实际的结束日期
    if end_date is None:
        actual_end = END_DATE
    elif isinstance(end_date, pd.Timestamp):
        actual_end = (end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        actual_end = end_date

    # 下载数据
    stock = yf.Ticker(yf_ticker)
    hist = stock.history(start=start_date, end=actual_end)

    if hist.empty:
        raise ValueError(f"[{ticker}] 获取到的数据为空")

    # 检查必需的列是否存在
    required_cols = ['Close', 'Adj Close']
    missing_cols = [col for col in required_cols if col not in hist.columns]

    if missing_cols:
        # 如果缺少Adj Close，尝试使用Close代替
        if 'Adj Close' in missing_cols and 'Close' in hist.columns:
            hist['Adj Close'] = hist['Close']
        else:
            raise ValueError(f"[{ticker}] 缺少必需的列: {missing_cols}")

    # 只保留Close和Adj Close列
    result = hist[['Close', 'Adj Close']].copy()
    result.columns = ['Close', 'Adj Close']
    result.index.name = 'Date'
    result.reset_index(inplace=True)

    # 如果指定了结束日期，过滤掉结束日期之后的数据
    if end_date is not None:
        if isinstance(end_date, pd.Timestamp):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = end_date
        result = result[result['Date'] <= end_date_str]

    return result

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""

    # 1. 读取历史成分股数据文件
    filename = 'S&P 500 Historical Components & Changes(11-09-2025).csv'
    df = get_table(filename)
    df.index = pd.to_datetime(df.index)

    # 2. 获取2010年1月1日的S&P 500股票列表
    target_date = pd.to_datetime(START_DATE)
    df_before_start = df[df.index <= target_date]

    if len(df_before_start) == 0:
        raise ValueError(f"无法找到 {START_DATE} 之前的数据")

    last_row_before_start = df_before_start.tail(1)
    tickers_str = last_row_before_start['tickers'].iloc[0]

    if pd.isna(tickers_str) or not isinstance(tickers_str, str) or len(tickers_str.strip()) == 0:
        raise ValueError(f"在 {last_row_before_start.index[0]} 的tickers数据无效")

    tickers_on_start_date = sorted([t.strip() for t in tickers_str.split(',') if t.strip()])

    # 3. 将tickers列转换为列表格式
    df['tickers'] = df['tickers'].apply(convert_tickers_to_list)

    # 确定每个股票在S&P 500中的结束日期
    ticker_end_dates = {}
    df_after_start = df[df.index >= target_date].sort_index()

    for ticker in tqdm(tickers_on_start_date, desc="确定股票移除日期"):
        end_date = None
        was_in_sp500 = True

        for date, row in df_after_start.iterrows():
            is_in_current = ticker in row['tickers']

            if was_in_sp500 and not is_in_current:
                end_date = date
                break

            was_in_sp500 = is_in_current

        ticker_end_dates[ticker] = end_date

    # 4. 批量获取所有股票的价格数据
    all_stock_data = {}

    for ticker in tqdm(tickers_on_start_date, desc="下载股票数据"):
        end_date = ticker_end_dates[ticker]

        # 获取价格数据
        price_data = get_stock_price(ticker, START_DATE, end_date)
        price_data['Ticker'] = ticker
        all_stock_data[ticker] = price_data

    # 5. 合并所有股票数据到一个DataFrame
    combined_df = pd.concat(all_stock_data.values(), ignore_index=True)
    combined_df = combined_df[['Ticker', 'Date', 'Close', 'Adj Close']]
    combined_df = combined_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # 6. 保存数据到CSV文件
    output_filename = 'sp500_prices_from_2010.csv'
    combined_df.to_csv(output_filename, index=False)

    return len(all_stock_data)


if __name__ == "__main__":
    success_count = main()
    print(f"成功获取 {success_count} 个股票的数据")