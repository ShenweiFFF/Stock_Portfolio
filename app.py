import gradio as gr
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import copy
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import datetime

def plot_cum_returns(data, title):    
    daily_cum_returns = 1 + data.dropna().pct_change()
    daily_cum_returns = daily_cum_returns.cumprod()*100
    fig = px.line(daily_cum_returns, title=title)
    return fig

def generate_random_portfolios(n_portfolios, returns, cov_matrix):
    """生成随机投资组合"""
    n_assets = len(returns)
    results = np.zeros((3, n_portfolios))
    
    for i in range(n_portfolios):
        # 生成随机权重
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        # 计算组合的预期收益和风险
        portfolio_return = np.sum(returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 计算夏普比率（假设无风险利率为2%）
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_std
        
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = sharpe_ratio
        
    return results

def plot_efficient_frontier_and_max_sharpe(mu, S):
    """绘制有效前沿和最优投资组合"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 生成随机投资组合
    results = generate_random_portfolios(2000, mu, S)
    
    # 绘制随机投资组合点
    scatter = ax.scatter(results[0,:], 
                        results[1,:],
                        c=results[2,:],
                        marker='o',
                        s=10,
                        alpha=0.3,
                        cmap='viridis')
    
    # 找到最优夏普比率的投资组合
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    
    # 绘制最优点
    ax.scatter(std_tangent, ret_tangent,
              marker='*',
              color='red',
              s=200,
              label='Maximum Sharpe ratio')
    
    # 生成有效前沿线
    ef_line = EfficientFrontier(mu, S)
    ret_range = np.linspace(mu.min(), mu.max(), 100)
    ef_frontier = []
    
    for ret in ret_range:
        try:
            ef_line.efficient_return(ret)
            std, ret = ef_line.portfolio_performance()[:2]
            ef_frontier.append([std, ret])
        except:
            continue
    
    ef_frontier = np.array(ef_frontier)
    ax.plot(ef_frontier[:,0], ef_frontier[:,1], 'b--', label='Efficient Frontier')
    
    # 添加标签和图例
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    
    # 添加颜色条
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    return fig

def output_results(start_date, end_date, tickers_string):
    tickers = tickers_string.split(',')
    
    # 获取股票数据
    stocks_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # 绘制个股价格
    fig_indiv_prices = px.line(stocks_df, title='Price of Individual Stocks')
    
    # 绘制个股累计收益
    fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
    
    # 计算并绘制相关性矩阵
    corr_df = stocks_df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True, title='Correlation between Stocks')
    
    # 计算预期收益和协方差矩阵
    mu = expected_returns.mean_historical_return(stocks_df)
    S = risk_models.sample_cov(stocks_df)
    
    # 绘制有效前沿
    fig_efficient_frontier = plot_efficient_frontier_and_max_sharpe(mu, S)
    
    # 获取最优权重
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.04)
    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    
    # 格式化输出数据
    expected_annual_return = '{}%'.format((expected_annual_return*100).round(2))
    annual_volatility = '{}%'.format((annual_volatility*100).round(2))
    sharpe_ratio = '{}%'.format((sharpe_ratio*100).round(2))
    
    # 创建权重数据框
    weights_df = pd.DataFrame.from_dict(weights, orient='index')
    weights_df = weights_df.reset_index()
    weights_df.columns = ['Tickers', 'Weights']
    
    # 计算优化后的投资组合收益
    stocks_df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight
    
    # 绘制优化后的投资组合累计收益
    fig_cum_returns_optimized = plot_cum_returns(
        stocks_df['Optimized Portfolio'],
        'Cumulative Returns of Optimized Portfolio Starting with $100'
    )
    
    return (fig_cum_returns_optimized, weights_df, fig_efficient_frontier, fig_corr,
            expected_annual_return, annual_volatility, sharpe_ratio, fig_indiv_prices, fig_cum_returns)

# Gradio界面部分
with gr.Blocks() as app:
    with gr.Row():
        gr.HTML("<h1>Bohmian's Stock Portfolio Optimizer</h1>")
    
    with gr.Row():
        start_date = gr.Textbox("2013-01-01", label="Start Date")
        end_date = gr.Textbox(datetime.datetime.now().date(), label="End Date")
    
    with gr.Row():        
        tickers_string = gr.Textbox("MA,META,V,AMZN,JPM,BA", 
                                    label='Enter all stock tickers to be included in portfolio separated \
                                    by commas WITHOUT spaces, e.g. "MA,META,V,AMZN,JPM,BA"')
        btn = gr.Button("Get Optimized Portfolio")
       
    with gr.Row():
        gr.HTML("<h3>Optimizied Portfolio Metrics</h3>")
        
    with gr.Row():
        expected_annual_return = gr.Text(label="Expected Annual Return")
        annual_volatility = gr.Text(label="Annual Volatility")
        sharpe_ratio = gr.Text(label="Sharpe Ratio")            
   
    with gr.Row():        
        fig_cum_returns_optimized = gr.Plot(label="Cumulative Returns of Optimized Portfolio (Starting Price of $100)")
        weights_df = gr.DataFrame(label="Optimized Weights of Each Ticker")
        
    with gr.Row():
        fig_efficient_frontier = gr.Plot(label="Efficient Frontier")
        fig_corr = gr.Plot(label="Correlation between Stocks")
    
    with gr.Row():
        fig_indiv_prices = gr.Plot(label="Price of Individual Stocks")
        fig_cum_returns = gr.Plot(label="Cumulative Returns of Individual Stocks Starting with $100")

    btn.click(fn=output_results, 
             inputs=[start_date, end_date, tickers_string],
             outputs=[fig_cum_returns_optimized, weights_df, fig_efficient_frontier, fig_corr,
                     expected_annual_return, annual_volatility, sharpe_ratio, 
                     fig_indiv_prices, fig_cum_returns])

app.launch()