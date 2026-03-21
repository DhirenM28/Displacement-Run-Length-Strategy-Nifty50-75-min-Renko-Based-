def calculate_metrics(trades_df):
    """Calculate performance metrics including yearly breakdown."""
    if trades_df.empty:
        return {
            'Total_Trades': 0,
            'Total_PnL': 0,
            'Max_Drawdown': 0,
            'Avg_Points': 0,
            'Win_Rate': 0,
            'PnL_DD_Ratio': 0
        }
    
    trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
    trades_df['Running_Max'] = trades_df['Cumulative_PnL'].cummax()
    trades_df['Drawdown'] = trades_df['Running_Max'] - trades_df['Cumulative_PnL']
    
    total_trades = len(trades_df)
    total_pnl = trades_df['PnL'].sum()
    max_drawdown = trades_df['Drawdown'].max()
    avg_points = total_pnl / total_trades if total_trades > 0 else 0
    winning_trades = (trades_df['PnL'] > 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    pnl_dd_ratio = (total_pnl / max_drawdown) if max_drawdown != 0 else 0
    
    trades_df['Year'] = trades_df['Exit_Datetime'].dt.year
    yearly_pnl = trades_df.groupby('Year')['PnL'].sum().to_dict()
    
    metrics = {
        'Total_Trades': total_trades,
        'Total_PnL': round(total_pnl, 2),
        'Max_Drawdown': round(max_drawdown, 2),
        'Avg_Points': round(avg_points, 2),
        'Win_Rate': round(win_rate, 2),
        'PnL_DD_Ratio': round(pnl_dd_ratio, 2)
    }
    
    for year, pnl in yearly_pnl.items():
        metrics[f'PnL_{year}'] = round(pnl, 2)
    
    return metrics
