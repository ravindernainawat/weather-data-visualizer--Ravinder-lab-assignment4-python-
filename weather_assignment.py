

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory
OUT_DIR = '.'

# A public CSV of weather data (Seattle 2016) hosted in Plotly's datasets
DATA_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv'

def download_and_load(url=DATA_URL):
    print('\n-- Downloading and loading dataset')
    try:
        df = pd.read_csv(url)
        print(f'Loaded CSV from URL: {url}')
    except Exception as e:
        print('Error loading URL, trying local fallback...')
        raise
    print('\n--- Data inspection (head)')
    print(df.head())
    print('\n--- Data info')
    print(df.info())
    print('\n--- Data describe (numeric)')
    print(df.describe(include='all'))
    return df

def detect_columns(df):
    cols = { 'date': None, 'temperature': None, 'rainfall': None, 'humidity': None }
    lower = {c: c.lower() for c in df.columns}
    # Date detection
    for c,l in lower.items():
        if 'date' in l or 'time' in l or 'day' in l:
            cols['date'] = c
            break
    # Temperature detection
    for c,l in lower.items():
        if 'temp' in l and 'mean' in l:
            cols['temperature'] = c
            break
    if cols['temperature'] is None:
        for c,l in lower.items():
            if 'temp' in l:
                cols['temperature'] = c
                break
    # Rainfall/precip detection
    for c,l in lower.items():
        if 'precip' in l or 'rain' in l or 'prcp' in l:
            cols['rainfall'] = c
            break
    # Humidity detection
    for c,l in lower.items():
        if 'humid' in l:
            cols['humidity'] = c
            break
    print('\nDetected columns mapping:')
    print(cols)
    return cols


def clean_data(df, cols):
    print('\n-- Cleaning data')
    df_clean = df.copy()
    # Date
    date_col = cols['date']
    if date_col is not None:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        # drop rows without date
        df_clean = df_clean.dropna(subset=[date_col])
    else:
        # if no date, create index-based date
        df_clean['Date'] = pd.date_range(start='2000-01-01', periods=len(df_clean), freq='D')
        date_col = 'Date'
    # Rename to unified names
    rename_map = {}
    if cols['temperature']:
        rename_map[cols['temperature']] = 'Temperature'
    if cols['rainfall']:
        rename_map[cols['rainfall']] = 'Rainfall'
    if cols['humidity']:
        rename_map[cols['humidity']] = 'Humidity'
    if date_col:
        rename_map[date_col] = 'Date'
    df_clean = df_clean.rename(columns=rename_map)
    # Keep only relevant columns plus Date
    keep_cols = ['Date'] + [c for c in ['Temperature','Rainfall','Humidity'] if c in df_clean.columns]
    df_clean = df_clean[keep_cols]
    # Handle numeric casting and NaNs
    for c in df_clean.columns:
        if c != 'Date':
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
            # Fill NaNs: for rainfall fill 0, else fill mean
            if c == 'Rainfall':
                df_clean[c] = df_clean[c].fillna(0.0)
            else:
                df_clean[c] = df_clean[c].fillna(df_clean[c].mean())
    print('\nMissing values after cleaning:')
    print(df_clean.isnull().sum())
    print(f"\nCleaned shape: {df_clean.shape}")
    return df_clean


def compute_statistics(df):
    print('\n-- Computing statistics')
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Daily statistics are just the raw data; compute overall stats
    if not numeric_cols:
        print('No numeric columns found for statistics.')
        return stats, pd.DataFrame(), pd.DataFrame()
    for col in numeric_cols:
        data = df[col].dropna().values
        stats[col] = {
            'mean': float(np.mean(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'std': float(np.std(data))
        }
    # Monthly and yearly aggregations (ensure Date is index)
    df_idx = df.set_index('Date')
    monthly = df_idx[numeric_cols].resample('M').agg(['mean','min','max','std'])
    yearly = df_idx[numeric_cols].resample('Y').agg(['mean','min','max','std'])
    print('\nOverall statistics (by column):')
    for k,v in stats.items():
        print(f"{k}: mean={v['mean']:.2f}, min={v['min']:.2f}, max={v['max']:.2f}, std={v['std']:.2f}")
    return stats, monthly, yearly


def create_visualizations(df, output_prefix='plot'):
    print('\n-- Creating visualizations')
    df_plot = df.copy()
    date = df_plot['Date']
    p1 = p2 = p3 = p4 = None
    # Plot 1: daily temperature
    if 'Temperature' in df_plot.columns:
        plt.figure(figsize=(12,5))
        plt.plot(date, df_plot['Temperature'], color='tab:red', linewidth=1.2)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Daily Temperature Trends')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p1 = os.path.join(OUT_DIR, f'{output_prefix}_daily_temperature.png')
        plt.savefig(p1, dpi=300)
        plt.close()
        print(f'Saved {p1}')
    # Plot 2: monthly rainfall totals
    if 'Rainfall' in df_plot.columns:
        monthly = df_plot.set_index('Date').resample('M')['Rainfall'].sum()
        plt.figure(figsize=(12,5))
        monthly.plot(kind='bar', color='steelblue')
        plt.xlabel('Month')
        plt.ylabel('Total Rainfall (units)')
        plt.title('Monthly Rainfall Totals')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(OUT_DIR, f'{output_prefix}_monthly_rainfall.png')
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f'Saved {p2}')
    # Plot 3: humidity vs temperature scatter
    p3 = None
    if 'Temperature' in df_plot.columns and 'Humidity' in df_plot.columns:
        plt.figure(figsize=(8,6))
        plt.scatter(df_plot['Temperature'], df_plot['Humidity'], alpha=0.6, s=30, color='green')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Humidity (%)')
        plt.title('Humidity vs Temperature')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(OUT_DIR, f'{output_prefix}_humidity_vs_temperature.png')
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f'Saved {p3}')
    # Plot 4: combined figure (temperature line + histogram)
    fig, axes = plt.subplots(2,1, figsize=(12,9))
    if 'Temperature' in df_plot.columns:
        axes[0].plot(date, df_plot['Temperature'], color='tab:red')
        axes[0].set_title('Temperature Over Time')
        axes[0].grid(alpha=0.3)
    if 'Temperature' in df_plot.columns:
        axes[1].hist(df_plot['Temperature'].dropna(), bins=20, color='coral', edgecolor='black')
        axes[1].set_title('Temperature Distribution')
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p4 = os.path.join(OUT_DIR, f'{output_prefix}_combined_subplots.png')
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f'Saved {p4}')
    return p1 if 'Temperature' in df_plot.columns else None, p2, p3, p4


def group_and_aggregate(df):
    print('\n-- Grouping and aggregation')
    df_idx = df.set_index('Date')
    monthly = df_idx.resample('M').agg(['mean','min','max','std'])
    season_map = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
    df_idx['Season'] = df_idx.index.month.map(season_map)
    grouped = df_idx.groupby('Season').agg(['mean','min','max','std'])
    print('\nMonthly head:')
    print(monthly.head())
    print('\nSeasonal aggregation:')
    print(grouped.head())
    return monthly, grouped


def export_and_report(df, stats, monthly, yearly, monthly_agg, seasonal_agg, plots):
    print('\n-- Exporting cleaned data and writing report')
    cleaned_file = os.path.join(OUT_DIR, 'cleaned_weather_real.csv')
    df.to_csv(cleaned_file, index=False)
    print(f'Exported cleaned CSV: {cleaned_file}')
    # Write markdown report
    report_file = os.path.join(OUT_DIR, 'weather_assignment_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('# Weather Data Analysis Report\n\n')
        f.write(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('## Dataset Overview\n')
        f.write(f'- Records (after cleaning): {len(df)}\n')
        f.write(f'- Date range: {df["Date"].min()} to {df["Date"].max()}\n\n')
        f.write('## Key Statistics\n')
        for col, vals in stats.items():
            f.write(f'### {col}\n')
            f.write(f'- Mean: {vals["mean"]:.2f}\n')
            f.write(f'- Min: {vals["min"]:.2f}\n')
            f.write(f'- Max: {vals["max"]:.2f}\n')
            f.write(f'- Std: {vals["std"]:.2f}\n\n')
        f.write('## Monthly Aggregation (sample)\n')
        f.write(monthly.head(6).to_string())
        f.write('\n\n')
        f.write('## Seasonal Aggregation (sample)\n')
        f.write(seasonal_agg.head(6).to_string())
        f.write('\n\n')
        f.write('## Visualizations\n')
        for p in plots:
            if p:
                f.write(f'- {os.path.basename(p)}\n')
        f.write('\n\n')
        f.write('## Interpretation and Insights\n')
        f.write('- Review the plotted trends: seasonal cycles and rainfall patterns.\n')
        f.write('- Use monthly/seasonal aggregates to plan resources.\n')
    print(f'Wrote report: {report_file}')
    return cleaned_file, report_file


def main():
    df = download_and_load()
    cols = detect_columns(df)
    df_clean = clean_data(df, cols)
    # Compute stats
    stats, monthly, yearly = compute_statistics(df_clean)
    # Visualizations
    plots = create_visualizations(df_clean, output_prefix='plot')
    # Group & aggregate
    monthly_agg, seasonal_agg = group_and_aggregate(df_clean)
    # Export and report
    cleaned_file, report_file = export_and_report(df_clean, stats, monthly, yearly, monthly_agg, seasonal_agg, plots)
    print('\n-- All done. Generated files:')
    print(' -', cleaned_file)
    for p in plots:
        if p:
            print(' -', p)
    print(' -', report_file)

if __name__ == '__main__':
    main()
