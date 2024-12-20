import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from finlab import data
import matplotlib.dates as mdates
import finlab
from matplotlib.backends.backend_pdf import PdfPages
import os
os.system('cls')


class StockPlotter:
    def __init__(self, stock_selection_file, stock_topics_file):
        """
        初始化股票繪圖器

        參數:
        stock_selection_file (str): 選股Excel檔案路徑
        stock_topics_file (str): 股票主題Excel檔案路徑
        """
        # 讀取數據文件
        self.stock_selection_df = pd.read_excel(stock_selection_file)
        self.stock_topics_df = pd.read_excel(stock_topics_file)

        # 初始化變數
        self.stock_counts = {}
        self.sorted_stocks = None
        self.result_df = None
        self.colors = None
        self.high_score_stocks = None

        # 設置全局字體樣式
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'

        # 忽略所有警告
        warnings.filterwarnings("ignore")

    def process_data(self):
        """處理股票數據"""
        # 處理策略選股數據
        strategies = self.stock_selection_df.columns

        for strategy in strategies:
            stocks = self.stock_selection_df[strategy].dropna()
            for stock in stocks:
                stock = int(float(stock))
                if stock in self.stock_counts:
                    self.stock_counts[stock]['count'] += 1
                    self.stock_counts[stock]['strategies'].append(strategy)
                else:
                    self.stock_counts[stock] = {
                        'count': 1,
                        'strategies': [strategy]
                    }

        # 根據被選次數排序
        self.sorted_stocks = sorted(
            self.stock_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        # 創建結果DataFrame
        result_data = []
        for stock, data in self.sorted_stocks:
            stock_info = self.stock_topics_df.loc[self.stock_topics_df['stock_no'] == stock]
            stock_name = stock_info['stock_name'].values[0] if len(
                stock_info) > 0 else "未知"
            stock_topic = stock_info['topic'].values[0] if len(
                stock_info) > 0 else "無"
            result_data.append({
                '股票代碼': stock,
                '股票名稱': stock_name,
                '主題': stock_topic,
                '被選次數': data['count'],
                '對應策略': data['strategies']
            })

        self.result_df = pd.DataFrame(result_data)

        # 處理K線圖所需的股票數據
        stock_set = set()
        for index, row in self.stock_selection_df.iterrows():
            for stock_id in row:
                if pd.notna(stock_id):
                    stock_set.add(str(int(stock_id)))
        self.high_score_stocks = sorted(stock_set)

    def setup_colors(self):
        """設置策略顏色"""
        all_strategies = list(
            set([strategy for strategies in self.result_df['對應策略']
                for strategy in strategies])
        )
        color_map = plt.cm.get_cmap('Set1')
        self.colors = {
            strategy: color_map(i/len(all_strategies))
            for i, strategy in enumerate(all_strategies)
        }

    def plot_stock(self, ax, s):
        """
        繪製單個股票的K線圖

        參數:
        ax: matplotlib軸對象
        s: 股票代碼
        """
        try:
            # 獲取K線數據
            open_data = data.get("price:開盤價")[s].tail(120)
            high_data = data.get("price:最高價")[s].tail(120)
            low_data = data.get("price:最低價")[s].tail(120)
            close_data = data.get("price:收盤價")[s].tail(120)

            dates = pd.date_range(
                end=close_data.index[-1], periods=len(close_data))

            # 繪製K線圖
            width = 0.8
            width2 = 0.2
            up = close_data > open_data
            color = np.where(up, 'red', 'green')

            # 繪製蠟燭實體
            ax.bar(dates[up], close_data[up] - open_data[up],
                   width, bottom=open_data[up], color='red', edgecolor='black', linewidth=0.5)
            ax.bar(dates[~up], open_data[~up] - close_data[~up],
                   width, bottom=close_data[~up], color='green', edgecolor='black', linewidth=0.5)

            # 繪製上下影線
            ax.bar(dates, high_data - low_data, width2,
                   bottom=low_data, color=color, zorder=3)

            # 設置標題和標籤
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(
                s)]
            if not stock_info.empty:
                stock_name = stock_info['stock_name'].values[0]
                topic = stock_info['topic'].values[0]
                title = f'{stock_name} ({s}) - {topic}'
            else:
                title = f'股票代號 {s}'

            ax.set_title(title, fontsize=30, fontweight='bold',
                         color='black', pad=10)
            ax.set_xlabel('日期', fontsize=30, fontweight='bold',
                          color='black', labelpad=5)

            # 設置網格和刻度
            ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='both', which='major',
                           labelsize=20, colors='black')
            plt.setp(ax.xaxis.get_majorticklabels(),
                     rotation=45, ha='right', color='black')
            ax.yaxis.set_label_coords(-0.02, 0.5)

            # 添加日期標記
            today = datetime.today().strftime('%Y-%m-%d')
            ax.text(0.95, 0.01, f'Date: {today}',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    color='gray', fontsize=25)

        except KeyError:
            print(f"股票代號 {s} 不存在於資料中，跳過該股票。")

    def calculate_stock_correlations(self):
        """計算股票之間的相關性"""
        # 創建一個字典來存儲每支股票的收盤價數據
        stock_prices = {}
        for stock in self.high_score_stocks:
            try:
                close_data = data.get("price:收盤價")[stock].tail(120)
                stock_prices[stock] = close_data
            except KeyError:
                continue

        # 創建相關性矩陣
        price_df = pd.DataFrame(stock_prices)
        correlations = price_df.corr()

        # 使用層次聚類來對股票進行分組
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # 將相關性矩陣轉換為距離矩陣
        distances = 1 - correlations
        condensed_dist = squareform(distances)

        # 進行層次聚類
        Z = linkage(condensed_dist, method='ward')
        clusters = fcluster(Z, t=3, criterion='maxclust')  # 分為3組

        # 將股票按照聚類結果重新排序
        clustered_stocks = []
        for i in range(1, max(clusters) + 1):
            cluster_stocks = [stock for j, stock in enumerate(correlations.index)
                              if clusters[j] == i]
            clustered_stocks.extend(cluster_stocks)

        return clustered_stocks


    def create_pdf(self):
        """創建包含所有圖表的PDF文件"""
        today = datetime.today().strftime('%Y%m%d')
        pdf_filename = f'tomstrategyonlykbar_{today}.pdf'

        sorted_stocks = self.calculate_stock_correlations()

        with PdfPages(pdf_filename) as pdf:
            # 計算實際的股票數量
            total_stocks = len(self.result_df)

            # 第一頁：策略選股統計圖
            stocks_per_plot = 10
            num_plots = (total_stocks + stocks_per_plot - 1) // stocks_per_plot
            num_plots = min(4, num_plots)

            for plot_group in range((num_plots + 1) // 2):  # 每頁放2個圖
                fig = plt.figure(figsize=(48, 24))

                for i in range(2):  # 每頁2個子圖
                    plot_index = plot_group * 2 + i
                    if plot_index < num_plots:
                        ax = fig.add_subplot(2, 1, i + 1)
                        start_idx = plot_index * stocks_per_plot
                        end_idx = min((plot_index + 1) *
                                    stocks_per_plot, total_stocks)
                        subset = self.result_df.iloc[start_idx:end_idx]
                        n_stocks = len(subset)
                        bottoms = np.zeros(n_stocks)

                        for strategy in self.colors.keys():
                            heights = [strats.count(strategy)
                                    for strats in subset['對應策略']]
                            bars = ax.bar(subset['股票名稱'] + '\n(' + subset['股票代碼'].astype(str) + ')\n' + subset['主題'],
                                        heights, bottom=bottoms,
                                        color=self.colors[strategy],
                                        edgecolor='black')
                            bottoms += heights

                            for bar, height, strat in zip(bars, heights, subset['對應策略']):
                                if height > 0:
                                    ax.annotate(f'{strategy}',
                                                xy=(bar.get_x() + bar.get_width() / 2,
                                                    bar.get_y() + bar.get_height() / 2),
                                                xytext=(0, 0), textcoords="offset points",
                                                ha='center', va='center', fontsize=14, color='black')

                        ax.set_title(
                            f'被多策略選到的標的(優先觀察) ({start_idx+1}-{end_idx})', fontsize=25)
                        ax.set_ylabel('被\n選\n次\n數', fontsize=18,
                                    rotation=0, labelpad=20)
                        ax.set_xticks(range(len(subset)))
                        ax.set_xticklabels(subset['股票名稱'] + '\n(' + subset['股票代碼'].astype(str) + ')\n' + subset['主題'],
                                        rotation=45, ha='right', fontsize=22)

                        max_count = subset['被選次數'].max()
                        ax.set_ylim(0, max_count * 1.2)
                        for j, count in enumerate(subset['被選次數']):
                            ax.text(j, count, str(count), ha='center',
                                    va='bottom', fontsize=16)
                        ax.tick_params(axis='y', labelsize=16)
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # 新頁：主題統計圖
            fig = plt.figure(figsize=(48, 24))
            ax = fig.add_subplot(111)
            topic_counts = self.result_df['主題'].value_counts()
            bar_colors = plt.cm.viridis(np.linspace(0, 1, len(topic_counts)))
            topic_counts.plot(kind='bar', color=bar_colors, ax=ax)

            ax.set_title('熱門族群觀察', fontsize=30)
            ax.set_ylabel('出\n現\n次\n數', fontsize=25, rotation=0, labelpad=20)

            new_labels = [
                f"{topic}\n({', '.join(
                    map(str, self.result_df[self.result_df['主題'] == topic]['股票代碼'].tolist()))})"
                for topic in topic_counts.index
            ]
            ax.set_xticklabels(new_labels, rotation=45,
                            ha='right', fontsize=22)

            for index, topic in enumerate(topic_counts.index):
                ax.text(index, topic_counts[topic], f'{topic_counts[topic]}',
                        ha='center', va='bottom', fontsize=20)

            ax.tick_params(axis='both', labelsize=20)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # K線圖頁面
            for i in range(0, len(sorted_stocks), 3):
                fig, (ax1, ax2, ax3) = plt.subplots(
                    1, 3, figsize=(48, 12), facecolor='white')

                if i < len(sorted_stocks):
                    self.plot_stock(ax1, sorted_stocks[i])
                else:
                    ax1.axis('off')

                if i + 1 < len(sorted_stocks):
                    self.plot_stock(ax2, sorted_stocks[i + 1])
                else:
                    ax2.axis('off')

                if i + 2 < len(sorted_stocks):
                    self.plot_stock(ax3, sorted_stocks[i + 2])
                else:
                    ax3.axis('off')

                plt.subplots_adjust(wspace=0.3)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)



# 主程序
if __name__ == "__main__":
    try:
        # 創建StockPlotter實例
        stock_plotter = StockPlotter(
            'select_only_kbar.xlsx', 'tw_stock_topics.xlsx')
        # 處理數據
        print("正在處理數據...")
        stock_plotter.process_data()
        # 設置顏色
        print("正在設置圖表顏色...")
        stock_plotter.setup_colors()
        # 創建PDF報告
        print("正在生成PDF報告...")
        stock_plotter.create_pdf()
        print("報告生成完成！檔案已保存為 tomstrategyonlykbar.pdf")
    except Exception as e:
        print(f"程式執行出錯：{str(e)}")


# python onlykbar.py
