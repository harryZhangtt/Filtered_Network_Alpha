import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


class DataAnalyzer:
    def __init__(self, file_path, initial_data_period, initial_graph_period):
        self.file_path = file_path
        self.data = self.load_data()


        def calculate_adjusted_close():
            df = self.data.copy()

            # Check for required columns
            if 'DIVIDEND' not in df.columns or 'SPLIT' not in df.columns:
                print("Required columns are missing.")
                return

            # Ensure the data is sorted by date
            df.sort_values(by='TRADINGDAY_x', inplace=True)

            # Initialize ADJ_CLOSE_PRICE with CLOSEPRICE
            df['cumulative_adjustment'] = ((df['SPLIT'] + df['ACTUALPLARATIO']) * df['CLOSEPRICE']) / \
                                          (df['CLOSEPRICE'] - df['DIVIDEND'] + df['ACTUALPLARATIO'] * df['PLAPRICE'])

            # Calculate the adjusted close price
            df['ADJ_CLOSE_PRICE'] = df['CLOSEPRICE'] * df['cumulative_adjustment']

            # Update the data
            self.data = df
            return df

        def calculate_adj_vwap():
            df = self.data.copy()
            # Calculate VWAP
            df['VWAP'] = df['TURNOVERVALUE'] / df['TURNOVERVOLUME']

            df['ADJ_VWAP'] = df['VWAP']
            df['cumulative_vwap_adjustment'] = ((df['SPLIT'] + df['ACTUALPLARATIO']) * df[
                'VWAP']) / \
                                               (df['VWAP'] - df['DIVIDEND'] + df[
                                                   'ACTUALPLARATIO'] * df['PLAPRICE'])

            # Calculate the adjusted VWAP
            df['ADJ_VWAP'] = df['ADJ_VWAP'] * df['cumulative_vwap_adjustment']
            df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_VWAP'].diff()
            df['next_day_return'].fillna(0, inplace=True)  # Fill NaNs that result from diff and shift

            # Calculate log return
            df['log_return'] = np.log(df['ADJ_VWAP'] / df.groupby('SECU_CODE')['ADJ_VWAP'].shift(1))

            # Calculate 21-day rolling volatility
            df['volatility'] = df.groupby('SECU_CODE')['next_day_return'].rolling(window=21).std().reset_index(level=0,
                                                                                                               drop=True)
            df['average_return'] = df.groupby('SECU_CODE')['next_day_return'].rolling(window=21).mean().reset_index(level=0,
                                                                                                                   drop=True)
            # Calculate Sharpe ratio
            df['sharpe_ratio'] = df['average_return'] / df['volatility']

            df['industry_neutral_return'] = df['log_return'] - df.groupby(['TRADINGDAY_x','SW2014F'])['log_return'].transform('mean')

            self.data = df

        def convert_sw2014t_to_int():
            # Convert 'SW2014T' to integer safely
            df= self.data.copy()
            if 'SW2014T' in df.columns:
                df['SW2014T'] = pd.to_numeric(df['SW2014T'], errors='coerce').fillna(0).astype(int)
            else:
                print("'SW2014T' column is missing in the data.")
            self.data=df
            return df

        convert_sw2014t_to_int()
        calculate_adjusted_close()
        calculate_adj_vwap()
        # Ensure initial_data_period and initial_graph_period are datetime objects
        self.initial_data_period = pd.to_datetime(initial_data_period, format='%Y%m%d')
        self.initial_graph_period = pd.to_datetime(initial_graph_period, format='%Y%m%d')
        print("Initial Data Period:", self.initial_data_period)
        print("Initial Graph Period:", self.initial_graph_period)

        # Ensure TRADINGDAY_x is treated as datetime
        self.data['TRADINGDAY_x'] = pd.to_datetime(self.data['TRADINGDAY_x'], format='%Y%m%d')

        # Create data_container with all data of 242 days up to the initial_data_period
        end_date = self.initial_data_period
        start_date = end_date - pd.DateOffset(years=1)
        self.data_container = self.data[(self.data['TRADINGDAY_x'] >= start_date) & (self.data['TRADINGDAY_x'] <= end_date)]

        end_date = self.initial_graph_period
        start_date = end_date - pd.DateOffset(months=1)
        # Create graph_container with all data of 30 days starting from the initial_graph_period
        self.graph_container = self.data[(self.data['TRADINGDAY_x'] >= start_date) & (self.data['TRADINGDAY_x'] <= end_date)]

        # Initialize containers
        self.update(self.initial_data_period, self.initial_graph_period)

        # Initialize a list to store results
        self.all_result = defaultdict(list)

    def calculate5DR(self):
        def compute_ndr_factor(days, delay=1):
            df = self.data.copy()

            if 'SECU_CODE' not in df.columns:
                print("Error: SECU_CODE column missing in data.")
                return

            # Ensure the data is sorted by TRADINGDAY_x within each SECU_CODE group before shifting
            df = df.sort_values(by=['SECU_CODE', 'TRADINGDAY_x'])

            # Calculate the NDR factor with the appropriate delay
            df['prev_close'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay)
            df['prev_close_days'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].shift(delay + days)
            df[f'{day}_ndr_factor'] = 1 - (df['prev_close'] / df['prev_close_days'])

            df['TOTALVALUE'] = df['TOTALSHARES'] * df['ADJ_CLOSE_PRICE']
            df['TURNOVERRATIO'] = df['TURNOVERVALUE'] / df['TOTALVALUE']

            self.data = df

        day = [1, 5, 20]
        for day in day:
            compute_ndr_factor(day)

    def load_data(self):
        # Load the data into dataframe
        if os.path.exists(self.file_path):
            try:
                df = pd.read_csv(self.file_path)
                print("Loaded data with columns:", df.columns)
                pd.to_datetime(df['TRADINGDAY_x'], format='%Y%m%d')
                df.sort_values(by='TRADINGDAY_x')
                return df
            except Exception as e:
                print("Failed to read file:", e)
                return pd.DataFrame()
        else:
            print("File does not exist at:", self.file_path)
            return pd.DataFrame()

    def update(self, new_data_period, new_graph_period):
        end_date = new_data_period
        start_date = end_date - pd.DateOffset(years=1)
        self.initial_data_period = new_data_period
        self.initial_graph_period = new_graph_period
        self.data_container = self.data[(self.data['TRADINGDAY_x'] >= start_date) & (self.data['TRADINGDAY_x'] <= end_date)]

        self.graph_container = self.data[(self.data['TRADINGDAY_x'] >= new_graph_period) &
                                         (self.data['TRADINGDAY_x'] < new_graph_period + pd.DateOffset(months=1))]

    def construct_graph(self, vector_column, exclude=False):
        # Ensure 'vector' column is numeric
        self.data_container = self.data_container.copy()
        self.data_container['vector'] = self.data_container[vector_column]

        # Pivot the data to get a matrix of 'vector' with SECU_CODE as columns
        pivot_data = self.data_container.pivot_table(values='vector', index='TRADINGDAY_x', columns='SECU_CODE')

        # Group by industry, excluding any NaN values in 'SW2014F'
        industry_groups = self.data_container.dropna(subset=['SW2014F']).groupby('SW2014F')['SECU_CODE'].unique()

        if vector_column != 'SW2014T':
            # Compute the correlation matrix if the vector_column is not 'SW2014T'
            correlation_matrix = pivot_data.corr(min_periods=30)

            # Extract the lower triangle of the correlation matrix
            return_corr_tril = correlation_matrix.values[np.tril_indices_from(correlation_matrix, -1)]

            # Calculate the benchmark as the average correlation of the lower triangle part of the matrix, excluding NaNs
            benchmark = np.nanmean(return_corr_tril)

            # Initialize dictionaries to store the most correlated tickers and their excess returns
            most_correlated_tickers = {}
            ticker_excess_return = {}

            for ticker in correlation_matrix.columns:
                if ticker in correlation_matrix.columns:
                    # Get the correlation series for the current ticker, drop NA and the ticker itself
                    corr_series = correlation_matrix[ticker].dropna().drop(ticker, errors='ignore')

                    # Find the tickers with the highest 1% correlation
                    top_correlated = corr_series[corr_series >= corr_series.quantile(0.99)].index.tolist()

                    if exclude:
                        top_correlated = self.exclude_same_industry(top_correlated, ticker, industry_groups)

                    # Add to the dictionary
                    most_correlated_tickers[ticker] = top_correlated

                    # Calculate the excess return for the ticker
                    if top_correlated:
                        excess_return = corr_series[top_correlated].mean() - benchmark
                        ticker_excess_return[ticker] = excess_return

            # Calculate the average industry distribution for the correlated tickers
            ratios, industry_distribution_list = self.calculate_industry_distribution(most_correlated_tickers)
            ratio_mean = np.mean(ratios)
            avg_industry_distribution = self.average_industry_distribution(industry_distribution_list)

            self.most_correlated_tickers = most_correlated_tickers
            self.ticker_excess_return = ticker_excess_return
            self.benchmark = benchmark

            # Prepare to add most correlated tickers for each period in self.data
            self.data['most_correlated_ticker'] = None

            # Prepare to add most correlated tickers for each date in self.data
            self.data['most_correlated_ticker'] = None

            # Iterate through each unique TRADINGDAY_x in self.graph_data
            for trading_day in pd.to_datetime(self.graph_data['TRADINGDAY_x']).unique():
                for ticker, top_correlated in most_correlated_tickers.items():
                    # Apply the top correlated ticker list for the specific trading day
                    self.data.loc[
                        (self.data['SECU_CODE'] == ticker) &
                        (self.data['TRADINGDAY_x'] == trading_day),
                        'most_correlated_ticker'
                    ] = [top_correlated] if top_correlated else None

            return ratio_mean, avg_industry_distribution

        else:
            # For SW2014T, handle industry-based correlation
            most_correlated_tickers = {}
            ticker_excess_return = {}

            # Create a dictionary mapping each SECU_CODE to its industry
            industry_dict = self.data_container.set_index('SECU_CODE')['SW2014T'].to_dict()

            for ticker in self.data_container['SECU_CODE'].unique():
                # Get the industry for the current ticker
                industry = industry_dict.get(ticker)
                if pd.notna(industry):
                    # Select all tickers that belong to the same industry, excluding the current ticker
                    top_correlated = self.data_container[self.data_container['SW2014T'] == industry]['SECU_CODE'].tolist()
                    top_correlated = [t for t in top_correlated if t != ticker]

                    if exclude:
                        top_correlated = self.exclude_same_industry(top_correlated, ticker, industry_groups)

                    # Add to the dictionary
                    most_correlated_tickers[ticker] = top_correlated


            self.most_correlated_tickers = most_correlated_tickers
            self.benchmark = 0  # Handle the case for 'SW2014T' appropriately



    def exclude_same_industry(self, top_correlated, ticker, industry_groups):
        industry_row = self.data_container[self.data_container['SECU_CODE'] == ticker]

        if not industry_row.empty:
            industry = industry_row['SW2014F'].iloc[0]
            if pd.notna(industry) and industry in industry_groups:
                same_industry_tickers = industry_groups[industry]
                top_correlated = [t for t in top_correlated if t not in same_industry_tickers]

        return top_correlated

    def industry_distribution(self, top_correlated, ticker):
        industry_counts = defaultdict(int)
        total_count = len(top_correlated)
        same_industry_count = 0

        for related_ticker in top_correlated:
            industry = self.data_container[self.data_container['SECU_CODE'] == related_ticker]['SW2014F'].iloc[0]

            if industry == self.data_container[self.data_container['SECU_CODE'] == ticker]['SW2014F'].iloc[0]:
                same_industry_count += 1

            if pd.notna(industry):
                industry_counts[industry] += 1

        industry_ratios = {industry: count / total_count for industry, count in industry_counts.items()}
        same_industry_ratio = same_industry_count / total_count if total_count > 0 else 0

        return industry_ratios, same_industry_ratio

    def calculate_industry_distribution(self, most_correlated_tickers):
        ratios = []
        industry_distribution_list = []

        for ticker, top_correlated in most_correlated_tickers.items():
            industry_ratios, ratio = self.industry_distribution(top_correlated, ticker)
            industry_distribution_list.append(industry_ratios)
            ratios.append(ratio)

        return ratios, industry_distribution_list

    def average_industry_distribution(self, industry_distribution_list):
        all_industries = set(itertools.chain.from_iterable(industry_distribution_list))
        avg_industry_distribution = {
            industry: np.mean([d.get(industry, 0) for d in industry_distribution_list]) for industry in all_industries
        }

        return avg_industry_distribution

    def calculate_alpha(self, vector_column):
        self.graph_container = self.graph_container.copy()
        self.graph_container['vector'] = self.graph_container[vector_column]
        alpha_values = []

        for trading_day in self.graph_container['TRADINGDAY_x'].unique():
            print(trading_day)
            day_mask = self.graph_container['TRADINGDAY_x'] == trading_day

            for ticker in self.graph_container[day_mask]['SECU_CODE'].unique():
                ticker_data = self.graph_container[(self.graph_container['SECU_CODE'] == ticker) & day_mask]

                if ticker in self.most_correlated_tickers:
                    related_ticker_means = []
                    for related_ticker in self.most_correlated_tickers[ticker]:
                        related_data = self.graph_container[
                            (self.graph_container['SECU_CODE'] == related_ticker) & day_mask]

                        if not related_data.empty:
                            related_ticker_means.append(related_data['vector'].values[0])

                    if related_ticker_means:
                        overall_mean = np.mean(related_ticker_means)

                        if not np.isnan(overall_mean) and not np.isnan(ticker_data['vector'].values[0]):
                            ticker_mean_return = ticker_data['vector'].values[0]
                            X = np.array([ticker_mean_return]).reshape(-1, 1)
                            y = np.array([overall_mean]).reshape(-1, 1)

                            # Fit the model
                            reg = LinearRegression().fit(X, y)
                            y_pred = reg.predict(X)
                            residuals = y - y_pred

                            alpha_values.append({
                                'TRADINGDAY_x': trading_day,
                                'SECU_CODE': ticker,
                                f'{vector_column}_alpha': residuals.flatten()[0]
                            })

        # Create a DataFrame from alpha_values to facilitate shifting
        alpha_df = pd.DataFrame(alpha_values)
        # Sort the DataFrame by 'TRADINGDAY_x'
        alpha_df = alpha_df.sort_values(by=['SECU_CODE', 'TRADINGDAY_x'])

        # Shift the alpha values by 1 to avoid using future data for each ticker
        alpha_df[f'{vector_column}_alpha'] = alpha_df.groupby('SECU_CODE')[f'{vector_column}_alpha'].shift(1)

        # Drop rows with NaN values after the shift
        alpha_df = alpha_df.fillna(0)

        # Update the graph_container with the calculated and shifted alpha values
        self.graph_container[f'{vector_column}_alpha'] = np.nan
        for _, alpha in alpha_df.iterrows():
            self.graph_container.loc[(self.graph_container['TRADINGDAY_x'] == alpha['TRADINGDAY_x']) &
                                     (self.graph_container['SECU_CODE'] == alpha['SECU_CODE']),
                                     f'{vector_column}_alpha'] = alpha[f'{vector_column}_alpha']

        return alpha_df.to_dict('records')

    def normalize_factors(self, vector_column):
        df = self.graph_container.copy()
        df = df.sort_values(by='TRADINGDAY_x')

        # Rank within each trading day
        df['rank'] = df.groupby('TRADINGDAY_x')[f'{vector_column}_alpha'].rank()

        # normalized factor for today
        df[f'{vector_column}_normalized_alpha'] = df.groupby('TRADINGDAY_x')['rank'].transform(
            lambda x: 2 * (x - 1) / (len(x) - 1) - 1 if len(x) > 1 else 0)

        self.graph_container = df

    def simple_backtest(self, alpha_column, initial_capital=1e8):
        df = self.graph_container.copy()
        df = df.sort_values(by='TRADINGDAY_x')
        if 'ADJ_CLOSE_PRICE' not in df.columns:
            print("ADJ_CLOSE_PRICE column is missing.")
            return None, None

        # Ensure there are no zero prices to avoid division by zero
        df['ADJ_CLOSE_PRICE'].replace(0, np.nan, inplace=True)
        df['ADJ_CLOSE_PRICE'].ffill(inplace=True)

        # Append the alpha to the DataFrame to align by date and stock code, assuming alpha is correctly indexed
        df['alpha'] = df[alpha_column]

        def weight_assignment(df):
            df = df.sort_values(by='TRADINGDAY_x')

            # Define masks for long and short investments based on the normalized factor
            df['long_weight'] = 0.0
            df['short_weight'] = 0.0

            df.loc[df['alpha'] >= 0, 'long_weight'] = abs(df['alpha']) / df[df['alpha'] >= 0].groupby('TRADINGDAY_x')[
                'alpha'].transform('sum')
            df.loc[df['alpha'] < 0, 'short_weight'] = abs(df['alpha']) / df[df['alpha'] < 0].groupby('TRADINGDAY_x')[
                'alpha'].transform('sum')

            df.loc[df['alpha'] >= 0, 'weight'] = df['long_weight']
            df.loc[df['alpha'] < 0, 'weight'] = df['short_weight']
            return df

        df = weight_assignment(df)

        # Allocate capital based on weights
        df['long_capital_allocation'] = initial_capital * df['long_weight']
        df['short_capital_allocation'] = initial_capital * df['short_weight']

        # Calculate investment amount in shares for long and short positions
        # round to the closest multiple of 100 smaller than the original number
        df['long_investments'] = ((df['long_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100
        df['short_investments'] = ((df['short_capital_allocation'] / df['ADJ_CLOSE_PRICE']) // 100) * 100

        # Assign investments based on the alpha value
        df['investment'] = 0  # Initialize investment column with zeros

        # Assign long investments to stocks with positive alpha
        df.loc[df['weight'] >= 0, 'investment'] = df['long_investments']

        # Assign short investments to stocks with negative alpha
        df.loc[df['weight'] < 0, 'investment'] = df['short_investments']

        # Calculate the next-day price change
        df['next_day_return'] = df.groupby('SECU_CODE')['ADJ_CLOSE_PRICE'].diff()
        df['next_day_return'].fillna(0, inplace=True)  # Fill NaNs that result from diff and shift

        # Shift investments to get the previous day's investments
        df['previous_investment'] = df.groupby('SECU_CODE')['investment'].shift(1).fillna(0)

        # Calculate investment changes
        df['investment_change'] = (df['investment'] - df['previous_investment']).fillna(0)
        df['abs_investment_change'] = df['investment_change'].abs()


        """
        notice that we hold only when the sign of pervious investment and current investment is the same
        """
        condition = df['previous_investment'] * df['investment'] > 0

        # Calculate hold_pnl based on the condition
        df['hold_pnl'] = np.where(condition, df['previous_investment'] * df['next_day_return'], 0)

        df['trade_pnl'] = df['investment_change'] * (
                df['ADJ_CLOSE_PRICE'] - df['ADJ_VWAP'])
        df['pnl'] = df['hold_pnl'].fillna(0) + df['trade_pnl'].fillna(0)

        df['long_pnl'] = df['pnl'] * (df['vector'] > 0)
        df['short_pnl'] = df['pnl'] * (df['vector'] <= 0)

        overall_pnl = df['pnl'].sum()

        self.graph_container = df
        return df,overall_pnl

    def verify_graph_strength(self, vector_column):
        self.graph_container = self.graph_container.copy()
        self.graph_container['vector'] = self.graph_container[vector_column]
        # Calculate the benchmark as the average correlation of the lower triangle part of the matrix, excluding the diagonal
        pivot_data = self.graph_container.pivot_table(values='vector', index='TRADINGDAY_x', columns='SECU_CODE')
        correlation_matrix = pivot_data.corr()
        lower_triangle = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape), k=-1).astype(bool))
        benchmark = lower_triangle.stack().mean()

        node_strength = {}

        valid_tickers = set(self.data_container['SECU_CODE'].unique()).intersection(set(correlation_matrix.columns))

        for ticker in valid_tickers:
            if ticker in self.most_correlated_tickers:
                # Filter correlated tickers that exist in the correlation matrix
                correlated_tickers = [t for t in self.most_correlated_tickers[ticker] if t in correlation_matrix.columns]
                if correlated_tickers:
                    corr_values = correlation_matrix.loc[ticker, correlated_tickers].dropna()
                    if not corr_values.empty:
                        node_strength[ticker] = corr_values.mean()

        self.benchmark = benchmark
        self.node_strength = node_strength

        # Calculate the mean of node strengths
        if node_strength:
            node_strength_mean = np.mean(list(node_strength.values()))
        else:
            node_strength_mean = None

        print("Benchmark:", self.benchmark)
        print("Node Strength:", self.node_strength)
        print("Node Strength Mean:", node_strength_mean)

        return {
            'benchmark': self.benchmark,
            'node_strength': self.node_strength,
            'node_strength_mean': node_strength_mean
        }

    import pandas as pd
    import numpy as np
    import itertools

    def calculate_metric_correlations(self):
        vectors = [ 'log_return','TURNOVERRATIO','volatility','SW2014T','1_ndr_factor']
        correlation_results = []
        df = self.data.copy()
        df = df.sort_values(by='TRADINGDAY_x')
        df = df[df['TRADINGDAY_x'] >= pd.to_datetime('20130301', format='%Y%m%d')]
        df['year_month'] = df['TRADINGDAY_x'].dt.to_period('M')

        for year_month in df['year_month'].unique():
            print(year_month)
            new_data_period = self.initial_data_period + pd.DateOffset(months=1)
            new_graph_period = self.initial_graph_period + pd.DateOffset(months=1)
            print("new graph period", new_graph_period)
            self.update(new_data_period, new_graph_period)

            # Store most correlated tickers for each metric
            all_most_correlated = {}

            for vector in vectors:
                print(vector)
                if vector != 'SW2014T':
                    try:
                        self.construct_graph(vector, exclude=False)
                        all_most_correlated[vector] = self.most_correlated_tickers
                    except KeyError as e:
                        print(f"KeyError for {vector}: {e}")
                        continue
                else:
                    self.construct_graph(vector, exclude=False)
                    all_most_correlated[vector] = self.most_correlated_tickers

            ticker_similarities = {}

            # Calculate similarities for each ticker
            for ticker in set().union(*[set(d.keys()) for d in all_most_correlated.values()]):
                similarities = []
                for vector_a in vectors:
                    for vector_b in vectors:
                        if(vector_a!=vector_b):
                            neighbors_a = set(all_most_correlated[vector_a].get(ticker, []))
                            neighbors_b = set(all_most_correlated[vector_b].get(ticker, []))
                            intersection = neighbors_a.intersection(neighbors_b)
                            union = neighbors_a.union(neighbors_b)
                            ratio = len(intersection) / len(union) if len(union) > 0 else 0.0
                            similarities.append(ratio)

                            correlation_results.append({
                                'vector_a': vector_a,
                                'vector_b': vector_b,
                                'ticker': ticker,
                                'similarity': ratio,
                                'year_month': year_month
                            })

            # Calculate the average similarity for each ticker across all vector pairs
            if ticker_similarities:
                for ticker, similarity in ticker_similarities.items():
                    correlation_results.append({
                        'ticker': ticker,
                        'average_similarity': similarity,
                        'year_month': year_month
                    })

        return pd.DataFrame(correlation_results)


    def plot_metric_correlations(self, correlation_df):
        # Ensure the DataFrame has the expected columns
        if 'vector_a' not in correlation_df.columns or 'vector_b' not in correlation_df.columns:
            raise KeyError("The DataFrame does not have the required 'vector_a' and 'vector_b' columns.")

        # Group by vector pairs and year_month, then calculate the average similarity
        avg_similarity_df = correlation_df.groupby(['vector_a', 'vector_b', 'year_month'])[
            'similarity'].mean().reset_index()

        # Get unique vectors
        vectors = ['1_ndr_factor', 'log_return', 'TURNOVERRATIO', 'volatility', 'SW2014T']

        for vector_a in vectors:
            plt.figure(figsize=(12, 6))

            # Plot correlations of vector_a with all other vectors except itself
            for vector_b in vectors:
                if vector_a != vector_b:
                    group = avg_similarity_df[
                        (avg_similarity_df['vector_a'] == vector_a) & (avg_similarity_df['vector_b'] == vector_b)]
                    if not group.empty:
                        plt.plot(group['year_month'].astype(str), group['similarity'], marker='o',
                                 label=f'{vector_a} to {vector_b}')

            plt.title(f'Average Correlation of {vector_a} with Other Vectors Over Time')
            plt.xlabel('Year-Month')
            plt.ylabel('Average Similarity')
            plt.xticks(rotation=45)

            # Reduce the density of x-axis labels
            xticks = plt.gca().get_xticks()
            plt.gca().set_xticks(xticks[::2])

            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()

            # Save the plot as an image file
            filename = f'corr_between_{vector_a}_and_others.png'
            plt.savefig(filename)
            plt.show()

    def run_analysis(self):
        vectors = [ '5_ndr_factor', '20_ndr_factor','TURNOVERRATIO','volatility', 'industry_neutral_return', 'log_return','1_ndr_factor']
        all_alpha_results = []
        all_results = []
        total_pnls =[]

        for vector in vectors:
            ratio_means = []
            average_industry_distributions = []
            self.update(pd.to_datetime('20130101', format='%Y%m%d'), pd.to_datetime('20130201', format='%Y%m%d'))
            df = self.data.copy()
            df = df.sort_values(by='TRADINGDAY_x')
            df = df[df['TRADINGDAY_x'] >= pd.to_datetime('20130301', format='%Y%m%d')]
            df['year_month'] = df['TRADINGDAY_x'].dt.to_period('M')

            for year_month in df['year_month'].unique():
                print(year_month)
                new_data_period = self.initial_data_period + pd.DateOffset(months=1)
                new_graph_period = self.initial_graph_period + pd.DateOffset(months=1)
                print("new graph period", new_graph_period)
                self.update(new_data_period, new_graph_period)

                if vector != 'SW2014T':
                    ratio_mean, average_industry_distribution = self.construct_graph(vector, exclude=False)
                    ratio_means.append(ratio_mean)
                    average_industry_distributions.append(average_industry_distribution)
                    result = self.verify_graph_strength(vector)
                    alpha_results = self.calculate_alpha(vector)
                    all_alpha_results.extend(alpha_results)
                    self.normalize_factors(vector)
                    backtest_df,overall_pnl = self.simple_backtest(f'{vector}_normalized_alpha')
                    print(f"pnl on {year_month}",overall_pnl)
                    total_pnls.append(overall_pnl)
                else:
                    self.construct_graph(vector, exclude=False)
                    result = self.verify_graph_strength('log_return')
                    alpha_results = self.calculate_alpha('log_return')
                    print('finished')
                    all_alpha_results.extend(alpha_results)
                    self.normalize_factors(vector)
                    backtest_df,overall_pnl = self.simple_backtest(f'{vector}_normalized_alpha')
                    print(f"pnl on {year_month}", overall_pnl)
                    total_pnls.append(overall_pnl)

                # # Append the backtesting results to all_results
                for alpha_result in alpha_results:
                    trading_day = alpha_result['TRADINGDAY_x']
                    secu_code = alpha_result['SECU_CODE']
                    alpha_value = alpha_result[f'{vector}_alpha']
                    normalized_alpha_value = backtest_df.loc[
                        (backtest_df['TRADINGDAY_x'] == trading_day) & (backtest_df['SECU_CODE'] == secu_code),
                        f'{vector}_normalized_alpha'].values[0]
                    weight_value = backtest_df.loc[
                        (backtest_df['TRADINGDAY_x'] == trading_day) & (backtest_df['SECU_CODE'] == secu_code),
                        'weight'].values[0]

                    all_results.append({
                        'TRADINGDAY_x': trading_day,
                        'SECU_CODE': secu_code,
                        'alpha': alpha_value,
                        f'{vector}_normalized_alpha': normalized_alpha_value,
                        'weight': weight_value
                    })
            print('overall pnl is ', np.sum(total_pnls))

            """
            save and plot pnl data"""
            # Plot the cumulative sum of total_pnls
            cumulative_pnl = np.cumsum(total_pnls)
            plt.figure(figsize=(10, 6))
            plt.plot(cumulative_pnl, marker='o')
            plt.title('Cumulative PnL Over Time')
            plt.xlabel('Time Period')
            plt.ylabel('Cumulative PnL')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('/Users/zw/Desktop/cumulative_pnl.png')
            plt.show()

            # Save the overall PnL to a CSV file
            pnl_df = pd.DataFrame(total_pnls, columns=['PnL'])
            pnl_df.to_csv('/Users/zw/Desktop/overall_pnl.csv', index=False)

            # Save all_results to a CSV file
            all_results_df = pd.DataFrame(all_results)
            all_results_df.to_csv('/Users/zw/Desktop/all_results.csv', index=False)

            """
            for verifying node strength
            """

            self.all_result[(vector, year_month)].append(result)

            if vector != 'SW2014T':
                print('Industry exposure in the graph:', np.mean(ratio_means))
                # Calculate overall average industry distribution
                overall_avg_industry_distribution = defaultdict(float)
                for dist in average_industry_distributions:
                    for industry, ratio in dist.items():
                        overall_avg_industry_distribution[industry] += ratio / len(average_industry_distributions)

                # Convert to DataFrame for plotting
                industry_df = pd.DataFrame.from_dict(overall_avg_industry_distribution, orient='index',
                                                     columns=['Ratio'])

                # Plot industry distribution
                fig, ax = plt.subplots()
                industry_df.plot(kind='bar', ax=ax)
                ax.set_ylabel(f'Average Industry Distribution Ratio for {vector}')
                ax.set_xlabel('Industry')
                ax.set_title('Overall Average Industry Distribution')
                plt.tight_layout()
                plt.savefig(f'/Users/zw/Desktop/average_industry_distribution for {vector}.png')
                plt.show()

            # Convert all_result to DataFrame
            results_list = []
            for (vec, year_month), result in self.all_result.items():
                if vec == vector:
                    for res in result:
                        results_list.append({
                            'vector': vec,
                            'year_month': year_month,
                            'benchmark': res['benchmark'],
                            'node_strength': res['node_strength'],
                            'node_strength_mean': res['node_strength_mean']
                        })

            results_df = pd.DataFrame(results_list)

            # Save results to CSV
            results_df.to_csv(f'/Users/zw/Desktop/{vector}_analysis_results.csv', index=False)

            # Plot average node strength and benchmark over time
            fig, ax1 = plt.subplots()

            ax1.set_xlabel('Year-Month')
            ax1.set_ylabel('Values')
            vector_data = results_df[results_df['vector'] == vector]
            ax1.plot(vector_data['year_month'].astype(str), vector_data['node_strength_mean'],
                     label=f'{vector} Node Strength')
            ax1.plot(vector_data['year_month'].astype(str), vector_data['benchmark'], linestyle='--',
                     label=f'{vector} Benchmark')

            ax1.tick_params(axis='y')
            # Set xticks manually
            ax1.set_xticks(vector_data['year_month'].iloc[::6].astype(str))
            ax1.set_xticklabels(vector_data['year_month'].iloc[::6].astype(str), rotation=45)

            # Set the same y-axis scale for both node strength and benchmark
            min_y = vector_data[['node_strength_mean', 'benchmark']].min().min()
            max_y = vector_data[['node_strength_mean', 'benchmark']].max().max()
            ax1.set_ylim(min_y, max_y)

            # Add legend
            ax1.legend()

            plt.title(f'Average Node Strength and Benchmark Over Time for {vector}')
            fig.tight_layout()

            plt.savefig(f'/Users/zw/Desktop/{vector}_average_node_strength_and_benchmark.png')
            plt.show()

            """
            for alpha """
            # Save alpha results to a separate CSV
            alpha_df = pd.DataFrame(all_alpha_results)
            alpha_df.to_csv(f'/Users/zw/Desktop/{vector}_alpha.csv', index=False)

            # # Save daily alpha results to CSV
            all_results_df = pd.DataFrame(all_results)
            all_results_df.to_csv(f'/Users/zw/Desktop/{vector}_all_results.csv', index=False)

        """
        calculate parameters similarity
        """
        correlations = self.calculate_metric_correlations()
        self.plot_metric_correlations(correlations)



if __name__ == "__main__":
    file_path_sample = "/Users/zw/Desktop/sample_data.csv"
    file_path_all = "/Users/zw/Desktop/combined_data.csv"
    file_path_HS300 = "/Users/zw/Desktop/combinedHS300_data.csv"
    file_path_HS300_recent = "/Users/zw/Desktop/sampleHS300_data.csv"
    analysis = DataAnalyzer(file_path_HS300, '20130101', '20130201')
    analysis.calculate5DR()
    analysis.run_analysis()
