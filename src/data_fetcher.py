"""
Module for fetching index and constituent stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# Dictionary mapping index names to their Yahoo Finance tickers and constituents
INDEX_CONSTITUENTS = {
    'CAC40': {
        'ticker': '^FCHI',
        'constituents': [
            'AIR.PA', 'AI.PA', 'MT.AS', 'CS.PA', 'BNP.PA', 'EN.PA', 
            'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA',
            'EL.PA', 'ERF.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA',
            'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA',
            'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA',
            'STMPA.PA', 'TEP.PA', 'HO.PA', 'FP.PA', 'URW.AS', 'VIE.PA',
            'DG.PA', 'VIV.PA', 'WLN.PA', 'ALO.PA'
        ]
    },
    'S&P500': {
        'ticker': '^GSPC',
        'constituents': []  # Too many to list here, would need web scraping
    },
    'DOW': {
        'ticker': '^DJI',
        'constituents': [
            'AAPL', 'MSFT', 'JPM', 'V', 'UNH', 'HD', 'PG', 'JNJ', 'MA',
            'DIS', 'CSCO', 'CRM', 'MRK', 'WMT', 'NKE', 'AMGN', 'MCD',
            'CAT', 'CVX', 'AXP', 'IBM', 'TRV', 'GS', 'HON', 'BA', 'MMM',
            'KO', 'DOW', 'VZ', 'WBA'
        ]
    },
    'NASDAQ100': {
        'ticker': '^NDX',
        'constituents': []  # Too many to list here
    },
    'DAX': {
        'ticker': '^GDAXI',
        'constituents': [
            'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
            'BMW.DE', 'CON.DE', 'DAI.DE', 'DB1.DE', 'DBK.DE', 'DHL.DE',
            'DTE.DE', 'EOAN.DE', 'FME.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE',
            'IFX.DE', 'LIN.DE', 'MBG.DE', 'MRK.DE', 'MTX.DE', 'MUV2.DE',
            'PAH3.DE', 'PSM.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'VOW3.DE',
            'VNA.DE', 'ZAL.DE', 'QIA.DE', 'P911.DE', 'SHL.DE', 'SY1.DE',
            'HNR1.DE', 'PUM.DE', 'HFG.DE', 'BNR.DE'
        ]
    },
    'FTSE100': {
        'ticker': '^FTSE',
        'constituents': []  # Can be added if needed
    }
}


class IndexDataFetcher:
    """
    Class to fetch historical data for an index and its constituent stocks.
    """
    
    def __init__(self, index_name: str, period: str = '1y', interval: str = '1d'):
        """
        Initialize the IndexDataFetcher.
        
        Parameters:
        -----------
        index_name : str
            Name of the index (e.g., 'CAC40', 'DOW', 'DAX')
        period : str, optional
            Time period to fetch data for (default: '1y')
            Valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval : str, optional
            Data interval (default: '1d')
            Valid values: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        self.index_name = index_name.upper()
        self.period = period
        self.interval = interval
        
        if self.index_name not in INDEX_CONSTITUENTS:
            raise ValueError(f"Index '{index_name}' not supported. "
                           f"Available indices: {list(INDEX_CONSTITUENTS.keys())}")
        
        self.index_ticker = INDEX_CONSTITUENTS[self.index_name]['ticker']
        self.constituents = INDEX_CONSTITUENTS[self.index_name]['constituents']
        
        self.index_data = None
        self.stocks_data = {}
        self.combined_data = None
    
    def fetch_index_data(self) -> pd.DataFrame:
        """
        Fetch historical data for the index.
        
        Returns:
        --------
        pd.DataFrame
            Historical data for the index
        """
        print(f"Fetching data for {self.index_name} ({self.index_ticker})...")
        
        try:
            index = yf.Ticker(self.index_ticker)
            self.index_data = index.history(period=self.period, interval=self.interval)
            
            if self.index_data.empty:
                print(f"Warning: No data retrieved for {self.index_name}")
            else:
                print(f"Successfully fetched {len(self.index_data)} records for {self.index_name}")
            
            return self.index_data
        
        except Exception as e:
            print(f"Error fetching data for {self.index_name}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_constituents_data(self, constituents: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all constituent stocks.
        
        Parameters:
        -----------
        constituents : List[str], optional
            List of stock tickers. If None, uses the default constituents for the index.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping stock ticker to its historical data
        """
        if constituents is None:
            constituents = self.constituents
        
        if not constituents:
            print(f"No constituents defined for {self.index_name}")
            return {}
        
        print(f"\nFetching data for {len(constituents)} constituent stocks...")
        
        successful = 0
        failed = []
        
        for ticker in constituents:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=self.period, interval=self.interval)
                
                if not data.empty:
                    self.stocks_data[ticker] = data
                    successful += 1
                else:
                    failed.append(ticker)
                    print(f"  Warning: No data for {ticker}")
            
            except Exception as e:
                failed.append(ticker)
                print(f"  Error fetching {ticker}: {str(e)}")
        
        print(f"\nSuccessfully fetched data for {successful}/{len(constituents)} stocks")
        if failed:
            print(f"Failed tickers: {', '.join(failed)}")
        
        return self.stocks_data
    
    def fetch_all_data(self, constituents: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch data for both the index and all constituent stocks.
        
        Parameters:
        -----------
        constituents : List[str], optional
            List of stock tickers. If None, uses the default constituents for the index.
        
        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
            Tuple containing (index_data, stocks_data_dict)
        """
        self.fetch_index_data()
        self.fetch_constituents_data(constituents)
        
        return self.index_data, self.stocks_data
    
    def create_combined_dataframe(self, column: str = 'Close') -> pd.DataFrame:
        """
        Create a combined DataFrame with the specified column for index and all stocks.
        
        Parameters:
        -----------
        column : str, optional
            Column to extract (default: 'Close')
            Valid values: Open, High, Low, Close, Volume
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with dates as index and tickers as columns
        """
        if self.index_data is None or not self.stocks_data:
            print("Data not fetched yet. Call fetch_all_data() first.")
            return pd.DataFrame()
        
        # Start with index data
        combined = pd.DataFrame({self.index_name: self.index_data[column]})
        
        # Add each stock's data
        for ticker, data in self.stocks_data.items():
            if column in data.columns:
                combined[ticker] = data[column]
        
        # Remove any rows with all NaN values
        combined = combined.dropna(how='all')
        
        self.combined_data = combined
        print(f"\nCreated combined DataFrame with shape {combined.shape}")
        print(f"Date range: {combined.index.min()} to {combined.index.max()}")
        
        return combined
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all fetched data.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics (count, mean, std, min, max, etc.)
        """
        if self.combined_data is None:
            self.create_combined_dataframe()
        
        return self.combined_data.describe()
    
    def save_to_csv(self, index_filename: str = None, stocks_filename: str = None, 
                    combined_filename: str = None):
        """
        Save the fetched data to CSV files.
        
        Parameters:
        -----------
        index_filename : str, optional
            Filename for index data
        stocks_filename : str, optional
            Filename prefix for individual stock data
        combined_filename : str, optional
            Filename for combined data
        """
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # Save index data
        if self.index_data is not None and not self.index_data.empty:
            if index_filename is None:
                index_filename = f"{self.index_name}_data_{timestamp}.csv"
            self.index_data.to_csv(index_filename)
            print(f"Index data saved to {index_filename}")
        
        # Save individual stock data
        if stocks_filename and self.stocks_data:
            for ticker, data in self.stocks_data.items():
                filename = f"{stocks_filename}_{ticker}_{timestamp}.csv"
                data.to_csv(filename)
            print(f"Individual stock data saved with prefix {stocks_filename}")
        
        # Save combined data
        if self.combined_data is not None and not self.combined_data.empty:
            if combined_filename is None:
                combined_filename = f"{self.index_name}_combined_{timestamp}.csv"
            self.combined_data.to_csv(combined_filename)
            print(f"Combined data saved to {combined_filename}")


def fetch_index_data(index_name: str, period: str = '1y', interval: str = '1d',
                    constituents: Optional[List[str]] = None,
                    save_csv: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Convenience function to fetch index and constituent stock data.
    
    Parameters:
    -----------
    index_name : str
        Name of the index (e.g., 'CAC40', 'DOW', 'DAX')
    period : str, optional
        Time period to fetch data for (default: '1y')
    interval : str, optional
        Data interval (default: '1d')
    constituents : List[str], optional
        Custom list of stock tickers. If None, uses default constituents.
    save_csv : bool, optional
        Whether to save data to CSV files (default: False)
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]
        Tuple containing (index_data, stocks_data_dict, combined_data)
    
    Example:
    --------
    >>> index_data, stocks_data, combined_data = fetch_index_data('CAC40', period='1y')
    >>> print(combined_data.head())
    """
    fetcher = IndexDataFetcher(index_name, period, interval)
    fetcher.fetch_all_data(constituents)
    combined = fetcher.create_combined_dataframe()
    
    if save_csv:
        fetcher.save_to_csv()
    
    return fetcher.index_data, fetcher.stocks_data, combined


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Index Data Fetcher - Example Usage")
    print("=" * 80)
    
    # Fetch CAC40 data
    fetcher = IndexDataFetcher('CAC40', period='1y')
    index_data, stocks_data = fetcher.fetch_all_data()
    
    # Create combined DataFrame with closing prices
    combined = fetcher.create_combined_dataframe(column='Close')
    
    # Display summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(fetcher.get_summary_statistics())
    
    # Display first few rows
    print("\n" + "=" * 80)
    print("First 5 rows of combined data")
    print("=" * 80)
    print(combined.head())
    
    # Save to CSV
    fetcher.save_to_csv()
