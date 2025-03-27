"""
Advanced Feature Engineering for Financial Time Series
- Technical Indicators
- Statistical Features
- Time-Based Features
- Window Operations
- Fourier/Wavelet Transforms
- Custom Feature Creation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pydantic import BaseModel, ValidationError, validator, Field
from scipy import fft, signal
from sklearn.decomposition import PCA
import talib as ta
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureConfig(BaseModel):
    """Configuration model for feature engineering operations"""
    technical_indicators: Dict[str, Dict] = {
        "SMA": {"window": 20},
        "RSI": {"window": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9}
    }
    fourier_components: Optional[int] = None
    wavelet_features: bool = False
    pca_components: Optional[int] = None
    time_features: bool = True
    lag_features: List[int] = [1, 3, 5, 7]
    volatility_windows: List[int] = [7, 14, 21]
    correlation_windows: List[int] = [30, 60]
    entropy_windows: List[int] = [50, 100]
    cleanup: bool = True

class FeatureValidator(BaseModel):
    """Validation model for engineered features"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    SMA_20: Optional[float]
    RSI_14: Optional[float] = Field(None, ge=0, le=100)
    MACD: Optional[float]
    MACD_signal: Optional[float]
    MACD_hist: Optional[float]
    volatility_14D: Optional[float]
    fourier_0: Optional[float]
    fourier_1: Optional[float]
    weekday: Optional[int] = Field(None, ge=0, le=6)

    @validator('high')
    def validate_high(cls, v, values):
        if 'open' in values and v < values['open']:
            raise ValueError("High price < Open price")
        return v

    @validator('low')
    def validate_low(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError("Low price > Open price")
        return v

class FeatureEngineer:
    """Advanced feature engineering pipeline for financial data"""
    
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.pca = None
        self.scaler = None
        self.feature_stats = {}
        self.ta_functions = {
            "SMA": self._add_sma,
            "EMA": self._add_ema,
            "RSI": self._add_rsi,
            "MACD": self._add_macd,
            "BBANDS": self._add_bollinger_bands,
            "STOCH": self._add_stochastic,
            "ADX": self._add_adx,
            "OBV": self._add_obv
        }

    def _add_sma(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        df[f'SMA_{window}'] = ta.SMA(df['close'], timeperiod=window)
        return df

    def _add_ema(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        df[f'EMA_{window}'] = ta.EMA(df['close'], timeperiod=window)
        return df

    def _add_rsi(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        df[f'RSI_{window}'] = ta.RSI(df['close'], timeperiod=window)
        return df

    def _add_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        macd, signal_line, hist = ta.MACD(df['close'],
                                         fastperiod=fast,
                                         slowperiod=slow,
                                         signalperiod=signal)
        df['MACD'] = macd
        df['MACD_signal'] = signal_line
        df['MACD_hist'] = hist
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, window: int, dev: float = 2) -> pd.DataFrame:
        upper, middle, lower = ta.BBANDS(df['close'],
                                        timeperiod=window,
                                        nbdevup=dev,
                                        nbdevdn=dev)
        df[f'BB_upper_{window}'] = upper
        df[f'BB_middle_{window}'] = middle
        df[f'BB_lower_{window}'] = lower
        df[f'BB_width_{window}'] = (upper - lower) / middle
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        dt_index = pd.to_datetime(df['timestamp'])
        df['hour'] = dt_index.hour
        df['weekday'] = dt_index.weekday
        df['month'] = dt_index.month
        df['quarter'] = dt_index.quarter
        df['year'] = dt_index.year
        df['is_month_end'] = dt_index.is_month_end.astype(int)
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling volatility metrics"""
        returns = df['close'].pct_change()
        for window in self.config.volatility_windows:
            df[f'volatility_{window}D'] = returns.rolling(window).std() * np.sqrt(252)
            df[f'zscore_{window}D'] = (
                (df['close'] - df['close'].rolling(window).mean()
            ) / df['close'].rolling(window).std()
        return df

    def _add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform-based features"""
        close_fft = fft.fft(np.asarray(df['close'].values))
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

        for num_ in range(0, self.config.fourier_components):
            df[f'fourier_{num_}'] = fft_df['absolute'].values[num_]
        return df

    def _add_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet transform features"""
        widths = np.arange(1, 31)
        cwt_matrix = signal.cwt(df['close'].values,
                               signal.ricker,
                               widths)
        df['wavelet_energy'] = np.sum(cwt_matrix**2, axis=0)
        df['wavelet_entropy'] = -np.sum(cwt_matrix**2 * np.log(cwt_matrix**2), axis=0)
        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged price features"""
        for lag in self.config.lag_features:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        return df

    def _add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlations"""
        for window in self.config.correlation_windows:
            df[f'corr_close_vol_{window}'] = (
                df['close'].rolling(window).corr(df['volume'])
            )
        return df

    def _add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate approximate entropy for volatility"""
        def _approximate_entropy(x):
            r = 0.2 * np.std(x)
            return np.mean(np.abs(np.diff(x)) / r
            
        for window in self.config.entropy_windows:
            df[f'entropy_{window}'] = (
                df['close'].rolling(window).apply(_approximate_entropy)
            )
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean intermediate NaN values"""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df.dropna()

    def validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate engineered features against schema"""
        validated_data = []
        errors = []
        for _, row in df.iterrows():
            try:
                validated = FeatureValidator(**row.to_dict())
                validated_data.append(validated.dict())
            except ValidationError as e:
                errors.append({"index": _, "error": str(e)})
        if errors:
            logger.error(f"Feature validation failed for {len(errors)} rows")
            raise ValidationError(errors)
        return pd.DataFrame(validated_data)

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline")

        # Technical Indicators
        for indicator, params in self.config.technical_indicators.items():
            if indicator in self.ta_functions:
                df = self.ta_functions[indicator](df, **params)

        # Time Features
        if self.config.time_features:
            df = self._add_time_features(df)

        # Volatility Features
        df = self._add_volatility_features(df)

        # Fourier Features
        if self.config.fourier_components:
            df = self._add_fourier_features(df)

        # Wavelet Features
        if self.config.wavelet_features:
            df = self._add_wavelet_features(df)

        # Lagged Features
        df = self._add_lagged_features(df)

        # Correlation Features
        df = self._add_correlation_features(df)

        # Entropy Features
        df = self._add_entropy_features(df)

        # Dimensionality Reduction
        if self.config.pca_components:
            numeric_cols = df.select_dtypes(include=np.number).columns
            self.pca = PCA(n_components=self.config.pca_components)
            pca_features = self.pca.fit_transform(df[numeric_cols])
            df = pd.concat([
                df,
                pd.DataFrame(pca_features, 
                            columns=[f'PCA_{i}' for i in range(pca_features.shape[1])],
                axis=1
            )

        # Final Validation and Cleaning
        df = self.validate_features(df)
        if self.config.cleanup:
            df = self._clean_data(df)

        logger.info(f"Engineered {len(df.columns)} features")
        return df

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """Load data from various formats"""
        path = Path(path)
        if path.suffix == ".csv":
            return pd.read_csv(path, parse_dates=['timestamp'])
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".json":
            return pd.read_json(path, orient='records')
        raise ValueError(f"Unsupported file format: {path.suffix}")

# Example Usage
if __name__ == "__main__":
    # Sample Configuration
    config = FeatureConfig(
        technical_indicators={
            "SMA": {"window": 20},
            "RSI": {"window": 14},
            "MACD": {"fast": 12, "slow": 26, "signal": 9}
        },
        fourier_components=3,
        lag_features=[1, 3, 5],
        volatility_windows=[7, 14],
        pca_components=5
    )

    engineer = FeatureEngineer(config)
    data = engineer.load_data("data/processed/clean_data.parquet")
    
    try:
        engineered_data = engineer.run_pipeline(data)
        engineered_data.to_parquet("data/features/final_features.parquet")
        print(f"Engineered features saved. Shape: {engineered_data.shape}")
    except ValidationError as e:
        logger.error(f"Feature validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
