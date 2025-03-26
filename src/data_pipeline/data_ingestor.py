"""
Data Ingestion Pipeline - Unified interface for financial data acquisition
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Type, Any
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, ValidationError
import json

from api_integration.alpha_vantage_client import AlphaVantageClient
from api_integration.coingecko_client import CoinGeckoClient
from api_utils import APIError, DataValidationError, retry_api, CacheManager

logger = logging.getLogger(__name__)

class MarketDataModel(BaseModel):
    """Base Pydantic model for market data validation"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class DataIngestor:
    """Unified data ingestion pipeline with validation and format support"""
    
    SUPPORTED_FORMATS = ['csv', 'json', 'parquet']
    DEFAULT_CHUNK_SIZE = 10000  # Records per chunk

    def __init__(self,
                 client: Union[AlphaVantageClient, CoinGeckoClient] = None,
                 cache_dir: str = "./data/raw"):
        self.client = client or AlphaVantageClient()
        self.cache = CacheManager()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry_api(max_retries=3, backoff_base=2)
    def ingest_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str = 'ohlcv',
        output_format: str = 'csv',
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Union[pd.DataFrame, List[Path]]:
        """
        Main ingestion method with format conversion and chunking support
        
        Args:
            symbol: Asset symbol/ticker
            start_date: Start of data range
            end_date: End of data range
            data_type: Type of data (ohlcv, fundamentals, etc.)
            output_format: Output file format
            chunk_size: Records per file chunk
            
        Returns:
            DataFrame if chunk_size=None, else list of file paths
        """
        # Validate inputs
        self._validate_inputs(output_format, start_date, end_date)

        # Fetch data from API
        raw_data = self._fetch_from_source(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type
        )

        # Validate and transform data
        validated_data = self._validate_data(raw_data, MarketDataModel)

        # Convert to pandas DataFrame
        df = pd.DataFrame([item.dict() for item in validated_data])

        # Handle output formatting
        if chunk_size:
            return self._save_in_chunks(df, symbol, output_format, chunk_size)
        return self._convert_to_format(df, output_format)

    def _validate_inputs(self, output_format: str, *dates):
        """Validate input parameters"""
        if output_format.lower() not in self.SUPPORTED_FORMATS:
            raise DataValidationError(
                f"Unsupported format: {output_format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        for date in dates:
            if isinstance(date, str):
                try:
                    datetime.fromisoformat(date)
                except ValueError:
                    raise DataValidationError(f"Invalid date format: {date}")

    def _fetch_from_source(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str
    ) -> List[Dict]:
        """Fetch raw data from configured API client"""
        try:
            if isinstance(self.client, AlphaVantageClient):
                return self.client.get_time_series(
                    symbol=symbol,
                    interval='daily',
                    output_size='full'
                )
            elif isinstance(self.client, CoinGeckoClient):
                return self.client.get_historical_data(
                    coin_id=symbol,
                    days=(end_date - start_date).days,
                    interval='daily'
                )
            else:
                raise APIError("Unsupported API client configured")
        except Exception as e:
            logger.error(f"Data fetch failed: {str(e)}")
            raise

    def _validate_data(
        self,
        raw_data: List[Dict],
        model: Type[BaseModel]
    ) -> List[BaseModel]:
        """Validate data against Pydantic model"""
        validated = []
        errors = []
        
        for idx, item in enumerate(raw_data):
            try:
                validated.append(model(**item))
            except ValidationError as e:
                errors.append({
                    "index": idx,
                    "error": str(e),
                    "data": item
                })
        
        if errors:
            logger.error(f"Validation failed for {len(errors)}/{len(raw_data)} items")
            raise DataValidationError(
                f"Data validation failed with {len(errors)} errors",
                details=errors
            )
            
        return validated

    def _convert_to_format(
        self,
        df: pd.DataFrame,
        output_format: str
    ) -> Union[pd.DataFrame, str]:
        """Convert DataFrame to requested format"""
        output_format = output_format.lower()
        
        if output_format == 'csv':
            return df.to_csv(index=False)
        elif output_format == 'json':
            return df.to_json(orient='records', indent=2)
        elif output_format == 'parquet':
            return df.to_parquet(index=False)
        else:
            return df  # Fallback to DataFrame

    def _save_in_chunks(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_format: str,
        chunk_size: int
    ) -> List[Path]:
        """Save large datasets in chunked files"""
        file_paths = []
        num_chunks = len(df) // chunk_size + 1
        
        for i in range(num_chunks):
            chunk = df[i*chunk_size:(i+1)*chunk_size]
            if chunk.empty:
                continue
                
            filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}_part{i+1}.{output_format}"
            file_path = self.cache_dir / filename
            
            if output_format == 'csv':
                chunk.to_csv(file_path, index=False)
            elif output_format == 'json':
                chunk.to_json(file_path, orient='records')
            elif output_format == 'parquet':
                chunk.to_parquet(file_path, index=False)
                
            file_paths.append(file_path)
            logger.info(f"Saved chunk {i+1}/{num_chunks} to {file_path}")
            
        return file_paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.client, 'close'):
            self.client.close()

# Example Usage
if __name__ == "__main__":
    with DataIngestor(client=CoinGeckoClient()) as ingestor:
        try:
            result = ingestor.ingest_data(
                symbol="bitcoin",
                start_date="2023-01-01",
                end_date="2023-12-31",
                output_format="parquet",
                chunk_size=5000
            )
            print(f"Ingested {len(result)} files")
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
