"""
CoinGecko API Client with advanced features for cryptocurrency data
"""

import os
import time
import logging
import requests
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, AsyncGenerator
from pydantic import BaseModel, Field, validator, HttpUrl
from functools import wraps
from pathlib import Path
from enum import Enum
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoinGeckoError(Exception):
    """Base exception for CoinGecko client errors"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)

class RateLimitExceededError(CoinGeckoError):
    """Exception raised when API rate limits are exceeded"""

class InvalidCurrencyError(CoinGeckoError):
    """Exception raised for invalid currency codes"""

class DataValidationError(CoinGeckoError):
    """Exception raised for data validation failures"""

# --------------------------
# Data Models (Pydantic 2.0)
# --------------------------

class Coin(BaseModel):
    id: str
    symbol: str = Field(..., min_length=3, max_length=10)
    name: str
    platforms: Dict[str, Optional[str]]
    contract_addresses: Dict[str, Optional[str]] = Field(default_factory=dict)

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalpha():
            raise ValueError('Symbol must contain only letters')
        return v.lower()

class MarketData(BaseModel):
    current_price: float
    market_cap: float
    total_volume: float
    price_change_24h: float
    price_change_percentage_24h: float
    market_cap_rank: Optional[int]
    last_updated: datetime

    @validator('last_updated', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class HistoricalData(BaseModel):
    timestamp: datetime
    price: float
    market_cap: float
    total_volume: float

class ExchangeRate(BaseModel):
    currency: str
    rate: float
    last_updated: datetime

class CoinDetails(BaseModel):
    id: str
    symbol: str
    name: str
    asset_platform_id: Optional[str]
    description: Dict[str, str]
    links: Dict[str, Union[HttpUrl, List[HttpUrl]]]
    market_data: MarketData
    developer_score: float
    community_score: float
    liquidity_score: float

# --------------------------
# Decorators
# --------------------------

def retry(max_retries: int = 3, backoff_factor: float = 0.5):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (RateLimitExceededError, requests.RequestException) as e:
                    if isinstance(e, RateLimitExceededError):
                        wait_time = min(60, backoff_factor * (2 ** retries))
                        logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    retries += 1
            raise CoinGeckoError(f"Max retries ({max_retries}) exceeded")
        return wrapper
    return decorator

def cache_response(cache_dir: str = ".cg_cache", ttl: int = 3600):
    """Cache API responses to disk with time-to-live"""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_key = f"{func.__name__}_{args}_{kwargs}".replace(' ', '_')
            cache_file = Path(cache_dir) / f"{hashlib.md5(cache_key.encode()).hexdigest()}.json"
            
            if cache_file.exists():
                modified_time = cache_file.stat().st_mtime
                if (time.time() - modified_time) < ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            
            result = func(self, *args, **kwargs)
            with open(cache_file, 'w') as f:
                json.dump(result, f, default=str)
            return result
        return wrapper
    return decorator

# --------------------------
# Core Client Implementation
# --------------------------

class CoinGeckoClient:
    """Advanced CoinGecko API Client with caching and rate limiting"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    RATE_LIMIT = 10  # Requests per minute
    MAX_CONCURRENT_REQUESTS = 5

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.session = requests.Session()
        self._last_request_time = 0
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)
        self._async_session = None

    # --------------------------
    # Public API Methods
    # --------------------------

    @retry(max_retries=5)
    @cache_response(ttl=300)
    def get_coins_list(self, include_platform: bool = False) -> List[Coin]:
        """Get list of all supported coins"""
        params = {"include_platform": str(include_platform).lower()}
        response = self._make_request("GET", "/coins/list", params=params)
        return [Coin(**coin_data) for coin_data in response]

    @retry(max_retries=3)
    @cache_response(ttl=60)
    def get_coin_by_id(self, coin_id: str) -> CoinDetails:
        """Get detailed information for a specific coin"""
        response = self._make_request("GET", f"/coins/{coin_id}")
        return self._parse_coin_details(response)

    @retry(max_retries=3)
    @cache_response(ttl=300)
    def get_market_data(
        self,
        vs_currency: str = "usd",
        ids: Optional[List[str]] = None,
        per_page: int = 100,
        page: int = 1
    ) -> List[MarketData]:
        """Get market data for coins"""
        params = {
            "vs_currency": vs_currency,
            "ids": ",".join(ids) if ids else None,
            "per_page": per_page,
            "page": page
        }
        response = self._make_request("GET", "/coins/markets", params=params)
        return [MarketData(**market_data) for market_data in response]

    @retry(max_retries=5)
    @cache_response(ttl=600)
    def get_historical_data(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: Union[int, str] = 30,
        interval: str = "daily"
    ) -> List[HistoricalData]:
        """Get historical market data"""
        valid_intervals = ["daily", "hourly", "minute"]
        if interval not in valid_intervals:
            raise DataValidationError(f"Invalid interval. Use one of: {valid_intervals}")

        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": interval
        }
        response = self._make_request("GET", f"/coins/{coin_id}/market_chart", params=params)
        return self._parse_historical_data(response)

    @retry(max_retries=3)
    @cache_response(ttl=3600)
    def get_exchange_rates(self) -> List[ExchangeRate]:
        """Get cryptocurrency exchange rates"""
        response = self._make_request("GET", "/exchange_rates")
        rates = response.get("rates", {})
        return [
            ExchangeRate(
                currency=currency.upper(),
                rate=float(data["value"]),
                last_updated=datetime.fromtimestamp(data["last_updated_at"])
            ) for currency, data in rates.items()
        ]

    @retry(max_retries=3)
    def search_coins(self, query: str) -> List[Coin]:
        """Search for coins by name or symbol"""
        params = {"query": query}
        response = self._make_request("GET", "/search", params=params)
        return [Coin(**coin_data) for coin_data in response.get("coins", [])]

    # --------------------------
    # Async Methods
    # --------------------------

    async def async_get_coins_list(self, include_platform: bool = False) -> List[Coin]:
        """Async version of get_coins_list"""
        params = {"include_platform": str(include_platform).lower()}
        response = await self._async_make_request("GET", "/coins/list", params=params)
        return [Coin(**coin_data) for coin_data in response]

    async def async_get_market_data(
        self,
        vs_currency: str = "usd",
        ids: Optional[List[str]] = None,
        per_page: int = 100,
        page: int = 1
    ) -> List[MarketData]:
        """Async version of get_market_data"""
        params = {
            "vs_currency": vs_currency,
            "ids": ",".join(ids) if ids else None,
            "per_page": per_page,
            "page": page
        }
        response = await self._async_make_request("GET", "/coins/markets", params=params)
        return [MarketData(**market_data) for market_data in response]

    # --------------------------
    # Private Helpers
    # --------------------------

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Core request handler with rate limiting"""
        self._check_rate_limit()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            self._handle_http_error(e)
        except requests.RequestException as e:
            raise CoinGeckoError(f"Request failed: {str(e)}") from e

    async def _async_make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Async core request handler"""
        async with self._semaphore:
            if not self._async_session:
                self._async_session = aiohttp.ClientSession()
            
            try:
                async with self._async_session.request(
                    method,
                    f"{self.BASE_URL}{endpoint}",
                    **kwargs
                ) as response:
                    if response.status != 200:
                        await self._handle_async_http_error(response)
                    return await response.json()
            except aiohttp.ClientError as e:
                raise CoinGeckoError(f"Async request failed: {str(e)}") from e

    def _check_rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < 60 / self.RATE_LIMIT:
            time.sleep((60 / self.RATE_LIMIT) - elapsed)
        self._last_request_time = time.time()

    def _handle_http_error(self, error: requests.HTTPError):
        """Handle HTTP errors"""
        status_code = error.response.status_code
        if status_code == 429:
            raise RateLimitExceededError("Rate limit exceeded", status_code)
        elif status_code == 400:
            raise DataValidationError("Invalid request parameters", status_code)
        elif status_code == 404:
            raise CoinGeckoError("Resource not found", status_code)
        else:
            raise CoinGeckoError(f"HTTP error {status_code}", status_code)

    async def _handle_async_http_error(self, response: aiohttp.ClientResponse):
        """Handle async HTTP errors"""
        status_code = response.status
        try:
            error_data = await response.json()
            message = error_data.get("error", "Unknown error")
        except json.JSONDecodeError:
            message = await response.text()
        
        if status_code == 429:
            raise RateLimitExceededError(f"Rate limit exceeded: {message}", status_code)
        else:
            raise CoinGeckoError(f"HTTP error {status_code}: {message}", status_code)

    def _parse_coin_details(self, data: dict) -> CoinDetails:
        """Parse detailed coin information"""
        return CoinDetails(
            id=data["id"],
            symbol=data["symbol"],
            name=data["name"],
            asset_platform_id=data.get("asset_platform_id"),
            description=data.get("description", {}),
            links=data.get("links", {}),
            market_data=self._parse_market_data(data["market_data"]),
            developer_score=data["developer_score"],
            community_score=data["community_data"]["twitter_followers"],
            liquidity_score=data["liquidity_score"]
        )

    def _parse_market_data(self, data: dict) -> MarketData:
        """Parse market data from API response"""
        return MarketData(
            current_price=data["current_price"],
            market_cap=data["market_cap"],
            total_volume=data["total_volume"],
            price_change_24h=data["price_change_24h"],
            price_change_percentage_24h=data["price_change_percentage_24h"],
            market_cap_rank=data.get("market_cap_rank"),
            last_updated=data["last_updated"]
        )

    def _parse_historical_data(self, data: dict) -> List[HistoricalData]:
        """Parse historical price data"""
        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        historical_data = []
        for price_point, mcap_point, vol_point in zip(prices, market_caps, volumes):
            historical_data.append(HistoricalData(
                timestamp=datetime.fromtimestamp(price_point[0]/1000),
                price=price_point[1],
                market_cap=mcap_point[1],
                total_volume=vol_point[1]
            ))
        return historical_data

    # --------------------------
    # Context Management
    # --------------------------

    async def close(self):
        """Close async session"""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":
    # Sync example
    with CoinGeckoClient() as client:
        btc_data = client.get_coin_by_id("bitcoin")
        print(f"Bitcoin Price: ${btc_data.market_data.current_price:,.2f}")

    # Async example
    async def main():
        async with CoinGeckoClient() as client:
            coins = await client.async_get_coins_list()
            print(f"Total coins: {len(coins)}")

    asyncio.run(main())
