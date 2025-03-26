"""
API Utilities Module - Core infrastructure for financial data integration
"""

import os
import re
import time
import json
import logging
import hashlib
import inspect
import functools
from datetime import datetime, timedelta
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, Union,
                    AsyncGenerator, TypeVar, Generic, cast)
from pathlib import Path
from enum import Enum
import concurrent.futures
from contextlib import contextmanager, asynccontextmanager

import requests
import aiohttp
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, validator, Field
from pydantic.generics import GenericModel
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential, RetryCallState, Retrying)
from cachetools import TTLCache, cachedmethod
from cachetools.keys import hashkey
from cryptography.fernet import Fernet

# Type Aliases
JSONType = Union[Dict[str, Any], List[Any]]
T = TypeVar('T')
R = TypeVar('R', covariant=True)

# --------------------------
# Error Handling Framework
# --------------------------

class APIError(Exception):
    """Base exception for API-related errors"""
    def __init__(self,
                 message: str,
                 status_code: Optional[int] = None,
                 url: Optional[str] = None):
        self.status_code = status_code
        self.url = url
        super().__init__(message)

class RateLimitError(APIError):
    """Rate limit exceeded exception"""
    def __init__(self,
                 reset_time: float,
                 limit: int,
                 window: int,
                 **kwargs):
        message = (f"Rate limit exceeded: {limit} requests per {window} seconds. "
                   f"Reset at {datetime.fromtimestamp(reset_time)}")
        super().__init__(message, **kwargs)
        self.reset_time = reset_time
        self.limit = limit
        self.window = window

class DataValidationError(APIError):
    """Data validation failed exception"""

class APITimeoutError(APIError):
    """API request timed out exception"""

class RetryExhaustedError(APIError):
    """All retry attempts exhausted exception"""

# --------------------------
# Logging Configuration
# --------------------------

def configure_logging(level: int = logging.INFO,
                      file: Optional[str] = None):
    """Configure global logging settings"""
    handlers = [logging.StreamHandler()]
    if file:
        handlers.append(logging.FileHandler(file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.captureWarnings(True)

# --------------------------
# Configuration Management
# --------------------------

class APIConfig(BaseModel):
    """Base configuration model for API clients"""
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 5  # Requests per second
    cache_ttl: int = 300  # Seconds
    timeout: int = 30 # Seconds
    retries: int = 3
    encryption_key: Optional[str] = None

    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if v and len(v) != 44:
            raise ValueError("Invalid encryption key - must be 44 character Fernet key")
        return v

def load_api_config(prefix: str = "API_") -> APIConfig:
    """Load configuration from environment variables"""
    return APIConfig(
        base_url=os.getenv(f"{prefix}BASE_URL", ""),
        api_key=os.getenv(f"{prefix}KEY"),
        rate_limit=int(os.getenv(f"{prefix}RATE_LIMIT", 5)),
        cache_ttl=int(os.getenv(f"{prefix}CACHE_TTL", 300)),
        timeout=int(os.getenv(f"{prefix}TIMEOUT", 30)),
        retries=int(os.getenv(f"{prefix}RETRIES", 3)),
        encryption_key=os.getenv(f"{prefix}ENCRYPT_KEY")
    )

# --------------------------
# Decorators & Utilities
# --------------------------

def retry_api(
    exceptions: Tuple[Type[Exception]] = (APIError,),
    max_retries: int = 3,
    max_delay: int = 60,
    backoff_base: float = 1.5
):
    """Advanced retry decorator with exponential backoff and jitter"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    sleep_time = min(
                        backoff_base ** attempt + np.random.uniform(0, 1),
                        max_delay
                    )
                    time.sleep(sleep_time)
            raise RetryExhaustedError(
                f"Max retries ({max_retries}) exceeded") from last_exc
        return wrapper
    return decorator

def rate_limited(max_per_second: int):
    """Decorator to enforce rate limiting"""
    min_interval = 1.0 / max_per_second
    def decorator(func):
        last_called = 0.0
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_called
            elapsed = time.monotonic() - last_called
            wait = max(min_interval - elapsed, 0)
            if wait > 0:
                time.sleep(wait)
            last_called = time.monotonic()
            return func(*args, **kwargs)
        return wrapper
    return decorator

class CacheManager:
    """Unified cache manager with encryption support"""
    def __init__(self,
                 ttl: int = 300,
                 max_size: int = 1000,
                 encrypt_key: Optional[str] = None):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.cipher = Fernet(encrypt_key) if encrypt_key else None

    def _encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode() if self.cipher else data

    def _decrypt(self, data: str) -> str:
        return self.cipher.decrypt(data.encode()).decode() if self.cipher else data

    def get(self, key: str) -> Any:
        encrypted_key = self._encrypt(key)
        if encrypted_key in self.cache:
            return json.loads(self._decrypt(self.cache[encrypted_key]))
        return None

    def set(self, key: str, value: Any):
        encrypted_key = self._encrypt(key)
        encrypted_value = self._encrypt(json.dumps(value))
        self.cache[encrypted_key] = encrypted_value

def cache_response(
    ttl: int = 300,
    max_size: int = 1000,
    key_func: Optional[Callable] = None
):
    """Secure caching decorator with encryption support"""
    def decorator(func):
        cache = CacheManager(ttl=ttl, max_size=max_size)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = key_func(*args, **kwargs) if key_func else hashkey(*args, **kwargs)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        return wrapper
    return decorator

# --------------------------
# Request Infrastructure
# --------------------------

class APIRequest:
    """Generic API request container"""
    def __init__(self,
                 method: str,
                 endpoint: str,
                 params: Optional[Dict] = None,
                 data: Optional[Dict] = None,
                 headers: Optional[Dict] = None):
        self.method = method.upper()
        self.endpoint = endpoint
        self.params = params or {}
        self.data = data or {}
        self.headers = headers or {}

    def fingerprint(self) -> str:
        """Generate unique request fingerprint"""
        components = [
            self.method,
            self.endpoint,
            json.dumps(self.params, sort_keys=True),
            json.dumps(self.data, sort_keys=True)
        ]
        return hashlib.sha256("|".join(components).encode()).hexdigest()

class APIResponse(GenericModel, Generic[T]):
    """Standard API response model"""
    data: T
    metadata: Dict[str, Any]
    pagination: Optional[Dict[str, Any]]
    request: Dict[str, Any]

    @classmethod
    def from_raw(cls,
                 data: T,
                 request: APIRequest,
                 metadata: Optional[Dict] = None,
                 pagination: Optional[Dict] = None):
        return cls(
            data=data,
            metadata=metadata or {},
            pagination=pagination,
            request={
                "method": request.method,
                "endpoint": request.endpoint,
                "params": request.params
            }
        )

class BaseAPIClient:
    """Abstract base class for API clients"""
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.async_session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_semaphore = concurrent.futures.Semaphore(config.rate_limit)
        self._setup_headers()

    def _setup_headers(self):
        """Initialize default headers"""
        self.session.headers.update({
            "User-Agent": "FinGenius/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        if self.config.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.api_key}"

    @retry_api()
    def request(self, req: APIRequest) -> APIResponse:
        """Execute API request with retry logic"""
        with self._rate_limit_semaphore:
            try:
                response = self.session.request(
                    method=req.method,
                    url=f"{self.config.base_url}{req.endpoint}",
                    params=req.params,
                    json=req.data,
                    headers=req.headers,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return self._process_response(response, req)
            except requests.RequestException as e:
                self._handle_request_error(e, req)

    async def async_request(self, req: APIRequest) -> APIResponse:
        """Execute async API request"""
        if not self.async_session:
            self.async_session = aiohttp.ClientSession(
                headers=self.session.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

        async with self.async_session.request(
            method=req.method,
            url=f"{self.config.base_url}{req.endpoint}",
            params=req.params,
            json=req.data,
            headers=req.headers
        ) as response:
            if response.status != 200:
                raise await self._handle_async_error(response, req)
            return await self._process_async_response(response, req)

    def _process_response(self,
                         response: requests.Response,
                         request: APIRequest) -> APIResponse:
        """Process and validate API response"""
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = response.text

        return APIResponse.from_raw(
            data=data,
            request=request,
            metadata={
                "status": response.status_code,
                "headers": dict(response.headers),
                "latency": response.elapsed.total_seconds()
            }
        )

    async def _process_async_response(self,
                                     response: aiohttp.ClientResponse,
                                     request: APIRequest) -> APIResponse:
        """Process async response"""
        data = await response.json()
        return APIResponse.from_raw(
            data=data,
            request=request,
            metadata={
                "status": response.status,
                "headers": dict(response.headers),
                "latency": response.latency
            }
        )

    def _handle_request_error(self,
                             error: requests.RequestException,
                             request: APIRequest):
        """Handle request errors and transform to APIError"""
        response = getattr(error, 'response', None)
        status_code = response.status_code if response else None
        
        if status_code == 429:
            reset_time = float(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            raise RateLimitError(
                message="Rate limit exceeded",
                status_code=status_code,
                reset_time=reset_time,
                limit=int(response.headers.get('X-RateLimit-Limit', 0)),
                window=int(response.headers.get('X-RateLimit-Window', 60)),
                url=request.endpoint
            )
        raise APIError(
            message=str(error),
            status_code=status_code,
            url=request.endpoint
        )

    async def _handle_async_error(self,
                                 response: aiohttp.ClientResponse,
                                 request: APIRequest) -> APIError:
        """Handle async request errors"""
        try:
            error_data = await response.json()
            message = error_data.get('error', await response.text())
        except json.JSONDecodeError:
            message = await response.text()

        if response.status == 429:
            return RateLimitError(
                message=message,
                status_code=response.status,
                reset_time=time.time() + 60,
                limit=0,
                window=60,
                url=request.endpoint
            )
        return APIError(
            message=message,
            status_code=response.status,
            url=request.endpoint
        )

    def close(self):
        """Clean up resources"""
        self.session.close()
        if self.async_session:
            asyncio.run(self.async_session.close())

# --------------------------
# Data Processing Utilities
# --------------------------

class DataNormalizer:
    """Normalize financial data from different API formats"""
    @staticmethod
    def normalize_ohlcv(data: List[Dict[str, Any]], 
                       source: str) -> pd.DataFrame:
        """Normalize OHLCV data from various sources"""
        mapping = {
            'coingecko': {
                'timestamp': ('time', lambda x: datetime.fromtimestamp(x/1000)),
                'open': ('open', float),
                'high': ('high', float),
                'low': ('low', float),
                'close': ('close', float),
                'volume': ('volumeto', float)
            },
            'alphavantage': {
                'timestamp': ('date', lambda x: datetime.fromisoformat(x)),
                'open': ('1. open', float),
                'high': ('2. high', float),
                'low': ('3. low', float),
                'close': ('4. close', float),
                'volume': ('5. volume', int)
            }
        }

        normalized = []
        for entry in data:
            norm_entry = {}
            for target_field, (source_field, converter) in mapping[source].items():
                norm_entry[target_field] = converter(entry[source_field])
            normalized.append(norm_entry)
        
        return pd.DataFrame(normalized).set_index('timestamp').sort_index()

    @staticmethod
    def normalize_orderbook(data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize order book data to standardized format"""
        return pd.DataFrame({
            'price': list(data['bids'].keys()) + list(data['asks'].keys()),
            'amount': list(data['bids'].values()) + list(data['asks'].values()),
            'side': ['bid'] * len(data['bids']) + ['ask'] * len(data['asks'])
        })

class Paginator:
    """Generic API pagination handler"""
    def __init__(self,
                 client: BaseAPIClient,
                 initial_request: APIRequest,
                 page_size: int = 100,
                 max_pages: int = 10):
        self.client = client
        self.initial_request = initial_request
        self.page_size = page_size
        self.max_pages = max_pages
        self.current_page = 0

    def __iter__(self):
        self.current_page = 0
        return self

    def __next__(self) -> APIResponse:
        if self.current_page >= self.max_pages:
            raise StopIteration

        request = self._build_request()
        response = self.client.request(request)
        self.current_page += 1

        if not self._has_next_page(response):
            raise StopIteration

        return response

    async def __anext__(self) -> APIResponse:
        if self.current_page >= self.max_pages:
            raise StopAsyncIteration

        request = self._build_request()
        response = await self.client.async_request(request)
        self.current_page += 1

        if not self._has_next_page(response):
            raise StopAsyncIteration

        return response

    def _build_request(self) -> APIRequest:
        """Build paginated request"""
        params = self.initial_request.params.copy()
        params.update({
            'page': self.current_page,
            'per_page': self.page_size
        })
        return APIRequest(
            method=self.initial_request.method,
            endpoint=self.initial_request.endpoint,
            params=params
        )

    def _has_next_page(self, response: APIResponse) -> bool:
        """Determine if more pages are available"""
        pagination = response.pagination or {}
        return pagination.get('has_next', False)

# --------------------------
# Date/Time Utilities
# --------------------------

class TimeFrame(Enum):
    """Standardized time intervals"""
    TICK = "tick"
    MINUTE = "1min"
    HOUR = "1hour"
    DAY = "1day"
    WEEK = "1week"
    MONTH = "1month"
    QUARTER = "3month"
    YEAR = "1year"

def convert_timestamp(
    ts: Union[int, float, str, datetime],
    target_tz: Optional[str] = None
) -> datetime:
    """Convert various timestamp formats to datetime"""
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts)
    elif isinstance(ts, str):
        try:  # ISO format
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError:
                       # Try alternative parsers
            try:
                from dateutil.parser import parse as dateutil_parse
            except ImportError:
                dateutil_available = False
            else:
                dateutil_available = True

            if dateutil_available:
                try:
                    dt = dateutil_parse(ts)
                except ValueError:
                    raise DataValidationError(f"Invalid timestamp format: {ts}")
            else:
                # Try common formats manually
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y/%m/%d %H:%M:%S",
                    "%d-%b-%Y %H:%M:%S",  # 01-Jan-2023 12:00:00
                    "%Y%m%d"
                ]
                for fmt in formats:
                    try:
                        dt = datetime.strptime(ts, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise DataValidationError(f"Unparseable timestamp: {ts}")

        # Handle timezone conversion
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        if target_tz:
            try:
                import pytz
                target_tz = pytz.timezone(target_tz)
                dt = dt.astimezone(target_tz)
            except ImportError:
                raise ImportError("pytz required for timezone conversion")
            except pytz.UnknownTimeZoneError:
                raise DataValidationError(f"Unknown timezone: {target_tz}")

    else:
        raise DataValidationError(f"Invalid timestamp type: {type(ts)}")

    return dt
