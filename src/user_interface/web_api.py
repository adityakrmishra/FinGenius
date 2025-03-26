"""
Web API Interface - REST API for financial operations
"""

from fastapi import FastAPI, APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import date
from typing import Optional
from simulation.backtester import Backtester
from simulation.monte_carlo import MonteCarloSimulator
from simulation.visualizer import Visualizer
import uvicorn

app = FastAPI(
    title="FinGenius API",
    description="REST API for Financial Analysis",
    version="1.0.0"
)

class BacktestRequest(BaseModel):
    strategy: str
    start_date: date
    end_date: date
    initial_capital: float = 100000.0

class MonteCarloRequest(BaseModel):
    ticker: str
    simulations: int = 10000
    time_horizon: int = 252
    confidence_level: float = 0.95

@app.get("/")
async def root():
    return {"message": "FinGenius Financial API"}

@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Execute strategy backtesting"""
    try:
        data = load_market_data(request.ticker, request.start_date, request.end_date)
        bt = Backtester(
            data=data,
            initial_capital=request.initial_capital
        )
        results = bt.run_backtest()
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/montecarlo")
async def run_monte_carlo(
    ticker: str,
    simulations: int = Query(10000, gt=0),
    days: int = Query(252, gt=0),
    confidence: float = Query(0.95, gt=0, lt=1)
):
    """Run Monte Carlo simulation"""
    try:
        returns = load_returns(ticker)
        mc = MonteCarloSimulator(
            returns=returns,
            num_simulations=simulations,
            time_horizon=days,
            confidence_level=confidence
        )
        mc.run_simulation()
        var, cvar = mc.calculate_var()
        return {
            "var": var,
            "cvar": cvar,
            "simulations": simulations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/historical")
async def get_historical(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
):
    """Get historical market data"""
    try:
        data = load_historical_data(ticker, period, interval)
        return {
            "ticker": ticker,
            "data": data.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def load_historical_data(ticker: str, period: str, interval: str):
    """Mock data loader - integrate with real data source"""
    # Implementation would connect to market data API
    return pd.DataFrame()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
