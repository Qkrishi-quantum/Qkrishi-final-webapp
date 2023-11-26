from datetime import datetime
from typing import Any
from pydantic import BaseModel, validator


class InputJSON(BaseModel):
    risk_factor_value: float
    investment_amount: int
    stock_names: list

    class Config:
        from_attributes = True
        # orm_mode = True


class CreatePortfolio(BaseModel):
    holdings: dict[str, Any]
    optimal_num_stocks: dict[str, Any]
    risk_ret: dict[str, Any]
    investment_amount: float

    class Config:
        from_attributes = True
        # orm_mode = True


class Update(CreatePortfolio):
    pass


class ShowPortfolio(BaseModel):
    id: int
    owner_id: int
    date_created: datetime
    holdings: dict[str, Any]
    optimal_num_stocks: dict[str, Any]
    risk_ret: dict[str, Any]
    initial_investment: float

    class Config:
        from_attributes = True
        # orm_mode = True


# class ShowPortfolioValuation(BaseModel):
#     # id: int
#     # owner_id: int
#     # date_created: datetime
#     portfolio_valueation_dict : dict[str, Any]
#     stocks_gain_loss : dict[str, Any]
#     class Config:
#         from_attributes = True
#         # orm_mode = True


class ShowPortfolioValuation(BaseModel):
    valuation: dict
    stocks_gain_loss: dict
    portfolio_start_date: str
    fd_returns: dict
    fd_projections :dict  
    portfolio_projections : dict
