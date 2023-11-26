from typing import List
import json

from apis.v1.route_login import get_current_user
from db.models.user import User
from db.repository.portfolio import create_new_portfolio
from db.repository.portfolio import delete_portfolio
from db.repository.portfolio import list_portfolio
from db.repository.portfolio import retreive_portfolio
from db.repository.portfolio import update_portfolio
from db.session import get_db
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from schemas.portfolio import InputJSON
from schemas.portfolio import ShowPortfolio, ShowPortfolioValuation
from schemas.portfolio import Update
from sqlalchemy.orm import Session
from core_business_logic.core_logic import dwave_classical_portfolio #SinglePeriod
from core_business_logic.get_portfolio_value import calculate_portfolio_value
from fastapi.responses import JSONResponse


router = APIRouter()


@router.post(
    "/api/portfolio", response_model=ShowPortfolio, status_code=status.HTTP_201_CREATED
)
def create_portfolio(
    data_json_file: InputJSON,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    my_portfolio = dwave_classical_portfolio(data_json_file)
    input_data = my_portfolio.portfolio_dwave()
    portfolio = create_new_portfolio(
        portfolio=input_data, db=db, owner_id=current_user.id
    )
    # print("type of portfolio obj is ", type(portfolio))
    return portfolio


@router.get("/api/portfolio/{id}", response_model=ShowPortfolioValuation)
def get_portfolio(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    portfolio = retreive_portfolio(id=id, db=db)

    if not portfolio:
        raise HTTPException(
            detail=f"Portfolio with ID {id} does not exist.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    if portfolio.owner_id != current_user.id:
        raise HTTPException(
            detail="Access to this portfolio is forbidden.",
            status_code=status.HTTP_403_FORBIDDEN,
        )

    # Calculate the portfolio values and stock gains/losses
    portfolio_valueation_dict, stock_gain_loss, portfolo_start_date, fd_returns_dict, fd_projection, portfolio_projection = calculate_portfolio_value(portfolio).get_portfolio_vals()

    # Create a dictionary with the required structure
    response_data = {
        "valuation": portfolio_valueation_dict,
        "stocks_gain_loss": stock_gain_loss, 
        "portfolio_start_date" :portfolo_start_date, 
        "fd_returns" : fd_returns_dict,
        'fd_projections': fd_projection, 
        'portfolio_projections': portfolio_projection,
        
    } # this is working fine, just needs to change the formating of the date for fd_returns part. 
    
    # will need to ask kunal to make the yearly returns and risk, and get its plot in the front end only. explain the formula. 
    # and for plotting fd, need to be careful about dates. 
    
    ## i have portfolio_valueation, and risk, returns of the portfolio based on historical data, how can plot the line chart of its projection.
    
    ### talk about this projection to venkat sir. it makes user worried about his own investment. 

    return response_data  # Return the dictionary directly


# @authenticate_user
@router.get("/api/listportfolio", response_model=List[ShowPortfolio])
def get_all_portfolio(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    portfolios = list_portfolio(id=current_user.id, db=db)
    # print("type of portfolio obj is ", type(portfolios))

    return portfolios


@router.put("/api/editportfolio/{id}", response_model=ShowPortfolio)
def update_a_portfolio(
    id: int,
    portfolio: Update,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    portfolio = update_portfolio(
        id=id, portfolio=portfolio, author_id=current_user.id, db=db
    )
    if isinstance(portfolio, dict):
        raise HTTPException(
            detail=portfolio.get("error"),
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return portfolio


@router.delete("/api/delete/{id}")
def delete_a_portfolio(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    message = delete_portfolio(id=id, author_id=current_user.id, db=db)
    if message.get("error"):
        raise HTTPException(
            detail=message.get("error"), status_code=status.HTTP_400_BAD_REQUEST
        )
    return {"msg": f"Successfully deleted blog with id {id}"}
