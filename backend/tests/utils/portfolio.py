from db.repository.portfolio import create_new_portfolio
from schemas.portfolio import CreatePortfolio
from sqlalchemy.orm import Session
from tests.utils.user import create_random_user


def create_random_portfolio(db: Session):
    portfolio = CreatePortfolio(holdings={"key": " "})
    user = create_random_user(db=db)
    portfolio = create_new_portfolio(portfolio=portfolio, db=db, owner_id=user.id)
    return portfolio
