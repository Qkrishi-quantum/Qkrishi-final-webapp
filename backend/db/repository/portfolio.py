from db.models.portfolio import Portfolio
from schemas.portfolio import CreatePortfolio
from datetime import datetime
from schemas.portfolio import Update
from sqlalchemy.orm import Session


def create_new_portfolio(portfolio: CreatePortfolio, db: Session, owner_id: int = 1):
    portfolio = Portfolio(**portfolio, owner_id=owner_id, date_created=datetime.now())
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def retreive_portfolio(id: int, db: Session):
    portfolio = db.query(Portfolio).filter(Portfolio.id == id).first()
    return portfolio


def list_portfolio(id: int, db: Session):
    portfolio = db.query(Portfolio).filter(Portfolio.owner_id == id).all()
    return portfolio


def update_portfolio(id: int, portfolio: Update, owner_id: int, db: Session):
    pf = db.query(Portfolio).filter(Portfolio.id == id).first()
    if not pf:
        return {"error": f"Blog with id {id} does not exist"}
    if not pf.owner_id == owner_id:
        return {"error": "Only the author can modify the blog"}
    pf.holdings = portfolio.holdings
    db.add(pf)
    db.commit()
    return pf


def delete_portfolio(id: int, owner_id: int, db: Session):
    portfolio = db.query(Portfolio).filter(Portfolio.id == id)
    if not portfolio.first():
        return {"error": f"Could not find portfolio with id {id}"}
    if not portfolio.first().owner_id == owner_id:
        return {"error": "Only the author can delete a blog"}
    portfolio.delete()
    db.commit()
    return {"msg": f"Deleted portfolio with id {id}"}
