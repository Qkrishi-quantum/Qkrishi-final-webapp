from datetime import datetime

from db.base_class import Base
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import DateTime, Date

from sqlalchemy import ForeignKey
from sqlalchemy import Integer, Float

from sqlalchemy import JSON
from sqlalchemy.orm import relationship


class Portfolio(Base):
    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="portfolios")
    holdings = Column(JSON)
    optimal_num_stocks = Column(JSON)
    risk_ret = Column(JSON)
    initial_investment = Column(Float)
    date_created = Column(Date, default=datetime.today().strftime('%Y-%d-%m'))

