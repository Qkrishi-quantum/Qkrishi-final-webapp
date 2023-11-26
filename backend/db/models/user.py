from db.base_class import Base
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import DateTime, Date
from datetime import datetime, date
from sqlalchemy.orm import relationship


class User(Base):
    id = Column(Integer, primary_key=True, unique=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False, unique=False)
    is_active = Column(Boolean, default=True)
    portfolios = relationship("Portfolio", back_populates="owner")
    # date_created = Column(DateTime, default=datetime.utcnow)
    date_created = Column(Date, default=date.today())
    # date_created = Column(Date, default=date.today().strftime('%Y-%d-%m'))

