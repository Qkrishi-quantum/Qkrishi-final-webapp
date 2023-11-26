from pydantic import BaseModel
from pydantic import EmailStr
from pydantic import Field
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr


# properties required during user creation
class CreateUser(UserBase):
    email: EmailStr
    password: str = Field(..., min_length=4)
    name: str

    class Config:
        from_attributes = True
        # orm_mode = True


class ShowUser(BaseModel):
    id: int
    name: str
    email: EmailStr
    date_created: datetime
    is_active: bool

    class Config:  # tells pydantic to convert even non dict obj to json
        from_attributes = True
        # orm_mode = True
