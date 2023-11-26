from db.repository.user import create_new_user, get_user_by_email
from db.session import get_db
from fastapi import APIRouter, Depends, status, HTTPException
# from fastapi import Depends
# from fastapi import status
# from fastapi import HTTPException
from schemas.user import ShowUser
from schemas.user import CreateUser
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/api/users", response_model=ShowUser, status_code=status.HTTP_201_CREATED)
def create_user(user: CreateUser, db: Session = Depends(get_db)) -> CreateUser:
    # Check if a user with the same email exists
    db_user = get_user_by_email(email=user.email, db=db)
    if db_user:
        # Fix the error message format with double quotes
        raise HTTPException(
            status_code=400, detail="User with that email already exists"
        )
    # Create a new user
    new_user = create_new_user(user=user, db=db)
    return new_user



# @router.post("/api/users", response_model=ShowUser, status_code=status.HTTP_201_CREATED)
# def create_user(user: CreateUser, db: Session = Depends(get_db)) -> CreateUser:
#     # Check if a user with the same email exists
#     db_user = get_user_by_email(email=user.email, db=db)
#     if db_user:
#         # Fix the error message format with double quotes
#         raise HTTPException(
#             status_code=400, detail="User with that email already exists"
#         )
#     # Create a new user
#     new_user = create_new_user(user=user, db=db)
#     return new_user

