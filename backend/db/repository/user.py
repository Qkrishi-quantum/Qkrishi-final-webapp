from core.hashing import Hasher
from fastapi import HTTPException
import email_validator as _email_check
from db.models.user import User
from schemas.user import CreateUser
from sqlalchemy.orm import Session


from datetime import datetime, date


def create_new_user(user: CreateUser, db: Session):
    try:
        valid = _email_check.validate_email(email=user.email)
        user.email = valid.email
    except _email_check.EmailNotValidError:
        raise HTTPException(
            status_code=404, detail="Please enter a valid email"
        )

    user = User(
        email=user.email,
        password=Hasher.get_password_hash(user.password),
        is_active=True,
        date_created=date.today(),
        # date_created=date.today().strftime('%Y-%d-%m'),
        name=user.name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(email: str, db: Session):
    return db.query(User).filter(User.email == email).first()
