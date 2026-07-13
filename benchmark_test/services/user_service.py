from models.user import User

def get_user_email(user: User):
    return user.primary_email