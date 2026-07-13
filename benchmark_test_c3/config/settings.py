import os

class Settings:
    def __init__(self):
        self.API_KEY = os.getenv('API_KEY')
        self.DB_PASS = os.getenv('DB_PASS')

settings = Settings()
