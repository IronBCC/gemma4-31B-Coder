import unittest
from models.user import User

class TestUser(unittest.TestCase):
    def test_email_assignment(self):
        u = User(1, 'test@example.com')
        self.assertEqual(u.primary_email, 'test@example.com')

if __name__ == '__main__':
    unittest.main()