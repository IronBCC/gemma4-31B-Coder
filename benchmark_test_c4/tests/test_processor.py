import unittest
from app.models import Order
from app.processor import OrderProcessor

class TestProcessor(unittest.TestCase):
    def test_processing(self):
        proc = OrderProcessor()
        self.assertEqual(proc.process(Order(1, 'standard')), 'Processed standard order 1')
        self.assertEqual(proc.process(Order(2, 'express')), 'Processed express order 2')
        self.assertEqual(proc.process(Order(3, 'digital')), 'Processed digital order 3')

if __name__ == '__main__':
    unittest.main()