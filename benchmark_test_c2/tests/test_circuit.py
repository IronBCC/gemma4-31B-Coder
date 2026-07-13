import unittest
import time
from app.payment_api import PaymentAPI
from app.circuit_breaker import CircuitBreaker

class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_trips(self):
        api = PaymentAPI()
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        api.fail_mode = True
        
        for _ in range(3):
            try: cb.call(api.charge, 100)
            except: pass
        
        self.assertEqual(cb.state, 'OPEN')
        
        with self.assertRaises(Exception) as cm:
            cb.call(api.charge, 100)
        self.assertEqual(str(cm.exception), 'Circuit Breaker is OPEN')

    def test_circuit_resets(self):
        api = PaymentAPI()
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        api.fail_mode = True
        try: cb.call(api.charge, 100)
        except: pass
        
        self.assertEqual(cb.state, 'OPEN')
        time.sleep(1.1)
        
        api.fail_mode = False
        result = cb.call(api.charge, 100)
        self.assertEqual(result, 'Charged 100')
        self.assertEqual(cb.state, 'CLOSED')

if __name__ == '__main__':
    unittest.main()