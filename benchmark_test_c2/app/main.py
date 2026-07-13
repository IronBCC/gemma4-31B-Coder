from app.payment_api import PaymentAPI
from app.circuit_breaker import CircuitBreaker

api = PaymentAPI()
breaker = CircuitBreaker()

# Mock endpoint for health check
def get_circuit_status():
    return {'state': breaker.state, 'failures': breaker.failures}