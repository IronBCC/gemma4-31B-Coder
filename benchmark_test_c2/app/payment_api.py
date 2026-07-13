import time

class PaymentAPI:
    def __init__(self):
        self.fail_mode = False

    def charge(self, amount):
        if self.fail_mode:
            raise Exception('API Connection Error')
        return f'Charged {amount}'