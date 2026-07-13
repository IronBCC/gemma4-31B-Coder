from app.strategies import STRATEGIES

class OrderProcessor:
    def process(self, order):
        strategy = STRATEGIES.get(order.type)
        if not strategy:
            raise ValueError('Unknown order type')
        return strategy.process(order)
