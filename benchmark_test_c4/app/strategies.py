from abc import ABC, abstractmethod

class OrderStrategy(ABC):
    @abstractmethod
    def process(self, order):
        pass

class StandardStrategy(OrderStrategy):
    def process(self, order):
        return f'Processed standard order {order.id}'

class ExpressStrategy(OrderStrategy):
    def process(self, order):
        return f'Processed express order {order.id}'

class DigitalStrategy(OrderStrategy):
    def process(self, order):
        return f'Processed digital order {order.id}'

STRATEGIES = {
    'standard': StandardStrategy(),
    'express': ExpressStrategy(),
    'digital': DigitalStrategy()
}
