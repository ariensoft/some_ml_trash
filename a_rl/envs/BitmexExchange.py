class BitmexExchange:

    def __init__(self, order_limit=2):
        self.orders = []
        self.long = {}
        self.short = {}
        self.order_limit = order_limit
        self.last_id = 0
        self.cp = [0.0, 0.0]  # bid, ask

    def add_order(self, side, contracts, price):
        if len(self.orders) < self.order_limit:
            new_order = {
                'id': self.last_id,
                'side': side,
                'contracts': contracts,
                'price': price
            }
            self.orders.append(new_order)
            self.last_id += 1
        else:
            return False
        return len(self.orders)

    def cancel_order(self, order_id):
        if len(self.orders) > 0:
            to_delete = -1
            for k, order in enumerate(self.orders):
                if order['id'] == order_id:
                    to_delete = k
            del self.orders[to_delete]
            return True
        else:
            return False

    def order_list(self):
        if len(self.orders) > 0:
            return self.orders
        else:
            return False

    def update(self, current_prices):
        self.cp = current_prices
        if len(self.orders) > 0:
            for order in self.orders:
                if order['side'] == 0 and order['price'] >= self.cp[1]:
                    self.go_long(order)
                if order['side'] == 1 and order['price'] <= self.cp[0]:
                    self.go_short(order)

    # buying at ASK!
    # buy order will be executed anywhere above current price
    # sell order will be executed anywhere below current price

    def go_long(self, order):
        order['exec_price'] = self.cp[1]
        order['profit'] = ((1 / order['exec_price']) - (1 / self.cp[1])) * order['contracts']
        if len(self.short) == 0:
            if len(self.long) == 0:
                self.long = order
                self.cancel_order(order['id'])
            else:
                self.long['contracts'] += order['contracts']
                avg_price = (self.long['exec_price'] + order['exec_price']) / 2
                self.long['profit'] = ((1 / avg_price) - (1 / self.cp[1])) * self.long['contracts']
                self.cancel_order(order['id'])
        else:
            pass

    def go_short(self, order):
        pass

    def show_state(self):
        print(self.long)
        print(self.short)