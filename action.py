class Action():

    def __init__(self, delta, tr, tf):
        self.delta = delta
        self.tr = tr
        self.tf = tf

    def __call__(self):
        return self.delta, self.tr, self.tf