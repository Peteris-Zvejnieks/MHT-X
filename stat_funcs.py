class statFunc():
    def __init__(self, f, conditions):
        self.f = f
        self.conditions = conditions

    def __repr__(self):
        return str(self.conditions)

    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)

    def check_conditions(self, Y1, Y2):
        if len(Y1) > 1: a ='n'
        else: a = len(Y1)
        if len(Y2) > 1: b ='n'
        else: b = len(Y2)
        return (self.conditions[0] == a) and (self.conditions[1] == b)
