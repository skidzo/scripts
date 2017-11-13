def some_function(x, y=2):
    a = x-y
    return a + x * y

def _slow_helper(a):
    return a + 1

class B:
    def __init__(self, b=0.):
        self.a = 3.
        self.b = b

    def foo(self, x):
        return (self.a*self.b*x + _slow_helper(1.0))
