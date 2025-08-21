import math 
 
class Value:
    def __init__(self, data, _children=(), op=""):
        self.data = data
        self._prev = _children
        self.op = op
        self.grad = 0
        self._backward = lambda: None
        
    def __add__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        out = Value(self.data + other.data, (self,other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
            
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __truediv__(self, other):
        return self * (other**-1.0)
    
    def __repr__(self):
        return f"Data([{self.data}])"
        
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self+other

    def __rtruediv__(self, other):
        return self / other

    def __pow__(self, other):
        assert isinstance(other, float), "Only float valeus are allowed for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad +=  ((other * self.data)**(other-1)) * out.grad 
        out._backward = _backward
        
        return out

    def exp(self):
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")
        
        def _backward():
            self.grad += result * out.grad

        out._backward = _backward
        
        return out

    def log(self):
        result =  math.log(self.data)
        out = Value(result, (self,), "log")

        def _backward():
            self.grad += (self.data**-1.0) * out.grad 
        out._backward = _backward()

        return out

    def tanh(self):
        e = math.exp((2 * self.data))
        result = (e-1)/(e+1)
        out = Value(result, (self,), "tanh")

        def _backward():
           self.grad += (1-(result**2)) * out.grad
        out._backward = _backward

        return out 

    def relu(self):
        result = max(self.data, 0)
        out = Value(result, (self,), "relu")
        def _backward():
            if(self.data>0):
                print(f"{self.data} greater then 0")
                self.grad += 1.0 *out.grad
            else:
                print("something")
                self.grad += 0
            # self.grad += 1.0 if self.data>0.1 else 0 * out.grad
        out._backward = _backward

        return out
        
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


        
