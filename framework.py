## TensorFlow Version 1 from scratch
import numpy as np
# from operators import *

class Graph():
    """Defines the computational graph for forward and backward propagation. Holds four arrays:
    operators, constants, variables, and placeholders. Initializes the graph name with a global
    hidden variable _g, which all nodes call when instantiated."""
    def __init__(self):
        self.operators = []
        self.constants = []
        self.variables = []
        self.placeholders = []
        self.as_default()
    def as_default(self):
        global _g
        _g = self

    def make_graph(self, ordering=None):
        """Allows us to visualize the computation graph directly in a Jupyter notebook.
        must have graphviz module installed. Takes as input the topological sorted ordering
        after calling the Session class"""
        from graphviz import Digraph
        if ordering is None:
            sess = Session()
            ordering = sess.order
        f = Digraph()
        f.attr(rankdir='LR', size='8,5')

        f.attr('node', shape='circle')
        for nd in ordering:
            f.node(str(nd))
            if hasattr(nd,'inputs'):
                for e in nd.inputs:
                    f.edge(str(e),str(nd),label=str(e.output))
        return f

class Operator():
    """Inputs: name=None, defaults to Op+i, for ith operator.
    Operators contain self.inputs incoming nodes, and returns self.outputs.
    during backprop we pass gradients through operators using self.gradient"""
    count=0
    def __init__(self,name=None):
        _g.operators.append(self)
        self.inputs = None
        self.output = None
        self.gradient = None
        if name==None:
            self.name = "Op"+str(Operator.count)
        else:
            self.name=name#+str(Operator.count)
        Operator.count+=1
    def __repr__(self):
        return self.name

class Constant():
    """Inputs: value, name=None. Constants contain a decorator function that prevents
    their value from being reassigned"""
    def __init__(self,value,name=None):
        _g.constants.append(self)
        self.output=value
        self.gradient=None
        self.name = (str(value) if name==None else name)
    def __repr__(self):
        return self.name
    @property
    def value(self):
        return self.__value
    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")

class Variable():
    """Inputs: value, name=None. If name is not specified, defaults to the ith variable.
    These are the weights of the model. Variables will be updated during training Using
    the parameter self.gradient"""
    count=0
    def __init__(self, value, name=None):
        _g.variables.append(self)
        self.output = value
        self.gradient = None
        self.name = ('Var'+str(Variable.count) if name==None else name)
        Variable.count+=1
    def __repr__(self):
        return self.name

class Placeholder():
    """Input: name, ptype=float. A placeholder for input types"""
    def __init__(self, name, ptype=float):
        _g.placeholders.append(self)
        self.output = None
        self.gradient = None
        self.name = name
    def __repr__(self):
        return self.name



class Session():
    """Opening a session locks the current Graph in place. __init__ function
    first runs a topological sort placing operators in order for forward/backward prop.
    Note we need only do this for operators, as all other nodes are assumed to have
    defined outputs."""
    def __init__(self):
        self.order = self._topological_sort()
        self.loss_history = []

    def _topological_sort(self):
        """Performs topological sort on nodes in _g.operators. Returns a list
        'ordering', which from left to right gives operators for forward prop."""
        unvis = set(_g.operators+_g.placeholders)
        ordering = []
        def _dfs(node):
            unvis.discard(node)
            for input_node in node.inputs:
                if isinstance(input_node, Operator) and input_node in unvis:
                    _dfs(input_node)
            ordering.append(node)
        while unvis:
            _dfs(unvis.pop())
        return ordering

    def forward(self, head_node=None, feed_dict={}):
        """inputs: head_node (desired node to find), feed_dict (dictionary containing
        values for all placeholders) feed_dict must have strings matching Placeholder.name"""
        # if target node to compute not specified, default to computing output on all nodes
        if head_node is None:
            head_node = self.order[-1]
        for node in self.order:
            if isinstance(node, Placeholder):
                node.output = feed_dict[node.name]
            elif isinstance(node, Operator):
                node.output = node.forward(*[x.output for x in node.inputs])
        # return desired output, though all previous nodes are now computed
        return head_node.output

    def backward(self, target_node=None):
        ### Actually need to fix this later, to allow for not calculating all grads.
        # if none specified, default to finding gradients for all nodes
        # by definition, dJ/dJ = 1
        self.order[-1].gradient = 1
        # traverse list backwards!
        for node in reversed(self.order):
            if isinstance(node,Operator):
                inputs = node.inputs
                # computes gradients for all input nodes. must provide it with dout
                grads = node.backward(*[x.output for x in inputs], dout=node.gradient)
                for input, grad in zip(inputs,grads):
                    # ensure that nodes with multiple outputs sum all contributions
                    input.gradient = grad
                    # input.gradient = (grad if input.gradient is None else input.gradient + grad)

    def update(self, lr=1e-3):
        for var in _g.variables:
            var.output -= lr*var.gradient

    def run(self, feed_dict={}, num_iters=1000, print_every=100, lr=1e-3):

        for i in range(num_iters):
            loss = self.forward(feed_dict=feed_dict)
            self.loss_history.append(loss)
            self.backward()
            self.update(lr=lr)

            if i % print_every == 0:
                print(f"iteration {i}/{num_iters}. Loss = {loss}")


    # def run(self,head_node,feed_dict={}):
    #     head_out, ordering = self.forward_prop(head_node,feed_dict=feed_dict)
    #     self.back_prop(ordering)
    #     return head_out
    #
    #     #return ordering
    # def update(self, learning_rate=1e-3, feed_dict={}):
    #     for var in _g.variables:
    #         if var.value.shape != var.gradient.shape:
    #             var.value -= learning_rate*np.sum(var.gradient,axis=0)
    #         else:
    #             var.value -= learning_rate*var.gradient



############## Begin operators ###########

class affine(Operator):
    def __init__(self, x, W, b, name='affine'):
        super().__init__(name)
        self.inputs=[x, W, b]
    def forward(self, x, W, b):
        """Inputs: x (N,H), W, (H,V), b (V,)
        Outputs: hidden layer (N,H)"""
        h =  b + x.dot(W)
        return h
    def backward(self, x, W, b, dout):
        """dout (H, V), Outputs: dh (N, T, H), dW_out (H, V), db_out (V)"""
        dx, dW, db, = dout.dot(W.T), x.T.dot(dout), np.sum(dout, axis=0)
        return dx, dW, db

class add(Operator):
    """Add two inputs a+b"""
    def __init__(self,a,b,name='add'):
        super().__init__(name)
        self.inputs=[a,b]
    def forward(self,A,B):
        return A+B
    def backward(self,A,B,dout):
        return dout, dout


class multiply(Operator):
    """Multiply two inputs a*b"""
    def __init__(self,a,b,name='multiply'):
        super().__init__(name)
        self.inputs=[a,b]
    def forward(self, A, B):
        return A*B
    def backward(self, A, B, dout):
        return dout*B, dout*A

class matmul(Operator):
    """Matrix multiplication A.dot(B) of matrices A,B"""
    def __init__(self,Mat,Vec,name='matmul'):
        super().__init__(name)
        self.inputs=[Mat,Vec]
    def forward(self, A, B):
        """In: A, B. Out A.B"""
        return A.dot(B)
    def backward(self, A, B, dout):
        """In: A, B. Out: dA, dB"""
        return dout.dot(B.T), (A.T).dot(dout)

class relu(Operator):
    def __init__(self,a,name='relu'):
        """Element-wise relu activation max(0,x)"""
        super().__init__(name)
        self.inputs=[a]
    def forward(self, A):
        """In: A. Out: relu(A)"""
        return np.maximum(0,A)
    def backward(self, A, dout):
        """In: A. Out: dA."""
        return (dout*np.sign(A),)

class sigmoid(Operator):
    def __init__(self,a,name='sigmoid'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[a]
    def forward(self, A):
        return np.where(A >= 0, 1 / (1 + np.exp(-A)), np.exp(A) / (1 + np.exp(A)))
    def backward(self, A, dout):
        return (dout*self.forward(A)*(1-self.forward(A)),)

class softmax_loss(Operator):
    """Inputs: x (N,D), y (N). x is score matrix, y is list of N integers
    with values y[i] in [0,D) of correct output.
    Outputs: loss, dx"""
    def __init__(self, x, y, name='softmax_loss'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[x,y]
    def forward(self, x, y):
        N = len(y)
        probs = np.exp(x-np.max(x, axis=1, keepdims=True))
        norms = np.sum(probs, axis=1, keepdims=True)
        probs = probs/norms
        loss = -np.sum(np.log(probs[np.arange(N),y]))/N
        dx = probs
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss
    def backward(self, x, y, dout=1):
        N = len(y)
        probs = np.exp(x-np.max(x, axis=1, keepdims=True))
        norms = np.sum(probs, axis=1, keepdims=True)
        probs = probs/norms
        probs[np.arange(N), y] -= 1
        dx = probs/N
        return dout*dx, None
    def predict(self, x):
        return np.argmax(probs, axis=1)




################ Old stuff #############
# class log_loss(Operator):
#     def __init__(self,a,b,name='log-loss'):
#         super().__init__(name)
#         self.inputs=[a,b]
#         self.output = None
#         self.gradient = None
#     def forward(self, A, y):
#         return -np.sum(y*np.log(A)+(1-y)*np.log(1-A))
#     def backward(self, A, y, out=1):
#         return -np.true_divide(A-y,A*(1-A)), 0
#
# class softmax(Operator):
#     def __init__(self, X_t, y_t, name='softmax'):
#         super().__init__(name)
#         self.inputs = [X_t,y_t]
#         self.output = None
#         self.gradient = None
#     def forward(self, X, y):
#         scores = np.exp(X-np.max(X, axis=1, keepdims=True))
#         return -np.mean(np.log(scores[range(X.shape[0]),y]/np.sum(scores,axis=1,keepdims=True)))
#     def backward(self, X, y, out=1):
#         scores = np.exp(X-np.max(X, axis=1, keepdims=True))
#         probs = scores/np.sum(scores, axis=1, keepdims=True)
#         probs[range(X.shape[0]),y] -= 1
#         return probs, y
#     def predict(self, X):
#         return np.argmax(X, axis=1)
#
# class svm(Operator):
#     def __init__(self, X_t, y_t, name='svm'):
#         super().__init__(name)
#         self.inputs = [X_t,y_t]
#         self.output = None
#         self.gradient = None
#     def forward(self, X, y):
#         Y = (y.reshape(-1,1)==range(X.shape[1]))
#         mat = (1-Y)*np.maximum(X-X[Y].reshape(-1,1)+1,0)
#         return np.mean(mat)
#     def backward(self, X, y, out=1):
#         Y = (y.reshape(-1,1)==range(X.shape[1]))
#         mat = (1-Y)*np.heaviside(X-X[Y].reshape(-1,1)+1,0)
#         mat -= Y*np.sum(mat, axis=1, keepdims=True)
#         mat = mat/mat.shape[0]
#         return mat, y
#     def predict(self, X):
#         return np.argmax(X, axis=1)
