## TensorFlow Version 1 from scratch
import numpy as np
from graphviz import Digraph
# from operators import *

class Graph():
    """Defines the computational graph for forward and backward propagation. Holds four sets:
    operators, constants, variables, and placeholders. Initializes the graph name with a global
    hidden variable _g, which all nodes call when instantiated."""
    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        self.train = True
        self.as_default()
    def as_default(self):
        global _g
        _g = self

    def make_graph(self):
        """Allows us to visualize the computation graph directly in a Jupyter notebook.
        must have graphviz module installed. Takes as input the topological sorted ordering
        after calling the Session class"""
        # from graphviz import Digraph
        f = Digraph()
        f.attr(rankdir='LR', size='10,8')
        names = {}
        f.attr('node', shape='circle')
        i = 1
        for nd in self.operators:
            s = 'op'+str(i)
            f.node(s, label=nd.opname)
            names[nd] = s
            i += 1
        i = 1
        f.attr('node', shape='circle')
        for nd in self.variables:
            s = 'var'+str(i)
            f.node(s, label='', style='filled', width='0.05')
            names[nd] = s
            i += 1
        i = 1
        for nd in self.constants:
            s = 'c'+str(i)
            f.node('c'+str(i))
            names[nd] = s
            i += 1
        i = 1
        f.attr('node', shape='box')
        for nd in self.placeholders:
            s = 'pl'+str(i)
            f.node('pl'+str(i), label=nd.name,)
            names[nd] = s
            i += 1
        for nd in self.operators:
            for e in nd.inputs:
                f.edge(names[e], names[nd], label=e.name)
        return f

class Operator():
    """Inputs: name=None, defaults to Op+i, for ith operator.
    Operators contain self.inputs incoming nodes, and returns self.outputs.
    during backprop we pass gradients through operators using self.gradient"""
    def __init__(self, name):
        _g.operators.add(self)
        self.inputs = []
        self.output = None
        self.gradient = None
        self.name = name
    def __repr__(self):
        return self.name

class Constant():
    """Inputs: value, name=None. Constants contain a decorator function that prevents
    their value from being reassigned"""
    def __init__(self, value, name=None):
        _g.constants.add(self)
        # self.inputs = None
        self.output = value
        self.gradient = None
        self.name = name
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
        _g.variables.add(self)
        # self.inputs = None
        self.output = value
        self.gradient = None
        self.name = ('Var'+str(Variable.count) if name==None else name)
        Variable.count+=1
    def __repr__(self):
        return self.name

class Placeholder():
    """Input: name, ptype=float. A placeholder for input types"""
    def __init__(self, name, ptype=float):
        _g.placeholders.add(self)
        # self.inputs = None
        self.output = None
        self.gradient = None
        self.name = name
    def __repr__(self):
        return self.name



class Session():
    """First runs a topological sort placing operators in order for forward/backward prop."""
    def __init__(self, graph):
        self.graph = graph
        self.loss_history = []
        self.order = self._topological_sort()
        self.order[-1].gradient = 1

    def _topological_sort(self, head_node=None):
        """Performs topological sort on all nodes prior to and including head_node. Returns an array
        of pointers to operators and placeholders, which may be evaluated sequentially from left to right
        in order to evaluate head_node."""
        vis = set()
        ordering = []
        def _dfs(node):
            if node not in vis:
                vis.add(node)
                if isinstance(node, Operator):
                    for input_node in node.inputs:
                        _dfs(input_node)
                ordering.append(node)
        if head_node is None:
            for node in self.graph.operators:
                _dfs(node)
        else:
            _dfs(head_node)
        return ordering

    def forward(self, head_node=None, feed_dict={}):
        """inputs: head_node (desired node to find), feed_dict (dictionary containing
        values for all placeholders) feed_dict must have strings matching Placeholder.name"""
        order = (self.order if head_node is None else self._topological_sort(head_node))
        for node in order:
            if isinstance(node, Placeholder):
                node.output = feed_dict[node.name]
            elif isinstance(node, Operator):
                node.output = node.forward(*[prev_node.output for prev_node in node.inputs])
        return order[-1].output

    def backward(self, target_node=None):
        """Returns a backward pass starting from the last node in the current order, down to the target_node.
        if target_node is None, backpropagate across whole graph."""
        vis = set()
        for node in reversed(self.order):
            if isinstance(node, Operator):
                # print("head node:", node)
                inputs = node.inputs
                # print("input nodes:",inputs)
                grads = node.backward(*[x.output for x in inputs], dout=node.gradient)
                for input, grad in zip(inputs,grads):
                    if input not in vis:
                        input.gradient = grad
                    else:
                        input.gradient += grad
                    vis.add(input)

    def update(self, lr=1e-3, update_rule='SGD'):
        if update_rule == 'SGD':
            for var in self.graph.variables:
                var.output -= lr*var.gradient
                var.gradient = None

    def run(self, feed_dict={}, num_iters=1000, print_every=100, lr=1e-3):
        for i in range(num_iters):
            loss = self.forward(feed_dict=feed_dict)
            self.loss_history.append(loss)
            self.backward()
            self.update(lr=lr)
            if i % print_every == 0:
                print(f"iteration {i}/{num_iters}. Loss = {loss}")

### A useful gradient checking function for individual Operators
# Inputs are the arguments to operator.forward. must match, and likely will
# need to be a numpy array. It prints out relative error in order of arguments
# that were fed/appear in operator.forward
def grad_check(op, *args):
    from gradient_checking import num_grad
    from gradient_checking import rel_error
    Graph()
    op = op(*args, name=None)
    n = len(args)
    dout = np.random.randn(*op.forward(*args).shape)
    dz = op.backward(*args, dout)
    for j in range(n):
        func = lambda z: op.forward(*[z if i==j else args[i] for i in range(n)])
        num_dz = num_grad(func, args[j], dout)
        print(rel_error(num_dz, dz[j]))

############## Begin operators ###########
# When implementing the backward pass, all outputs MUST BE TUPLES or the sess.backward
# function won't be able to properly unpack the arguments

class affine(Operator):
    def __init__(self, x, W, b, name='affine'):
        super().__init__(name)
        self.opname = 'affine'
        self.inputs=[x, W, b]
    def forward(self, x, W, b):
        """Inputs: x (N,H), W, (H,V), b (V,)
        Outputs: hidden layer (N,H)"""
        h =  b + x.dot(W)
        return h
    def backward(self, x, W, b, dout):
        """dout (H, V), Outputs: dh (N, T, H), dW_out (H, V), db_out (V)"""
        if x.ndim == 1:
            dout = dout.reshape(1,-1)
            x = x.reshape(1,-1)
        dx, dW, db, = dout.dot(W.T), x.T.dot(dout), np.sum(dout, axis=0)
        return dx, dW, db

class add(Operator):
    """Add two inputs a+b"""
    def __init__(self,a,b,name='add'):
        super().__init__(name)
        self.inputs=[a,b]
        self.opname = 'add'
    def forward(self,A,B):
        return A+B
    def backward(self,A,B,dout):
        return dout, dout

class multiply(Operator):
    """Multiply two inputs a*b"""
    def __init__(self,a,b,name='multiply'):
        super().__init__(name)
        self.inputs=[a,b]
        self.opname = 'multiply'
    def forward(self, A, B):
        return A*B
    def backward(self, A, B, dout):
        return dout*B, dout*A

class matmul(Operator):
    """Matrix multiplication A.dot(B) of matrices A,B"""
    def __init__(self,Mat,Vec,name='matmul'):
        super().__init__(name)
        self.inputs=[Mat,Vec]
        self.opname = 'matmul'
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
        self.opname = 'relu'
    def forward(self, A):
        """In: A. Out: relu(A)"""
        return np.maximum(0,A)
    def backward(self, A, dout):
        """In: A. Out: dA."""
        return (dout*np.heaviside(A,0),)

class sigmoid(Operator):
    def __init__(self,a,name='sigmoid'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[a]
        self.opname = 'sigmoid'
    def forward(self, A):
        return np.where(A >= 0, 1 / (1 + np.exp(-A)), np.exp(A) / (1 + np.exp(A)))
    def backward(self, A, dout):
        return (dout*self.forward(A)*(1-self.forward(A)),)

class tanh(Operator):
    def __init__(self, a, name='tanh'):
        """Tanh function"""
        super().__init__(name)
        self.inputs=[a]
        self.opname = 'tanh'
    def forward(self, A):
        return np.tanh(A)
    def backward(self, A, dout):
        return (dout*(1-np.tanh(A)**2),)

class softmax_loss(Operator):
    """Inputs: x (N,D), y (N). x is score matrix, y is list of N integers
    with values y[i] in [0,D) of correct output.
    Outputs: loss, dx"""
    def __init__(self, x, y, name='softmax_loss'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[x,y]
        self.opname = 'softmax_loss'
    def forward(self, x, y):
        N = len(y)
        probs = np.exp(x-np.max(x, axis=1, keepdims=True))
        norms = np.sum(probs, axis=1, keepdims=True)
        probs = probs/norms
        loss = -np.sum(np.log(probs[np.arange(N),y]))/N
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

class conv2D(Operator):
    """Inputs A (N,C,H,W), w (F,C,H_w,W_w), stride, pad. Outputs: (N,F,H',W').
    Convolves C channels to F Filters"""
    def __init__(self, A, w, b, stride=1, pad=1, name='conv2D'):
        """Tanh function"""
        super().__init__(name)
        self.inputs=[A, w, b]
        self.opname = 'conv2D'
        self.stride = stride
        self.pad = pad
    def forward(self, A, w, b):
        s, p = self.stride, self.pad
        N, C, H, W = A.shape
        F, C, H_w, W_w = w.shape
        m = int((H-H_w+2*p)/s)+1
        n = int((W-W_w+2*p)/s)+1
        A = np.pad(A,((0,0),(0,0),(p,p),(p,p)))
        out = np.zeros((N,F,m,n))
        for i in range(m):
            for j in range(n):
                A_red = A[:,:,s*i:s*i+H_w,s*j:s*j+W_w]
                out[:,:,i,j] = np.tensordot(A_red,w,axes=[[1,2,3],[1,2,3]])+b
        return out
    def backward(self, A, w, b, dout):
        """Inputs: dout (N,F,H',W') and cache from forward pass
        Outputs: dA (N,C,H,W) and dw (F,C,H_w,W_w)"""
        s, p = self.stride, self.pad
        N, C, H, W = A.shape
        F, C, H_w, W_w = w.shape
        N, F, m, n = dout.shape

        A = np.pad(A,((0,0),(0,0),(p,p),(p,p)))
        dw = np.zeros_like(w)
        dA = np.zeros_like(A)

        db = np.sum(dout, axis=(0,2,3))

        for a in range(H_w):
            for b in range(W_w):
                imax = min(m,int(np.ceil((H+2-a)/s)))
                jmax = min(n,int(np.ceil((W+2-b)/s)))
                dout_red = dout[:,:,:imax,:jmax]
                A_red = A[:,:,a:a+imax*s:s, b:b+jmax*s:s]
                dw[:,:,a,b] = np.tensordot(dout_red, A_red, axes=[[0,2,3],[0,2,3]])

        for a in range(A.shape[2]):
            for b in range(A.shape[3]):
                imin = max(0,int(np.floor((a-H_w)/s))+1)
                imax = min(m, int(a/s)+1)
                jmin = max(0,int(np.floor((b-W_w)/s))+1)
                jmax = min(n, int(b/s)+1)

                w_mat = w[:,:,np.arange(a-imin*s,a-imax*s,-s),:][:,:,:,np.arange(b-jmin*s,b-jmax*s,-s)]
                dout_mat = dout[:,:,imin:imax,jmin:jmax]
                dA[:,:,a,b] = np.tensordot(dout_mat, w_mat, axes=[[1,2,3],[0,2,3]])
        dA = dA[:,:,p:-p,p:-p]

        return dA, dw, db

class maxpool2D(Operator):
    def __init__(self, A, height=2, stride=2, width=2, name='max_pool2D'):
        """Tanh function"""
        super().__init__(name)
        self.inputs=[A]
        self.opname = 'max_pool2D'
        self.height = height
        self.stride = stride
        self.width = width
        self.mask = None
    def forward(self, A):
        """Inputs A (N,C,H,W), pool_params (dict containing pool_height, pool_width, stride) to reduced H//s, and W//s."""
        Hp, Wp, s = self.height, self.width, self.stride
        N, C, H, W = A.shape
        m = int((H-Hp)/s)+1
        n = int((W-Wp)/s)+1
        out = np.zeros((N,C,m,n))
        self.mask = np.zeros_like(A)
        for i in range(m):
            for j in range(n):
                A_window = A[:,:,i*s:i*s+Hp,j*s:j*s+Wp]
                maxval = np.max(A_window, axis=(2,3), keepdims=True)
                out[:,:,i,j] = maxval.reshape(N,C)
                self.mask[:,:,i*s:i*s+Hp,j*s:j*s+Wp] = np.where(A_window==maxval,1,0)
        return out
    def backward(self, A, dout):
        dA, Hp, Wp, s = self.mask, self.height, self.width, self.stride
        N, C, m, n = dout.shape
        for i in range(m):
            for j in range(n):
                dA[:,:,s*i:s*i+Hp,s*j:s*j+Wp] *= dout[:,:,i,j].reshape(N,C,1,1)
        return (dA,)

class batchnorm(Operator):
    def __init__(self, A, gamma, beta, name='batchnorm'):
        """batch normalization. looks at the truth value _g.train"""
        super().__init__(name)
        self.inputs=[A, gamma, beta]
        self.opname = 'batchnorm'
        self.running_mean = 0
        self.running_var = 0
        self.xhat = 0
        self.sigma = 0
    def forward(self, x, gamma, beta):
        if _g.train:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            self.sigma = np.sqrt(1e-5+var)
            self.xhat = (x-mu)/self.sigma
            y = gamma*self.xhat + beta
            self.running_mean = 0.95*self.running_mean + 0.05*mu
            self.running_var = 0.95*self.running_var + 0.05*var
            return y
        mu = self.running_mean
        var = self.running_var
        sigma = np.sqrt(1e-5+var)
        xhat = (x-mu)/sigma
        y = gamma*xhat + beta
        return y
    def backward(self, x, gamma, beta, dout):
        N, xhat, sigma = dout.shape[0], self.xhat, self.sigma
        dbeta = np.sum(dout, axis=0, keepdims=True)
        dgamma = np.sum(dout*xhat, axis=0, keepdims=True)
        dx = (gamma/sigma)*(dout - dbeta/N - xhat*dgamma/N)
        return dx, np.squeeze(dgamma), np.squeeze(dbeta)

class dropout(Operator):
    def __init__(self, A, p, name='batchnorm'):
        """dropout regularization. looks at the truth value _g.train"""
        super().__init__(name)
        self.inputs=[A]
        self.opname = 'batchnorm'
        self.p = p
        self.mask = 0
    def forward(self, A):
        if _g.train:
            p = self.p
            self.mask = np.random.choice(2,size=x.shape,p=[1-p,p])/p
            return self.mask*A
        return A
    def backward(self, A, dout):
        if _g.train:
            return self.mask*dout
        return (dout,)

class flatten(Operator):
    def __init__(self, A, name='flatten'):
        """flatten array into (N,D) shape"""
        super().__init__(name)
        self.inputs=[A]
        self.opname = 'flatten'
    def forward(self, A):
        N = A.shape[0]
        return A.reshape(N,-1)
    def backward(self, A, dout):
        N, C, H, W = A.shape
        x = dout.reshape(N,C,H,W)
        return (x,)

class svm_loss(Operator):
    """Inputs: x (N,D), y (N). x is score matrix, y is list of N integers
    with values y[i] in [0,D) of correct output.
    Outputs: loss, dx"""
    def __init__(self, x, y, name='softmax_loss'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[x,y]
        self.opname = 'softmax_loss'
        self.margins = None
    def forward(self, x, y):
        N = x.shape[0]
        correct_class_scores = x[np.arange(N), y]
        margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
        margins[np.arange(N), y] = 0
        self.margins = margins
        loss = np.sum(margins)/N
        return loss
    def backward(self, x, y, dout=1):
        N, margins = x.shape[0], self.margins
        num_pos = np.sum(margins > 0, axis=1)
        dx = np.zeros_like(x)
        dx[margins > 0] = 1
        dx[np.arange(N), y] -= num_pos
        dx /= N
        return dout*dx, None

class rnn_step(Operator):
    def __init__(self, x, Wx, h, Wh, b, name='affine'):
        super().__init__(name)
        self.opname = 'rnn_step'
        self.inputs=[x, Wx, h, Wh, b]
    def forward(self, x, Wx, h, Wh, b):
        """Inputs: x (N,H), W, (H,V), b (V,)
        Outputs: hidden layer (N,H)"""
        out = np.tanh(b + x.dot(Wx) + h.dot(Wh))
        return out
    def backward(self, x, Wx, h, Wh, b, dout):
        """dout (H, V), Outputs: dh (N, T, H), dW_out (H, V), db_out (V)"""
        ctan = (1-self.output**2)
        dout = ctan*dout
        if x.ndim == 1:
            x = x.reshape(1,-1)
            h = h.reshape(1,-1)
            dout = dout.reshape(1,-1)
        dx = dout.dot(Wx.T)
        dh = dout.dot(Wh.T)
        dWx = x.T.dot(dout)
        dWh = h.T.dot(dout)
        db = np.sum(dout, axis=0)
        return dx, dWx, dh, dWh, db

class softmax1D(Operator):
    """Inputs: x (N,D), y (N). x is score matrix, y is list of N integers
    with values y[i] in [0,D) of correct output.
    Outputs: loss, dx"""
    def __init__(self, x, y, name='softmax1D'):
        """Numerically stable implementation of sigmoid"""
        super().__init__(name)
        self.inputs=[x,y]
        self.opname = 'softmax1D'
    def forward(self, x, y):
        prob = x-x[y]
        loss = np.log(np.sum(np.exp(x)))
        return loss
    def backward(self, x, y, dout=1):
        dx = np.exp(x-np.max(x))
        dx = dx/np.sum(x)
        dx[y] -=1
        return dout*dx, None


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
