from gradient_checking import *
import unittest
import numpy as np
from layers import *

import sys
sys.stdout = open("test_grad_output.txt", "w")


############### Check affine backward ##############
if True:
    N, T, H, D = 4, 5, 6, 7
    print('-'*40)
    print("Test affine_backward")
    x = np.random.randn(N,D)
    W = np.random.randn(D,H)
    b = np.random.randn(H)
    dout = np.random.randn(N,H)

    h, cache = affine_forward(x, W, b)
    dx, dW, db = affine_backward(dout, cache)

    funcx = lambda x: affine_forward(x, W, b)[0]
    funcW = lambda W: affine_forward(x, W, b)[0]
    funcb = lambda b: affine_forward(x, W, b)[0]

    num_dx = num_grad(funcx, x, dout)
    num_dW = num_grad(funcW, W, dout)
    num_db = num_grad(funcb, b, dout)

    print(f"Testing x=({N,D}), W=({D,H}), b=({H})")
    print("dx error:", rel_error(num_dx, dx))
    print("dW error:", rel_error(num_dW, dW))
    print("db error:", rel_error(num_db, db))

############### Check convolution2D layer forward/backwards #############
if False:
    # Check the gradients numerically
    # Make random A, w, as well as a random dout matrix
    # pad=1 stride=1 by default
    print('-'*40)
    print('Testing Convlution layer backwards')
    A = np.random.randn(7,7)
    w = np.random.randn(3,3)

    for stride, pad in [(1,1),(2,1),(1,2),(2,2)]:
        m = int((A.shape[0]-w.shape[0]+2*pad)/stride)+1
        dout = np.random.randn(m,m)

        # Analytically compute derivatives
        out, cache = conv_forward(A, w, stride=stride, pad=pad)
        dA, dw = conv_backward(dout, cache)

        # the forward function must be a scalar, by definition we pick dout_ij * (A*w)_ij
        # this obeys the chain rule that dJ/dw_mn = dout_ij*d(A*w)_ij/dw_mn, which is what we want
        funcw = lambda w: conv_forward(A,w,stride,pad)[0]
        funcA = lambda A: conv_forward(A,w,stride,pad)[0]

        num_dw = num_grad(funcw, w, dout)
        num_dA = num_grad(funcA, A, dout)

        print(f"Testing w={w.shape}, A={A.shape}, stride={stride}, pad={pad}")
        print("dw error:", rel_error(num_dw, dw))
        print("dA error:", rel_error(num_dA, dA))

############### Check sigmoid and relu ##############
if True:
    N, T, H, D = 4, 5, 6, 7
    print('-'*40)
    print("Test sigmoid")
    x = np.random.randn(N,T,D)
    dx1, dx2 = sigmoid(x)*(1-sigmoid(x)), np.heaviside(x,0)
    num_sig, num_rel = num_grad(sigmoid,x,1), num_grad(relu,x,1)
    print("dx error (sigmoid):", rel_error(num_sig,dx1))
    print("dx error (relu):", rel_error(num_sig,dx1))

############### Check softmax loss  ################
if True:
    print('-'*40)
    print("Test softmax_loss.")
    N, D, H = 7,6,8
    y = np.random.randint(D,size=N)
    x = np.random.randn(N,D)

    loss, dx = softmax_loss(x,y)

    func = lambda x: softmax_loss(x,y)[0]
    num_dx = num_grad(func, x, 1)

    print(f"Testing x={N,D}, y = {N,}")
    print(f"Loss should be ~{np.log(N)}, computed value:{loss}")
    print(f"dx error: {rel_error(num_dx, dx)}")

############## Check Max pool backwards2D ##################
if False:
    print('-'*40)
    print('Testing Max pool layer backwards')
    N = 10
    A = np.random.randn(N,N)

    for w,h,s in [(2,2,2),(3,3,3)]:
        m = int((N-w)/s)+1
        n = int((N-h)/s)+1
        dout = np.random.randn(m,n)
        pool_params = {'pool_width':w,'pool_height':h,'stride':s}
        out, cache = max_pool_forward(A, pool_params)
        dA = max_pool_backward(dout, cache)

        func = lambda A: max_pool_forward(A, pool_params)[0]
        num_dA = num_grad(func, A, dout)
        print(f'pool width={w}, pool height={h}, pool stride ={s}')
        print("dA error:", rel_error(dA, num_dA))

############ Check Batchnorm forward/backward ##################
if True:
        print('-'*40)
        print("Test batchnorm_forward.")
        N, D = 7, 3
        x = np.random.randn(N,D)
        dy = np.random.randn(N,D)
        gamma, beta = 3, 1

        print("Before normalization,")
        print(f"x mean:{np.mean(x, axis=0)}")
        print(f"x std:{np.std(x, axis=0)}")
        y, cache = batchnorm_forward(x, gamma, beta)
        print(f"Using gamma={gamma} and beta={beta}. After we find:")
        print(f"y mean:{np.mean(y, axis=0)}")
        print(f"y std:{np.std(y, axis=0)}")

        print("\nTesting batchnorm_backward...")
        gamma, beta = np.random.randn(D), np.random.randn(D)
        _, cache = batchnorm_forward(x, gamma, beta)
        dx, dgamma, dbeta = batchnorm_backward(dy, cache)

        funcx = lambda x: batchnorm_forward(x, gamma, beta)[0]
        funcg = lambda gamma: batchnorm_forward(x, gamma, beta)[0]
        funcb = lambda beta: batchnorm_forward(x, gamma, beta)[0]
        num_dx = num_grad(funcx, x, dy)
        num_dgamma = num_grad(funcg, gamma, dy)
        num_dbeta = num_grad(funcb, beta, dy)

        print(f"Testing x={N,D}, gamma={gamma}, beta={beta}")
        print(f"dx error: {rel_error(num_dx, dx)}")
        print(f"dgamma error: {rel_error(num_dgamma, dgamma)}")
        print(f"dbeta error: {rel_error(num_dbeta, dbeta)}")

############ Check dropout backward ##################
if True:
        print('-'*40)
        print("Test dropout_backward.")
        N, D, p = 2, 3, 0.5
        x = np.random.randn(N,D)
        dy = np.random.randn(N,D)

        mask = np.random.choice(2,size=x.shape,p=[1-p,p])/p
        dx = dropout_backward(dy, mask)

        func = lambda x: mask*x
        num_dx = num_grad(func, x, dy)

        print("dx error:", rel_error(num_dx, dx))

############### Check convolution layer forward/backwards #############
import time

if True:
    # pad=1 stride=1 by default
    print('-'*40)
    print('Testing Convolution layer backwards')
    N, C, H, W, F, f = 2,2,5,5,3,3
    A = np.random.randn(N,C,H,W)
    w = np.random.randn(F,C,f,f)

    for stride, pad in [(1,1)]:#,(2,1),(1,2),(2,2)]:
        m = int((H-f+2*pad)/stride)+1
        n = int((W-f+2*pad)/stride)+1
        dout = np.random.randn(N,F,m,n)

        print("Computing forward convolution...")
        # tic = time.process_time()
        out, cache = conv_forward(A, w, stride=stride, pad=pad)
        # toc = time.process_time()
        # print(f"Took {toc-tic:.5f} seconds")
        print("Computing backward pass...")
        dA, dw = conv_backward(dout, cache)
        tic = time.process_time()
        # print(f"Took {tic-toc:.5f} seconds")

        funcw = lambda w: conv_forward(A,w,stride,pad)[0]
        funcA = lambda A: conv_forward(A,w,stride,pad)[0]

        num_dw = num_grad(funcw, w, dout)
        num_dA = num_grad(funcA, A, dout)

        print(f"Testing w={w.shape}, A={A.shape}, stride={stride}, pad={pad}")
        print("dw error:", rel_error(num_dw, dw))
        print("dA error:", rel_error(num_dA, dA))

############## Check Max pool backwards ##################
if True:
    print('-'*40)
    print('Testing Max pool layer backwards')
    N, C, H, W = 2,3,6,6
    A = np.random.randn(N, C, H, W)

    for w,h,s in [(2,2,2),(3,3,3)]:
        m = int((H-w)/s)+1
        n = int((W-h)/s)+1
        dout = np.random.randn(N,C,m,n)
        pool_params = {'pool_width':w,'pool_height':h,'stride':s}
        out, cache = max_pool_forward(A, pool_params)
        dA = max_pool_backward(dout, cache)

        func = lambda A: max_pool_forward(A, pool_params)[0]
        num_dA = num_grad(func, A, dout)
        print(f'pool width={w}, pool height={h}, pool stride ={s}')
        print("dA error:", rel_error(dA, num_dA))
sys.stdout.close()
