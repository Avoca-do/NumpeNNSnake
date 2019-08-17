import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))

def tanh_prime(x):
    return 1 - np.tanh(x)**2

n_hidden = 105
n_in = 10
n_out = 10
n_sample = 300

learning_rate = 0.01
momentum = 0.9

np.random.seed(5)

def train(x, t, V, W, Bv, Bw):
    A = np.dot(x, V) + Bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + Bw
    Y = sigmoid(B)

    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

    return loss, (dV, dW, Ev, Ew)


def predict(x, V, W, bV, bW):
    A = np.dot(x, V) + bV
    B = np.dot(np.tanh(A), W) + bW
    return (sigmoid(B) > 0.5).astype(int)

V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

for epoch in range(150):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()

    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)
        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append(loss)

    print('Epoch: %d, Loss: %.8f, Time: %.4f' % (epoch, np.mean(err), time.clock() - t0))

for i in range(10):
    x = np.random.binomial(1, 0.5, n_in)
    print('XOR  prediction')
    print(x)
    print(predict(x, *params))
