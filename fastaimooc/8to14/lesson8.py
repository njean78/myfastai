import pickle
import gzip

from fastai import datasets
from torch import tensor
from torch.nn import init
import torch


def load_data():
    MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"
    path = datasets.download_data(MNIST_URL, ext=".gz")

    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(
            f, encoding="latin-1"
        )
    ## tensorify
    (x_train, y_train, x_valid, y_valid) = map(
        tensor, (x_train, y_train, x_valid, y_valid)
    )

    n, c = x_train.shape

    ##
    img = x_train[0]
    img.view(28, 28).type()
    # plt.imshow(img.view(28,28))
    return (x_train, y_train, x_valid, y_valid)


def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c


def matmul2(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i, j] = (a[i, :] * b[:, j]).sum()
    return c


# c.unsqueeze(0) ~ c[None, :] - c[None, ...]
# c.unsqueeze(1) ~ c[:, None]
def matmul3_broadcasting(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i, :] = (a[i, None] * b).sum(dim=0)
    return c


def matmul4_ein(a, b):
    return torch.einsum("ik,kj->ij", a, b)

def matmul5(a, b):
    return a.matmul(b) # a@b

######################
def normalize(x,m,s):
    return (x-m)/s

def lin(x,w,b):
    return x@w +b

def relu(x):
    return x.clamp_min(0.)

def model(xb, w1, b1, w2, b2):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3

def mse(output, target):
    return (output.squeeze(1) - target).pow(2).mean()

def mse_grad(inp, target):
    inp.g = 2. * (inp.squeeze(1) - target).unsqueeze(-1)/inp.shape[0]

def relu_grad(inp, target):
    inp.g = (inp>0).float() * target.g

def lin_grad(inp, target, w,b):
    inp.g = target.g @ w.t()
    w.g = (inp.unsqueeze(-1) * target.unsqueeze(1)).sum(0)
    b.g = target.g.sum(0)

## forward_and_backward(x_train, y_train, ...)
def forward_and_backward(inp, target, w1, b1, w2, b2):
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    loss = mse(out, target)

    mse_grad(out, target)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)


def linear_model(data, nh = 50):
    (x_train, y_train, x_valid, y_valid) = data
    # weights = torch.randn(784, 10)
    # bias = torch.zeros(10)

    train_mean, train_std = x_train.mean(), x_train.std()

    x_train = normalize(x_train, train_mean, train_std)
    x_valid = normalize(x_valid, train_mean, train_std)

    nrows, ncols = x_train.shape
    nclasses = y_train.max() +1
    
    ## kaiming initialization
    #w1 = torch.randn(ncols, nh) / math.sqrt(2/ncols)
    w1 = torch.zeros(ncols, nh)
    w1 = init(w1, mode = 'fan_out')

    b1 = torch.zeros(nh)

    w2 = torch.randn(nh,1) / math.sqrt(nh)
    b2 = torch.zeros(1)

    # forward
    output = model(x_valid, w1, b1, w2, b2)
    res = mse(output, y_valid.float())

    

    y = bias + matmul(weights, data)


## class Relu -> __call__ and backward methods 

if __name__ == "__main__":
    load_data()
