from __future__ import division

from math import fsum
from numpy import (
    arccosh,
    arcsinh,
    arctanh,
    cos,
    cosh,
    exp,
    log,
    pi,
    sin,
    sinh,
    sqrt,
    sum,
    tan,
    tanh,
)
from scipy.misc import logsumexp


def h(j, beta):
    return beta * j


def h_star(j, beta):
    return arctanh(exp(-2 * beta * j))


def gamma(n, j, beta, r):
    return arccosh(
        cosh(2 * h_star(j, beta)) * cosh(2 * h(j, beta)) -
        sinh(2 * h_star(j, beta)) * sinh(2 * h(j, beta)) * cos(r * pi / n))


def log_z(n, j, beta):
    return (
        -log(2) + 1 / 2 * n**2 * log(2 * sinh(2 * h(j, beta))) + logsumexp([
            fsum([
                log(2 * cosh(n / 2 * gamma(n, j, beta, 2 * r)))
                for r in range(n)
            ]),
            fsum([
                log(2 * sinh(n / 2 * gamma(n, j, beta, 2 * r)))
                for r in range(n)
            ]),
            fsum([
                log(2 * cosh(n / 2 * gamma(n, j, beta, 2 * r + 1)))
                for r in range(n)
            ]),
            fsum([
                log(2 * sinh(n / 2 * gamma(n, j, beta, 2 * r + 1)))
                for r in range(n)
            ]),
        ]))


def free_energy(n, j, beta):
    return -1 / n**2 / beta * log_z(n, j, beta)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',type=int,default=4,help='L')
    parser.add_argument('-T',type=float,default=2.269185314213022,help='T')
    #parser.add_argument('-beta',type=float,default=1.0,help='beta')
    args = parser.parse_args()
    beta = 1/args.T
    print ('#n:', args.n, 'beta:', beta)
    print(log_z(n=args.n, j=1, beta=beta))
    print(beta, free_energy(n=args.n, j=1, beta=beta))
