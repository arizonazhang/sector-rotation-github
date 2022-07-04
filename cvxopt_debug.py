import pandas as pd
import numpy as np
from cvxopt import matrix, solvers  # quadratic programming


def example1():
    Vmat = pd.read_csv("Vmat.csv", index_col=0)
    Vmat = Vmat.values
    Evec = pd.read_csv("Evec.csv", index_col=0)
    Evec = Evec.iloc[:, 0].values
    Vmat = Vmat.astype('float')  # must convert otherwise the package gives a warning
    Evec = Evec.astype('float')

    # optimization
    solvers.options['show_progress'] = False  # do not show optimization output
    solvers.options['abstol'] = 0.1 ** -6
    solvers.options['reltol'] = 0.1 ** -5
    Q = 2 * matrix([list(Vmat[i, :]) for i in range(11)])
    q = -1 * matrix(list(Evec))
    G = matrix([[0.0] * i + [-1.0] + [0.0] * (11 - i - 1) for i in range(11)])
    h = matrix([0.0] * 11)
    A = matrix([1.0] * 11, (1, 11))
    b = matrix(1.0)
    sol = solvers.qp(Q, q, G, h, A, b)
    print(sol['x'])

def example2():
    Q = 2 * matrix([[2, .5], [.5, 1]])
    p = matrix([1.0, 1.0])
    G = matrix([[-1.0, 0.0], [0.0, -1.0]])
    h = matrix([0.0, 0.0])
    A = matrix([1.0, 1.0], (1, 2))
    b = matrix(1.0)
    sol = solvers.qp(Q, p, G, h, A, b)
    print(sol['x'])

example1()
# [0.12541357802219638, 0.12048440823518934, 0.0503191387389004, 0.1368823073451066, 0.08854003515011662,
# 0.08832834240954336, 0.0638219851967258, 0.08191277455099356, 0.0661470514535505, 0.0624822433526731, 0.11566813554500445]

example2()
# 0.25, 0.75