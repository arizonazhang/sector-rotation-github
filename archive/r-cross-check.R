# maximum sharpe ratio
library(quadprog)
path = "C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation"
Vmat = read.csv("C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation\\vmat.csv")
Evec = read.csv("C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation\\evec.csv")
Vmat = as.matrix(Vmat)
dvec = rep(0, 10)
Evec = as.matrix(Evec)
Amat = cbind(rep(1, 10), Evec[1:10], diag(10))
wvec = c(1, 0.02, rep(0, 10))
gamma = 4
res = solve.QP(Vmat*2, dvec, Amat, wvec, meq = 2)
res$solution
sum(res$solution)

# maximum utility
library(quadprog)
path = "C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation"
Vmat = read.csv("C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation\\vmat.csv")
Evec = read.csv("C:\\Users\\Qianhua Zh\\Desktop\\AIFT\\sector-rotation\\evec.csv")
Vmat = as.matrix(Vmat)
Evec = as.matrix(Evec)
Amat = cbind(rep(1, 10), diag(10))
wvec = c(1, rep(0, 10))
gamma = 2
res = solve.QP(Vmat*gamma, Evec, Amat, wvec, meq = 1)
w = res$solution
sum(res$solution)
gamma*t(w)%*%Vmat%*%w - sum(Evec*w)

w2 = c(0, 0, 0, 0, 0.43507, 0.42411, 0.14082, 0, 0, 0)
gamma*t(w2)%*%Vmat%*%w2 - sum(Evec*w2)
