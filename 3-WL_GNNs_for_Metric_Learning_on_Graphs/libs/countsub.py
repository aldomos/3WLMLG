import libs.utils_matlang as mat
import numpy as np
import numpy.linalg as lin


def tailtriangle(A):
    return (mat.one(A).T@(lin.matrix_power(A,3)*mat.diag(A@mat.one(A)-2))@mat.one(A))[0,0]/2

def fivecycle(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    A5=A4.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    r = (A5-2*A3*A2)*I-2*(np.diag((((a*(A2))@a)*J).sum(1)) - A3/2*I)
    r -= 2*(np.diag(a@((A3/2*I).sum(1)))-A3*I)
    return 1/10*r.sum()

def fivecyclenode(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    A5=A4.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    r = (A5-2*A3*A2)*I-2*(np.diag((((a*(A2))@a)*J).sum(1)) - A3/2*I)
    r -= 2*(np.diag(a@((A3/2*I).sum(1)))-A3*I)
    return r.sum(1)/2

def square(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    return 1/8*(A4*I-A2*J-A2*A2*I).sum()

def trisquare(A):
    a=A
    A2=a.dot(a)
    one = mat.one(a)
    return 1/4*(A2*A*(A2*A-(A2*A>0))).sum()


def tailsquare(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    return 1/2*((A4*I-np.diag((A2*J).sum(1))-A2*A2*I)*(mat.diag(a.dot(one)-2))- np.diag((((a*(A2))*((a*A2)%2 ==0)).sum(1)))).sum()

def sixcycle(A):
    a=A
    A2=a.dot(a)
    A3=A2.dot(a)
    A4=A3.dot(a)
    A5=A4.dot(a)
    A6=A5.dot(a)
    one = mat.one(a)
    I = mat.diag(one)
    J = one@one.T - I
    C = (A4*I-np.diag((A2*J).sum(1))-A2*A2*I)
    C1 = A*A3 - A@(A2*I)-(A2*I-I)@A
    r =  A6*I - A2*A2*A2*I - 2*mat.diag(((A2*J)@one)*(A@one)) 
    r -= mat.diag((J*A3 - A@(A2*I)-(A2*I-I)@A)@one)
    r -= mat.diag(((A@(A2*I-I))*(A@(A2*I-I)))@one)
    r -= A3*A3*I + 2*mat.diag((A2*A)@(A3*I-2*I)@one)
    # r += (A3*I)*(A3 -2*(A3>0))/2
    r -= 2*(A2*I)@C +  2*mat.diag(((C1)@(A2*I-I))@one)  
    r -=  mat.diag((((A2*J)*(A2-(A2>0)))@(A2*I-2*I))@one)
    r -= (A@C@A)*I - (A@C1 + C1@A)*I 
    r += 9*mat.diag(((A2*A)*(A2-(A2>0)))@one)
    # r += 3*mat.diag(((A@(A2*A)*(A2-(A2>0))/2)*J-(A*A2)*(A2-(A2>0))/2)@one)
    return r.sum()/12

