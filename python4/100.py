import numpy as np
import matplotlib.pyplot as plt

import w4_unittest

A = np.array([[2, 3],[2, 1]])
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])

A_eig = np.linalg.eig(A)

print("Matrix A:\n", A, "\n\n Eigenvalues and eigenvectors of matrix A:\n", A_eig)


A_rotation = np.array([[0, 1],[-1, 0]])
A_rotation_eig = np.linalg.eig(A_rotation)

print("Matrix A_rotation:\n", A_rotation,"\n\n Eigenvalues and eigenvectors of matrix A_rotation:\n", A_rotation_eig)

A_identity = np.array([[1, 0],[0, 1]])
A_identity_eig = np.linalg.eig(A_identity)

print("Matrix A_identity:\n", A_identity, "\n\n Eigenvalues and eigenvectors of matrix A_identity:\n", A_identity_eig)

A_scaling = np.array([[2, 0],[0, 2]])
A_scaling_eig = np.linalg.eig(A_scaling)

print("Matrix A_scaling:\n", A_scaling, "\n\n Eigenvalues and eigenvectors of matrix A_scaling:\n", A_scaling_eig)


A_projection = np.array([[1, 0],[0, 0]])
A_projection_eig = np.linalg.eig(A_projection)

print("Matrix A_projection:\n", A_projection, "\n\n Eigenvalues and eigenvectors of matrix A_projection:\n", A_projection_eig)



P = np.array([
    [0, 0.75, 0.35, 0.25, 0.85],
    [0.15, 0, 0.35, 0.25, 0.05],
    [0.15, 0.15, 0, 0.25, 0.05],
    [0.15, 0.05, 0.05, 0, 0.05],
    [0.55, 0.05, 0.25, 0.25, 0]
])
X0 = np.array([[0], [0], [0], [1], [0]])

X1 = np.matmul(P, X0)


print(sum(P))



X = np.array([[0], [0], [0], [1], [0]])
m = 20

for t in range(m):
    X = P @ X

print(X)

np.linalg.eig(P)

X_inf = np.linalg.eig(P)[1][:,0]

print("Eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))



def check_eigenvector(P, X_inf):
    X_check = np.matmul(P, X_inf)

    return X_check


X_check = check_eigenvector(P, X_inf)

print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))
