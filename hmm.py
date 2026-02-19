import numpy as np


def normalize_rows(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)


def forward(obs_seq, A, B, pi):
    N = len(pi)
    T = len(obs_seq)

    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = pi * B[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]

    likelihood = np.sum(alpha[T-1])
    return alpha, likelihood


def backward(obs_seq, A, B):
    N = A.shape[0]
    T = len(obs_seq)

    beta = np.zeros((T, N))
    beta[T-1] = 1

    for t in reversed(range(T-1)):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs_seq[t+1]] * beta[t+1])

    return beta


def baum_welch(obs_seq, N, M, iterations=20):

    # Random initialization
    pi = np.random.dirichlet(np.ones(N))
    A = normalize_rows(np.random.rand(N, N))
    B = normalize_rows(np.random.rand(N, M))

    likelihoods = []

    for _ in range(iterations):

        alpha, likelihood = forward(obs_seq, A, B, pi)
        beta = backward(obs_seq, A, B)

        T = len(obs_seq)

        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))

        for t in range(T):
            denom = np.sum(alpha[t] * beta[t])
            gamma[t] = (alpha[t] * beta[t]) / denom

        for t in range(T-1):
            denom = np.sum(
                alpha[t][:, None] *
                A *
                B[:, obs_seq[t+1]] *
                beta[t+1]
            )
            for i in range(N):
                xi[t, i] = (
                    alpha[t, i] *
                    A[i] *
                    B[:, obs_seq[t+1]] *
                    beta[t+1]
                ) / denom

        # Re-estimation
        pi = gamma[0]

        for i in range(N):
            A[i] = np.sum(xi[:, i, :], axis=0) / np.sum(gamma[:-1, i])

        for i in range(N):
            for k in range(M):
                mask = (obs_seq == k)
                B[i, k] = np.sum(gamma[mask, i]) / np.sum(gamma[:, i])

        likelihoods.append(likelihood)

    return pi, A, B, likelihoods
