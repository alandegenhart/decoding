"""Doomsday fuel.


"""


def solution(m):
    import numpy as np
    import fractions

    if len(m) == 1:
        return [1, 1]

    n_states = len(m)
    mask = [False if mi == [0] * n_states else True for mi in m]
    idx = np.concatenate([np.arange(n_states)[mask], np.arange(n_states)[np.logical_not(mask)]])
    M = np.array(m)
    M = M[idx, :]
    M = M[:, idx]

    # Convert to probabilities
    M = [np.array(mi)/np.sum(mi).astype(float) if np.sum(mi) > 0 else mi for mi in M]

    # The two test cases are both already sorted, so to test we can just use
    # them as-is.  Eventually we'll have to sort, though.
    M = np.array(M)
    n_transient = sum(mask)  # Will need to figure this out
    Q = M[0:n_transient, 0:n_transient]
    R = M[0:n_transient, n_transient:]
    N = np.linalg.inv(np.eye(n_transient) - Q)
    B = np.matmul(N, R)

    # Convert the solution into a fraction
    frac = [fractions.Fraction(si).limit_denominator(1000) for si in B[0, :]]
    numerator = [f.numerator for f in frac]
    denominator = [f.denominator for f in frac]
    d = int(np.lcm.reduce(denominator))
    numerator = [int(ni * d / di) for ni, di in zip(numerator, denominator)]

    return numerator + [d]


if __name__ == '__main__':
    m = [
        [0, 1, 0, 0, 0, 1],  # s0, the initial state, goes to s1 and s5 with equal probability
        [0, 0, 0, 0, 0, 0],  # s2 is terminal, and unreachable (never observed in practice)
        [0, 0, 0, 0, 0, 0],  # s3 is terminal
        [0, 0, 0, 0, 0, 0],  # s4 is terminal
        [0, 0, 0, 0, 0, 0],  # s5 is terminal
        [4, 0, 0, 3, 2, 0],  # s1 can become s0, s3, or s4, but with different probabilities
    ]
    expected_result = [9, 0, 3, 2, 14]
    result = solution(m)
    print('\n')
    print(expected_result)
    print(result)
    # print(f'Result: {result}, Expected: {expected_result}')

    m = [
        [0, 1, 0, 0, 0, 1],  # s0, the initial state, goes to s1 and s5 with equal probability
        [4, 0, 0, 3, 2, 0],  # s1 can become s0, s3, or s4, but with different probabilities
        [0, 0, 0, 0, 0, 0],  # s2 is terminal, and unreachable (never observed in practice)
        [0, 0, 0, 0, 0, 0],  # s3 is terminal
        [0, 0, 0, 0, 0, 0],  # s4 is terminal
        [0, 0, 0, 0, 0, 0],  # s5 is terminal
    ]
    expected_result = [0, 3, 2, 9, 14]
    result = solution(m)
    print('\n')
    print(expected_result)
    print(result)

    m = [
        [0, 2, 1, 0, 0],
        [0, 0, 0, 3, 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    expected_result = [7, 6, 8, 21]
    result = solution(m)
    print('\n')
    print(expected_result)
    print(result)

    m = [
        [1]
    ]
    expected_result = [1, 1]
    result = solution(m)
    print('\n')
    print(expected_result)
    print(result)