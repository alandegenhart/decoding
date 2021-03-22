"""Bunny worker solutions

"""

import timeit


def solution(x, y, mode=1):
    # This is essentially a Fibonacci sequence problem

    if mode == 1:
        # Calculate starting number
        c_0 = 1
        for i in range(1, y+1):
            c_0 += i - 1

        # Calculate ID
        bunny_id = c_0
        for i in range(2, x+1):
            bunny_id += y + i - 1

    if mode == 2:
        # We can also use geometry to solve this
        d = x + y
        bunny_id = (0.5) * d**2 - 1.5 * d + x + 1
        bunny_id = int(bunny_id)

    return str(bunny_id)


if __name__ == '__main__':
    xy = [1, 1]
    o_expected = 1
    o = solution(*xy)
    print(f'Input: {xy} -> Output: {o}, Expected: {o_expected}')

    xy = [3, 2]
    o_expected = 9
    o = solution(*xy)
    print(f'Input: {xy} -> Output: {o}, Expected: {o_expected}')

    xy = [2, 3]
    o_expected = 8
    o = solution(*xy)
    print(f'Input: {xy} -> Output: {o}, Expected: {o_expected}')

    xy = [5, 10]
    o_expected = 96
    o = solution(*xy)
    print(f'Input: {xy} -> Output: {o}, Expected: {o_expected}')

    # Time
    t_1 = timeit.timeit(solution(100000, 100000, mode=1), number=10000000)
    print(f'Using loops: {t_1}')
    t_2 = timeit.timeit(solution(100000, 100000, mode=2), number=10000000)
    print(f'Using geometry: {t_2}')

    # Note -- there doesn't seem to be much of a difference between these two.
    # In one case we have two O(N)(?) operations, where N = {x, y}.
    #
    # What is the difference between addition and multiplication in big-O notation?
