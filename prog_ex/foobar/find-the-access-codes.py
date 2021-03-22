def solution(l):
    """
    Solution 4 again passes all but the last test case.  Try to speed things
    up some using a dynamic programming-like approach.

    This solution wound up passing all of the test cases -- the key here is to
    uses a memorization/dynamic programming approach.  A core component of this
    problem involves finding all multiples of a number after a given number in
    the list.  In the brute force approach, we do the following:

    0: for each li:
        1: for each lj such that j > i:
            2: if li divides lj:
                3: for each lk such that k > j:
                    4: if lj divides lk:
                        (li, lj, lk) is a valid solution

    Note that steps 3 and 4 involve counting the number of valid values of lk
    for a given lj.  Since we are evaluating all possible values of lj for each
    possible value of li, this means that we would potentially repeat steps 3
    and 4 multiple times for the *same value of lj*.

    Take the example:
    l = [1, 1, 1, 1, 3]
    In this case we would evaluate the number of valid lks for the final '1'
    3 times.  In the worst case, where l is of length N and consists of
    all 1's, we would be finding the valid lks for the penultimate lj (N-2)
    times.

    To improve on this, we can cache/memorize the values as we compute them.
    We'll store the smallest computation -- the number of possible values of lk
    for a given lj.  Then, as we traverse the list, if we have already
    computed the values of lk for a given lj, we just use the value that we
    previously computed.  This touches on the concept of Dynamic Programming.
    """
    # Make sure no numbers are less than 1 or greater than 999999
    for li in l:
        if li > 999999 or li < 1:
            return 0

    # Get number of elements in the list
    n_l = len(l)

    # If there are fewer than 3 elements in the list, then there
    # can't be any lucky triples, so return 0
    if n_l < 3 or n_l > 2000:
        return 0

    # Initialize counts -- d_cts[j] corresponds to the number of valid values
    # of l[k] for l[j].
    d_cts = [-1] * n_l

    ctr = 0
    # First iterate over i
    for i in range(n_l-2):
        for j in range(i+1, n_l-1):
            if l[j] % l[i] == 0:
                # Check to see if we already computed this
                if d_cts[j] == -1:
                    # Count the number of valid divisors for l[j]
                    d_ctr = 0
                    for k in range(j+1, n_l):
                        if l[k] % l[j] == 0:
                            d_ctr += 1
                        d_cts[j] = d_ctr

                # Add the pre-computed value
                ctr += d_cts[j]

    return ctr


def solution_4(l):
    # Solution 3 passes all but the last test case.  I suspect this is a timing
    # problem, so see if we can speed things up.

    # Make sure no numbers are less than 1 or greater than 999999
    for li in l:
        if li > 999999 or li < 1:
            return 0

    # Get number of elements in the list
    n_l = len(l)

    # If there are fewer than 3 elements in the list, then there
    # can't be any lucky triples, so return 0
    if n_l < 3 or n_l > 2000:
        return 0

    ctr = 0
    # First iterate over i
    for i in range(n_l-2):
        for j in range(i+1, n_l-1):
            if l[j] % l[i] == 0:
                for k in range(j+1, n_l):
                    if l[k] % l[j] == 0:
                        ctr += 1

    return ctr


def solution_3(l):
    # Solution 2 appears to work but fails all 3 hidden test cases.  The only
    # thing I can guess is that duplicates can be counted -- this seems to be
    # implied in the description but isn't explicitly said, so it might be the
    # case that we should count duplicates.  The fact that the output fits in a
    # 32-bit integer might also be a clue -- it seems unlikely that this would
    # need to be stated if we were only counting unique triples

    # Make sure no numbers are less than 1 or greater than 999999
    for li in l:
        if li > 999999 or li < 1:
            return 0

    # Get number of elements in the list
    n_l = len(l)

    # If there are fewer than 3 elements in the list, then there
    # can't be any lucky triples, so return 0
    if n_l < 3 or n_l > 2000:
        return 0

    # Create a flipped version of l -- this is easier to traverse
    l_flip = l[::-1]

    # Define maximum value
    max_val = int(2**32 / 2) - 1

    # For this case, we have to iterate through the entire list
    ctr = 0
    for k, l_k in enumerate(l_flip[:-2]):
        # Get all possible values of l_j
        l_j_all = l_flip[k+1:]
        l_j_valid = [lja for lja in set(l_j_all) if l_k % lja == 0]  # All valid unique numbers

        # Iterate over each unique value of l_j
        for l_j in l_j_valid:
            # Get the number of times l_j appears
            n_l_j = l_j_all.count(l_j)
            # Find each instance
            st = k + 1
            for nlj in range(n_l_j):
                # Get the index for the current instance of nlj
                j = l_flip.index(l_j, st)

                # Look for all valid numbers after index j
                l_i_all = l_flip[j+1:]
                l_i_valid = [l_i for l_i in l_i_all if l_j % l_i == 0]
                ctr += len(l_i_valid)

                # Perform a check on the counter -- since the output must be
                # represented by a signed int, then if the output is greater
                # than this we could run into problems
                if ctr > max_val:
                    return max_val

                # Update the starting point
                st = j + 1

    return ctr


def solution_2(l):
    # Make sure no numbers are less than 1 or greater than 999999
    for li in l:
        if li > 999999 or li < 1:
            return 0

    # Get number of elements in the list
    n_l = len(l)

    # If there are fewer than 3 elements in the list, then there
    # can't be any lucky triples, so return 0
    if n_l < 3 or n_l > 2000:
        return 0

    # Create a flipped version of l -- this is easier to traverse
    l_flip = l[::-1]

    # Get unique elements
    lucky_triples = []
    lk_unique = set(l_flip)
    for lk in lk_unique:
        # Find the first instance of lk in l_flip
        k = l_flip.index(lk)

        # If there aren't at least two elements left, continue -- this
        # means that there aren't sufficient elements left in the list
        # to create a valid tuple.
        if n_l - k < 3:
            continue

        # Get all unique elements after k and filter by those which
        # are divisible by lk
        lj_unique = set(l_flip[k+1:])
        lj_unique = [lju for lju in lj_unique if lk % lju == 0]

        # Iterate over possible ljs:
        for lj in lj_unique:
            # Find the first instance of lj in l_flip starting at k+1
            j = l_flip.index(lj, k+1)

            # Get all possible unique elements after j and filter
            li_unique = set(l_flip[j+1:])
            li_unique = [liu for liu in li_unique if lj % liu == 0]

            # Add tuples
            for liu in li_unique:
                lucky_triples.append((liu, lj, lk))

    # If no lucky triples were found, return 0
    if not lucky_triples:
        return 0

    # Remove any duplicates and return the number of elements
    lucky_triples = set(lucky_triples)

    return len(lucky_triples)


def solution_1(l):
    # NOTE -- this solution does not appear to work -- the 'verify'
    # command seems to time out.  I'm guessing that's because it
    # takes too long (execution time appears to be limited).

    # Get number of elements in the list
    n_l = len(l)

    # If there are fewer than 3 elements in the list, then there
    # can't be any lucky triples, so return 0
    if n_l < 3:
        return 0, 0

    # Iterate over list in reverse
    lucky_triples = []  # List of all lucky triples
    for k in range(n_l-1, 1, -1):
        for j in range(k-1, 0, -1):
            # Check to see if lj divides lk
            if l[k] % l[j] == 0:
                # Iterate over all possible remaining numbers.  Here
                # we can just focus on the unique numbers.
                li_unique = set(l[0:j])
                for li in li_unique:
                    if l[j] % li == 0:
                        # Add the tuple to the list
                        lucky_triples.append((li, l[j], l[k]))

    # If no lucky  triples were found, return 0
    if lucky_triples == []:
        return 0, 0

    # Remove any duplicates and return the number of elements
    lucky_triples = set(lucky_triples)
    return len(lucky_triples), 0


if __name__ == '__main__':
    """
    l = [1, 1, 1]
    o_expected = 1
    o, l_o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')
    print(l_o)

    l = [1, 2, 3, 4, 5, 6]
    o_expected = 3
    o, l_o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')
    print(l_o)
    """

    l = [1, 2, 3, 4, 5, 6]
    o_expected = 3
    o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')

    l = [1, 1, 1]
    o_expected = 1
    o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')

    l = [1, 2, 3]
    o_expected = 0
    o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')

    l = [1, 1, 3]
    o_expected = 1
    o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')

    l = [3, 6, 5, 30, 18]
    o_expected = 2
    o = solution(l)
    print(f'Input: {l} -> Output: {o}, Expected: {o_expected}')