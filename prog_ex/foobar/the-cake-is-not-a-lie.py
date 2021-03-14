"""foo.bar -- 'The cake is not a lie'

Input:
solution.solution("abcabcabcabc")
Output:
    4

Input:
solution.solution("abccbaabccba")
Output:
    2

"""


def solution(s):
    """Find the number of substrings the input string s can be split into such that
    each string is the same.
    
    Input:
    Non-empty string less than 200 characters in length.
    
    Output:
    Maximum number of equal parts the cake can be cut into.
    """
    
    # Check to see if all elements in the string are the same.  If so, the
    # number of sub-strings is equal to the length of the string.
    s = s.lower()
    n_s = len(s)
    if len(set(s)) == 1:
        # All elements are the same
        return n_s

    # Get all possible factors
    n_s = len(s)
    factors = [i for i in range(2, n_s) if n_s % i == 0]

    # Check to see if all of the substrings are the same.  If so, return
    # Iterate over factors.  Note that we go through the factors in increasing
    # size to use them as the sub-string size.
    for f in factors:
        # Split string into sub strings
        sub_string_list = [s[i:i+f] for i in range(0, n_s, f)]

        # If all of the sub strings are the same, return the current factor
        if len(set(sub_string_list)) == 1:
            return int(n_s/f)  # Number of splits

    # If all factors have been tried and we have still not found an answer,
    # then there is only 1 possible split
    return 1


if __name__ == '__main__':
    s = 'abcabcabcabc'
    o_expected = 4
    o = solution(s)
    print(f'Input: {s} -> Output: {o}, Expected: {o_expected}')

    s = 'abccbaabccba'
    o_expected = 2
    o = solution(s)
    print(f'Input: {s} -> Output: {o}, Expected: {o_expected}')
