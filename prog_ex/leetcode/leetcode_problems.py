#%% Setup

# Setup autoreload
%reload_ext autoreload
%autoreload 2

# Note -- the leetcode problems use the typing library
from typing import *  # Not necessarily best practice

#%% 412. Fizz Buzz

"""412. Fizz Buzz
Write a program that outputs the string representation of numbers. For
multiples of 3 it should output 'Fizz', for numbers of five it should output
'Buzz'. For multiples of both three and five it should output 'FizzBuzz'.
"""

class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        # Handle case where n is zero
        if n < 1:
            return []

        # Iterate over numbers
        output = []
        for i in range(1, n+1):
            # Check for divisibility
            if (i % 3) == 0 and (i % 5) == 0:
                output.append('FizzBuzz')
            elif (i % 5) == 0:
                output.append('Buzz')
            elif (i % 3) == 0:
                output.append('Fizz')
            else:
                output.append('{}'.format(i))

        return output

# Test
output = Solution().fizzBuzz(15)
print('{} (naive)'.format(output))

# --- Alternative approaches --

# There is a multiplication trick you can do in Python:
s = 'String_1' * False + 'String_2' * True  # will be 'String_2'

# So, to do things in one shot:
n = 15
s = [str(i) * (i % 3 != 0 and i % 5 != 0)
     + 'Fizz' * (i % 3 == 0)
     + 'Buzz' * (i % 5 == 0)
     for i in range(1, n + 1)]
print('{} (list comprehension)'.format(output))

# Another way to do this would be to use a dict.  A key advantage of this
# approach is that it is very easy to add new conditions by modifying the
# dictionary.
cond = {
    3: 'Fizz',
    5: 'Buzz'
}

# This allows you to iterate over the keys
n = 15
output = []
for i in range(1, n+1):
    # Iterate through all conditions to check
    s = ''  # Initialize s
    for k in cond.keys():
        if i % k == 0:
            s += cond[k]

    # Check if the string is still empty.  If so, add the current number
    if s == '': s = str(i)

    # Add result to output string
    output.append(s)

print('{} (dictionary)'.format(output))

#%% 387. First unique character in a string

"""387. First unique character in a string

Given a string, find the first non-repeating character in it and return its
index. If it doesn't exist, return -1.

Examples:
    input: "leetcode"
    output: 0
    
    input: "loveleetcode"
    output: 2

"""

class Solution:
    def firstUniqChar(self, s: str) -> int:
        """Start with a brute force approach. Iterate over the elements in the
        string. In theory, this *should* work, but apparently takes too long
        for really long strings.
        """
        # Handle unexpected input (empty strings). This actually shouldn't be
        # necessary given the current implementation.
        if s == '':
            return -1

        n = len(s)  # Number of elements in the string

        # If there is only one character, then we know it must be unique.
        if n == 1:
            return 0

        # Iterate over elements in the string.
        for i in range(n):
            # Keep track of if a duplicate has been found yet
            flag = False

            # Iterate over remaining elements
            for j in range(i+1, n):
                if s[i] == s[j]:
                    flag = True

            # A duplicate was not found, return the index. Also want to make
            # sure that the letter isn't in the previous segment of the string.
            if not flag and s[i] not in s[:i]:
                return i

        # If the entire string has been searched and no unique element has been
        # found, return -1
        return -1


class Solution:
    def firstUniqChar(self, s: str) -> int:
        """An alternative approach is to first find all unique elements in the
        string and find the number of times that element is in the string.

        The issue with iterating over the string element-by-element is that
        the time required to complete this operation will grow with the size
        of the string. Instead, the set operation can be used to find the
        unique elements in the string. We can then search for the smallest
        index for all elements that only appear once.
        """
        # Handle unexpected input (empty strings). This actually shouldn't be
        # necessary given the current implementation.
        if s == '':
            return -1

        uni_char = set(s)
        uni_occ = []
        for uc in uni_char:
            # Get the number of times the element is in the string
            count = s.count(uc)

            # If the character was only found once, find its index
            if count == 1:
                uni_occ.append(s.find(uc))

        # Check to see if the list of unique indices is empty. If so, there are
        # no non-duplicate values.
        if not uni_occ:
            return -1

        # Return the smallest number in the list of unique indices
        return min(uni_occ)


# Test
S = Solution()
S.firstUniqChar('dddccdbba')


# -- Alternative approaches --
class Solution:
    def firstUniqChar(self, s: str) -> int:
        """
        A more elegant approach is to use the built-in 'Counter', which
        generates a hash map. The idea here is that you iterate over the entire
        string once and keep track of the number of occurrences of each element.
        You can then iterate through the elements in this map and return the
        first item with only one occurrence.
        """
        import collections
        count = collections.Counter(s)  # Generates the hash map

        # Iterate over elements
        for idx, k in enumerate(count):
            if count[k] == 1:
                # Once the first non-duplicated element is found, find its
                # corresponding index. This is necessary because the Counter
                # map will only have values for the unique elements. Thus, the
                # index of the map cannot be used as the index in the string.
                return s.find(k)  # This is

        # If no counts were 1, return -1
        return -1


# Test
S = Solution()
S.firstUniqChar('dddccdbba')


#%% 350. Intersection of two arrays 2

"""350. Intersection of two arrays 2.

Given two arrays, write a function to find their intersection.

Examples:
    Input: nums1 = [1, 2, 2, 1], nums2 = [2, 2]
    Output: [2, 2]
    
    Input: nums1 = [4, 5, 9], nums2 = [9, 4, 9, 8, 4]
    Output: [4, 9]
"""


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """Intersection of two arrays

        Return the intersection of two lists, including repeats.

        """

        intersect_list = []
        for n1 in nums1:
            # Check to see if n1 is in nums2
            if n1 in nums2:
                # Add item to intersect list and remove from nums2
                intersect_list.append(n1)
                nums2.remove(n1)

        return intersect_list


nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
S = Solution()
S.intersect(nums1, nums2)

nums1 = [4, 5, 9]
nums2 = [9, 4, 9, 8, 4]
S = Solution()
S.intersect(nums1, nums2)

# Other solutiosn:
# (1) Use two pointers. Seems like this might only be a valid solution if the
# arrays were sorted, however.
# (2) Use dictionaries to count.

# Other considerations --

#%%