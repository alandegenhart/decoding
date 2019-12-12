"""
7. Reverse integer

Given an 32-bit signed integer, reverse the digits of an integer.

Example:
    Input: 123
    Output: 321

"""

class Solution:
    def reverse(self, x: int) -> int:
        # One way to do this is to iterate through powers of 10 and find the
        # remainder after division.  The reversed number would then be the
        # remainder multiplied by the previous digit to the power of 10.
        import numpy as np

        # Handle case where the input is zero.
        if x == 0:
            return 0

        sgn = np.sign(x)
        x = np.abs(x)
        pow = range(11, -1, -1)  # 10^0 to 10^10
        r = np.array([])
        x_temp = x
        for p in pow:
            n = int(x / 10**p)
            r = np.append(r, n)
            if n > 0:  # If the remainder is not zero, then keep the digit
                x = x % 10**p  # Update x with the remainder

        # r now contains the digits that need to be reversed.  However, we first
        # need to remove any leading zeros
        i = (r > 0).nonzero()  # i is an array of ndarray
        r = r[i[0][0]:]

        # Now iterate over the elements in r
        x = 0
        for i in range(len(r)):
            x = x + r[i] * 10**i

        x = x * sgn
        if (x < -2**31) | (x > (2**31 - 1)): x = 0

        return int(x)

S = Solution()
S.reverse(123)
S.reverse(-123)
S.reverse(0)  # Edge case -- causes a runtime error if not handled explicitly


"""
9. Palindrome number

Determine whether an integer is a palindrome.

Example 1:
    Input: 121
    Output: true

Example 2:
    Input: -121
    Output: false (is '121-' in reverse)

Example 3:
    Input: 10
    Output: false (reads '01' in reverse)

"""

class Solution:
    def isPalindrome(self, x: int) -> bool:
        # For purposes of this exercise, it is probably easiest to convert the
        # input into a string and reverse it.  Doing things this way should
        # make the problem fairly easy.
        s = str(x)
        s_rev = s[::-1]
        return s == s_rev

S = Solution()
S.isPalindrome(121)
S.isPalindrome(-121)
S.isPalindrome(10)


"""
13. Roman numeral to integer

Convert a roman numeral to an integer.

I - 1
V - 5
X - 10
L - 50
C - 100
D - 500
M - 1000

Rules:
- Each letter can be repeated up to 3 times
- The preceeding letter can be placed in front to decrease the value (IV = 4)

Example 1:
    Input: "III"
    Output: 3

Example 2:
    Input: "IV"
    Output: 4

Example 3:
    Input: "IX"
    Output: 9

Example 4:
    Input: "LVIII"
    Output: 58
"""

class Solution:
    def romanToInt(self, s: str) -> int:
        # Define two dicts. The first one indicates the order of the letters
        # from largest to smallest. This is used to determine the sign of the
        # numbers corresponding to each letter. The second dict specifies the
        # absolute value of the number for each letter.
        codes = {
            'M': 1,
            'D': 2,
            'C': 3,
            'L': 4,
            'X': 5,
            'V': 6,
            'I': 7
        }
        values = {
            'M': 1000,
            'D': 500,
            'C': 100,
            'L': 50,
            'X': 10,
            'V': 5,
            'I': 1
        }

        # Approach: iterate through the letters in the input.  If the current
        # letter specifies a lower number and is before a higher number (e.g.,
        # 'I' before 'M'), then that number is negative. Once this is done, the
        # Number multiplied by the sign can simply added to the overall sum.

        # Initialize the output
        num = 0

        # Iterate over the letters in the input string
        for i in range(len(s)):
            # First determine the sign. If the code for the current letter is
            # greater than that of the next letter, the sign is negative.
            # Otherwise, the sign is positive.
            if i == len(s) - 1:
                # If this is the last letter, the sign will always be positive
                sign = 1
            elif codes[s[i]] > codes[s[i+1]]:
                # If the code for the current number is less than the code for
                # the next number, then the sign is negative
                sign = -1
            else:
                sign = 1

            # Add the number for the current letter to the total number
            num = num + (values[s[i]] * sign)

        return num

S = Solution()
S.romanToInt('III')
S.romanToInt('IV')
S.romanToInt('IX')
S.romanToInt('LVIII')


"""
14. Largest common prefix

Write a function to find the longest common prefix from an array of strings. If
there is no common prefix, return and empty string "".

Example 1:
    Input: ["flower", "flow", "flight"]
    Output: "fl"

Example 2:
    Input: ["dog", "racecar", "car"]
    Output: ""

"""

from typing import List
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # Handle case where the input is empty.
        if len(strs) == 0:
            return ''

        # Find the shortest element
        l = len(min(strs, key=len))

        # Iterate over elements in all
        prefix = ''
        for i in range(l):
            # Check to see if all strings in the list have the same letter.
            # Make sure to lower the case of all strings.
            letter = []
            for s in strs:
                letter.append(s[i].lower())

            # Compare all of the letters to see if they are the same.  If so,
            # add the letter to the list
            if len(set(letter)) == 1:
                prefix = prefix + letter[0]
            else:
                break

        # Return the answer
        return prefix

S = Solution()
S.longestCommonPrefix(['Flower', 'flow', 'flight'])
S.longestCommonPrefix(['dog', 'racecar', 'car'])
S.longestCommonPrefix([])


"""
20. Valid parentheses

Given a string containing just the characters '(', ')', '[', ']', '{', or '}',
determine if the input string is valid.  An input string is valid if:

1. Open brackets are closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

An empty string is also considered valid.

Example 1:
    Input: '()'
    Output: true

Example 2:
    Input: '()[]{}'
    Output: true

Example 3:
    Input: '(]'
    Output: false

Example 4:
    Input: '([)]'
    Output: false

Example 5:
    Input: '{[]}'
    Output: true

"""

class Solution:
    def isValid(self, s: str) -> bool:
        # Approach: use a list as a queue -- If an open bracket is encountered,
        # push the appropriate closed bracket onto the queue. If a closed
        # bracket is encountered, compare it to the most recent element on the
        # queue. If the next element in the string is not the most recent
        # element on the queue, return FALSE. If the entire string is parsed,
        # return TRUE.

        # Check to see if the number of elements in the string is even. If not,
        # the input string cannot be valid.
        if len(s) % 2 != 0:
            return False

        # Define open and closed bracket sets
        valid_open = {'(', '[', '{'}
        valid_closed = {')', ']', '}'}
        bracket_pairs = {
            '(': ')',
            '[': ']',
            '{': '}',
        }

        # Iterate over string
        buffer = []
        for elem in s:
            if elem in valid_open:
                buffer.append(bracket_pairs[elem])

            if elem in valid_closed:
                # Check to make sure the queue is not empty.  If it is, then
                # return False, since this would mean there was no matching
                # open bracket.
                if len(buffer) == 0:
                    return False

                if elem != buffer.pop():
                    return False

        # If the entire string has been parsed, check to make sure there are
        # no elements remaining in the cue. If so, the string is invalid.
        if len(buffer) > 0:
            return False
        else:
            return True

S = Solution()
S.isValid('()')
S.isValid('()[]{}')
S.isValid('(]')
S.isValid('([)]')
S.isValid('{[]}')
S.isValid('')
S.isValid('((')
S.isValid('){')


"""
21. Merge two sorted lists

Merge two sorted linked lists and return the merged list. The new list should be
made by splicing together the nodes of the first two lists.

A singly-linked list contains two elements: a value and a link to the next
element.

Example:
    Input: 1->2->4, 1->3->4
    Output: 1->1->2->3->4->4

"""

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        We can take advantage of the fact that the two input lists are already
        sorted. To do this, step through the two lists at the same time.
        Add the smaller of the two to the merged list and repeat until the end
        of one of the two lists is found. At this point, the next element in the
        merged list can be set to the next element in the list that has not
        been fully parsed.
        """

        # Initialize the ListNode. Here we create two identical pointers so that
        # we can keep track of the starting point of the list.
        l_a = l_b = ListNode(0)

        # While both lists have not been parsed, keep adding the smaller of the
        # two values to the merged list
        while l1 and l2:  # True as long as one element isn't 'None'
            if l1.val <= l2.val:
                l_a.next = l1  # Update the pointer to the next element
                l1 = l1.next
            else:
                l_a.next = l2
                l2 = l2.next

            l_a = l_a.next  # Update pointer

        # Once the while loop has completed, either l1 or l2 will still have
        # elements left. Thus, just point the merged list to the node that is
        # not 'None'
        l_a.next = l1 or l2
        return l_b.next

# Define lists for testing
l1 = ListNode(1)
l1.next = ListNode(2)
l1.next.next = ListNode(4)

l2 = ListNode(1)
l2.next = ListNode(3)
l2.next.next = ListNode(4)

# Test
S = Solution()
l3 = S.mergeTwoLists(l1, l2)


"""
26. Remove duplicates from a sorted array

Given a sorted array, remove the duplicates in-place such that each element
appears only once. Return the new length of the array. For purposes of this
exercise, assume that the array is passed by reference, meaning that sorting the
list in the function will also update the list outside of the function.

Example 1:
    Input: [1, 1, 2]
    Output: 2

Example 2:
    Input: [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    Output: 5

"""

from typing import List
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        Approach

        Define:
            uni_num -- current unique number
            n_uni_el -- current number of elements

        - Initialize UNI_NUM to the first element, N_UNI_EL to 1
        - Iterate through the list one element at a time
        - If the current number in the list is greater than UNI_NUM, update the
            list accordingly and increment N_UNI_EL
        """

        # Check to see if input list is empty
        n_el = len(nums)
        if n_el == 0:
            return 0

        # Iterate through list
        uni_num = nums[0]
        n_uni_el = 1
        for i in range(1, n_el):
            # Check to see if the current number is greater than the current
            # unique number
            if nums[i] > uni_num:
                uni_num = nums[i]
                nums[n_uni_el] = uni_num
                n_uni_el += 1

        return n_uni_el

S = Solution()
nums = [1, 1, 2]
S.removeDuplicates(nums)
print(nums)

nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
S.removeDuplicates(nums)
print(nums)


"""
27. Remove element

Given an array NUMS and a value VAL, remove all instances of that value
in-place and return the new length. Do not allocate extra space for another
array, this must be done by modifying the input array in-place with O(1) extra
memory. The order of the elements can be changed. It doesn't matter what is left
beyond the new length.

Example 1:
    Input: nums = [3, 2, 2, 3], val = 3
    Output: 2, first two elements of nums should be [2, 2]

Example 2:
    Input: nums = [0, 1, 2, 2, 3, 0, 4, 2], val = 2
    Output: 5, first 5 elements contains 0, 1, 3, 0, and 4

"""

from typing import List
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        # Get the number of elements in the input list
        n = n_el = len(nums)

        # Iterate over elements in the list
        for i in range(n):
            # Check to see if the current number in the list should be removed
            # Note that this might need to happen multiple times.
            while nums[i] == val:

                # Decrement number of elements
                n_el -= 1

                # Shift all elements
                for j in range(i, n-1):
                    nums[j] = nums[j+1]

                # Once n_el is equal to i+1, all numbers have been parsed
                if (i >= n_el) or n_el == 0:
                    return n_el

        # Need to return n_el if the entire list has been parsed
        return n_el

# Test cases
S = Solution()

nums = [3, 2, 2, 3]
val = 3
S.removeElement(nums, val)

nums = [0, 1, 2, 2, 3, 0, 4, 2]
val = 2
S.removeElement(nums, val)

nums = [4, 5]
val = 4
S.removeElement(nums, val)

nums = [3, 3]
val = 3
S.removeElement(nums, val)


"""
28. Implement strStr()

Return the first occurance of the substring NEEDLE in the longer string
HAYSTACK. Return '-1' if the substring is not part of the string.

Example 1:
    Input: haystack = 'hello', needle = 'll'
    Output: 2

Example 2:
    Input: haystack = 'aaaaa', needle = 'bba'
    Output: -1

"""

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        """
        Approach

        - Iterate though HAYSTACK element-by-element
        - Check the next substring of the same length as NEEDLE
        - If the string is the same, return the current index
        - If the entire string has been parsed, return '-1'
        """

        # If the substring is empty, return 0
        n = len(haystack)  # Number of elements in the search string
        n_sub = len(needle)  # Number of elements in the input string
        if n_sub == 0:
            return 0

        # Check to see if the search string is longer than the input string.
        # In this case, return -1
        if n < n_sub:
            return -1

        # Iterate over elements in the string
        for i in range(0, n-n_sub+1):
            if needle == haystack[i:i+n_sub]:
                # Substring has been found, return the current index
                return i

        # The entire string has been parsed w/o finding the string, return -1
        return -1

# Test cases
S = Solution()

haystack = 'hello'
needle = 'll'
S.strStr(haystack, needle)

haystack = 'aaaaa'
needle = 'bba'
S.strStr(haystack, needle)

haystack = 'test'
needle = 'st'
S.strStr(haystack, needle)

haystack = 'a'
needle = 'a'
S.strStr(haystack, needle)
