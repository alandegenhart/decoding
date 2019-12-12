# Practice Python Excercise 06 -- String lists
#
# This script asks the user for an input and then determines if the string is a palindrome or not.

# Ask user for input
s = input('Enter a string:')

# Create copy of string in reverse
sb = s[::-1]

# Compare two strings to determine if they are the same
if s == sb:
    print('The string "{}" is a palindrome.'.format(s))
else:
    print('The string "{}" is not a palindrome'.format(s))

# More difficult way -- using the list reverse function
s2 = list(s)  # Convert the string to a list
s2.reverse()  # Reverses the list object
s2 = ''.join(s2)

# The 'join' method above assumes the elements in the input list are all strings.  If this is not the case, the line
# will cause an error.  To prevent this, convert each element to a string:
s2 = ''.join(str(s) for s in s2)
