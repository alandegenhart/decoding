# Python practice exercise 02:  Odd or even
#
# This script asks the user for input and determines if that input is odd or even.
#
# A note on quotes -- both single (') and double (") quotes are valid in Python.  However, these cannot be
# interchanged in the same string and thus can be both used when you want single or double quotes in the string.
# For example, if you want to use single quotes in the string, use double quotes to define it (and vice-versa).  Triple
# quotes (''') can also be used if both single and double quotes are desired in the same string.

# Prompt user for input
x = input('Enter a number : ')

# Input using the 'input' function are strings.  Need to first typecast this to an float.  This will generate an
# error if the input is not a float or integer, so need to handle this exception.
try:
    x = float(x)
except ValueError:
    print("ERROR: The provided input ('{}') cannot be converted to an integer.".format(x))
    exit(1)

# Now check to see if the number is actually a float or not.  If it is, display a message indicating that it has been
# rounded.
if x % int(x) != 0:
    print("Provided input ('{}') is not an integer and will be rounded accordingly.".format(x))

# Divide the result by 2 and check remainder to determine if it is odd or even.
x = int(x)
if x % 2 == 0:
    s = 'even'
else:
    s = 'odd'

# Display output message
print('The number {} is {}.'.format(x, s))
