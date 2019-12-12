# Practice Python Exercise 4 -- Divisors
#
# This script asks the user for a number and returns all of the divisors of that number.

# Get user input and convert to a number
x = input('Enter a number: ')

# Verify that input number is an integer
try:
    x = float(x)

except ValueError:
    print('The provided in put must be a number.\n')
    exit(1)


# Convert input number to an integer in case it was a float
x = int(x)

if x < 1:
    print('The provided input must be greater than 0.')
    exit(1)

# Iterate over values and get those which the input is divisible by
a = [i for i in range(1, x+1) if x % i == 0]
print('The provided input ({}) is divisible by {}.'.format(x, a))