# Practice Python Exercise 11 -- Check primality
#
# This script prompts the user for an input and then lets them know if the number is prime or not.


# Define function to return all divisors
def find_divisors(x):
    # Take absolute value of input number
    x = abs(x)

    # Find all numbers from 0 to x where the remainder after division is zero
    d = [i for i in range(1, x+1) if x % i == 0]
    return d


def main():
    # Prompt user for input
    x = input('Enter a number: ')
    x = int(x)

    # Determine if number is prime
    y = find_divisors(x)
    s = ''
    if y != [1, x]:
        s = 'not '

    # Display result
    print('The number {} is {}prime.'.format(x, s))


main()
