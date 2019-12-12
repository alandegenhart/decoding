# Practice Python Exercise 13 -- Fibonacci
#
# This script prompts the user for the length of a fibonacci sequence to generate and then outputs that sequence.


# Define function to return fibonacci sequence
def fibonacci(x):
    # Initialize y
    y = [1, 1]
    for i in range(2, x):
        # "Standard" way: using index
        # y[i] = y[i-1] + y[i-2]
        # Better way: use the append method
        y.append(y[i-1] + y[i-2])

    # Only return first x elements of sequence.  This is only needed if x is 1 or 2, b/c these elements are pre-defined.
    y = y[0:x]
    return y


# Main program
def main():
    # Get user input
    x = input('Enter length of Fibonacci sequence to generate: ')

    try:
        x = int(x)
    except ValueError:
        print('Error: input must be a number.  Quitting...')
        exit(1)

    y = fibonacci(x)
    print('Fibonacci sequence of length {}: {}'.format(x, y))


# Call main function
main()
