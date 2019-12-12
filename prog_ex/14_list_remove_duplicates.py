# Practice Python Exercise 14 -- List remove duplicates
#
# This function prompts the user for an input list and returns the unique elements of the list.


# Get unique elements using a loop
def unique_1(x):
    # This will not work.  This is because the result of the list comprehension isn't bound to y until it is finished.
    y = []
    y = [i for i in x if i not in y]

    # An alternative way of doing the same thing that kind of works -- it generates a bunch of 'nones'
    y = []
    [y.append(i) for i in x if i not in y]

    # An alternative way that works.  The append() method adds the *entire* input argument as a new element to the end
    # of the object.  Thus, appending '[2, 3]' to [1] will result in '[1, [2, 3]]'.  The extend() method operates on a
    # list, and is thus iteratable.
    y = []
    y.extend(i for i in x if i not in y)
    return y


# Get unique elements using sets
def unique_2(x):
    x = set(x)
    return list(x)


# Main function
def main():
    # Get user input
    s = input('Enter a list of numbers: ')

    # Convert string to list of numbers
    x = [int(i) for i in s if i.isnumeric()]

    # Get unique numbers using both methods
    y = unique_1(x)
    y2 = unique_2(x)

    # Output result
    print('Unique elements of the list (Method 1): {}'.format(y))
    print('Unique elements of the list (Method 2): {}'.format(y2))


# Call main function
main()
