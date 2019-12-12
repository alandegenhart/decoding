# Practice Python Exercise 15 -- Reverse word order
#
# This script asks the user to input a string and then displays the strings in reverse order.


def reverse_string_order(s):
    # Split string into words
    s_list = s.split()

    # Reverse order of items in list
    s_list_rev = s_list[::-1]
    s_rev = ' '.join(s_list_rev)  # the 'join' method returns a string (doesn't change the object)

    return s_rev


def main():
    # Get input from the user
    s = input('Enter sentence to reverse: ')

    # Reverse sentence
    s_rev = reverse_string_order(s)

    # Print output
    print('Reversed sentence: "{}"'.format(s_rev))


# Call main function
main()
