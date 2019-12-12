# Practice Python Exercise 16 -- Password generator
#
# This script generates a random password based on user input.


def generate_password(num_char):
    """ Generate a random password of a specified length

    Some considerations:
    - Want to split the password into a number of smaller elements, where the number of elements is randomly-chosen.
    - Randomly assign each of these elements to several options: word, number, special character

    :param num_char:
    :return password:
    """

    import random

    # Define word list, number list, and special character list
    word_file = "/usr/share/dict/words"
    word_list = open(word_file).read().splitlines()
    num_list = [str(i) for i in range(0,10)]
    char_list = ['[', ']', '#', '$', '%', '*', '&']

    # Define some parameters
    element_names = ['word', 'num', 'char']
    length_bounds = {'word': [3, 10],
                     'num': [1, 3],
                     'char': [0, 1]}

    # Start adding elements
    password = ''
    while len(password) < num_char:
        # Choose an element
        el = element_names[random.sample(range(3), 1)[0]]

        # Get the length bounds for the current element and update depending on the current length of the password
        el_range = length_bounds[el]
        char_remain = num_char - len(password)
        if char_remain < el_range[0]:
            continue  # Minimum length of element is too short, choose again
        elif char_remain < el_range[1]:
            el_range[1] = char_remain

        # Choose the number of characters for the current element
        if el_range[0] == el_range[1]:
            # Handle case where there is only one possible length.  In this case don't randomly sample, as this will
            # cause an error.
            num_el_char = el_range[0]
        else:
            num_el_char = random.sample(range(el_range[0], el_range[1]), 1)[0]

        # Generate random string of the desired length
        if el == 'word':
            # Choose a random word of the appropriate length
            valid_words = [w for w in word_list if len(w) == num_el_char]
            idx = random.sample(range(0, len(valid_words)), 1)[0]
            el_str = valid_words[idx]
        elif el == 'num':
            # Generate a random number of the appropriate length
            idx = random.sample(range(0, len(num_list)), num_el_char)
            el_str = ''.join([num_list[i] for i in idx])
        elif el == 'char':
            # Choose a random special character
            el_str = char_list[random.sample(range(len(char_list)), 1)[0]]

        # Add the new element to the password
        password += el_str

    return password


def main():
    """ Main function

    :return:
    """

    # Prompt user for the length of the password
    num_char = int(input('Length of password: '))

    # Generate password
    password = generate_password(num_char)

    # Output generated password
    print('Password: {}'.format(password))


# Run main function
main()
