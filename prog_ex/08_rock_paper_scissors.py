# Practice Python Exercise 08 -- Rock, paper, scissors
#
# This script allows the user to play rock, paper, scissors.  The user is prompted for one of three possible inputs,
# randomly chooses an input for the computer, and displays the result.

import random as rnd


def choose_winner(sel):
    # Function to determine the winner.  This function implements a if/elif chain that is generally fragile and
    # inefficient.
    if sel[0] == sel[1]:
        result = 0
    elif sel == ('rock', 'paper'):
        result = 2
    elif sel == ('rock', 'scissors'):
        result = 1
    elif sel == ('paper', 'rock'):
        result = 1
    elif sel == ('paper', 'scissors'):
        result = 2
    elif sel == ('scissors', 'rock'):
        result = 2
    elif sel == ('scissors', 'paper'):
        result = 1
    else:
        result = None

    return result


def choose_winner_2(sel):
    # Function to choose the winner.  This implements a more elegant solution where the selection is used as the key
    # into a dictionary of possible options.

    if sel[0] == sel[1]:
        result = 0
    else:
        all_outcomes = {
            ('rock', 'paper'): 2,
            ('rock', 'scissors'): 1,
            ('paper', 'rock'): 1,
            ('paper', 'scissors'): 2,
            ('scissors', 'rock'): 2,
            ('scissors', 'paper'): 1}

        result = all_outcomes[sel]

    return result


# Define result strings
output_str = ['You and the computer tied.',
              'You win!',
              'The computer wins.']

# Continuously loop until the user decides to exit
while True:
    # Get input from the user
    s = input('Enter an input (rock, paper, or scissors).  Type "exit" to quit: ')

    # Convert input to lower case, define valid options
    s = s.lower()  # Note: the 'lower' function does not chance the input object
    valid_options = ['rock', 'paper', 'scissors']

    if s in valid_options:
        # Choose a random input for the computer player
        idx = rnd.sample(range(len(valid_options)), 1)[0]
        print('The computer choose {}.'.format(valid_options[idx]))

        # Determine the winner
        r = choose_winner_2((s, valid_options[idx]))

        # Display the result
        print(output_str[r])
    elif s == 'exit':
        # Break out of loop
        break
    else:
        # Inform the user that the input was invalid
        print('Invalid input.')
