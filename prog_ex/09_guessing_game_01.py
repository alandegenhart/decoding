# Practice Python Exercise 09 -- Guessing game 1
#
# This script prompts the user to guess a number between 0 and 9 and tells them if they guessed high/low

import random

choose_num = True

# Continuously loop to get user input
while True:
    # Choose new number if necessary
    if choose_num:
        num = random.sample(range(1, 10), 1)[0]
        num_guesses = 0
        choose_num = False

    # Prompt user for input
    guess = input('Guess a number between 0 and 9 ("exit" to quit): ')
    num_guesses += 1

    # Check to see if the user wants to exit
    if guess == 'exit':
        break

    # Convert input to integer and determine the response
    guess = int(guess)
    if guess == num:
        # User guessed correctly.  Choose a new number
        msg = 'You guessed correctly!  Total number of guesses: {}.'.format(num_guesses)
        choose_num = True
    elif guess < num:
        # Guessed lower
        msg = 'You guessed lower.  Try again.'
    else:
        # Guessed higher
        msg = 'You guessed higher.  Try again.'

    # Display message
    print(msg)
