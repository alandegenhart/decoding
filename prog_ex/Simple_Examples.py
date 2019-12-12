# Practice Python -- Simple examples
#
# This script has simple examples from the Practice Python website.  Excercises are placed here if they are too short
# or simple to warrant a standalone script.


# Exercise 07 -- List comprehensions
#
# In a single line of Python code, take a list of numbers and create a new list that only has the even numbers.

# To do this, use the mod operator (%), which returns the remainder after division.
r = 4 % 2  # Should return 0
r = 4 % 3  # Should return 1

# Define the list
a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Get all even numbers in a single line
b = [x for x in a if x % 2 == 0]


# Exercise 12 -- List ends
#
# Write a program that takes a list of numbers and makes a new list of the numbers at the start and end.

a = [5, 10, 15, 20, 25]

b = [a[0], a[-1]]

# Inefficient way to do it using list comprehensions
c = [a[i] for i in range(len(a)) if (i == 0) | (i == (len(a) - 1))]

