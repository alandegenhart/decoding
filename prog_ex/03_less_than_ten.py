# Practice Python Exercise 3 -- Less than ten
#
# This exercise takes a list of numbers and prints out those that are less than a specific value

import numpy as np

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# The easy way -- using boolean indexing.  This requires using a numpy array, as boolean indexing is not defined for
# generic lists.
b = np.array(a)
print('Elements less than 10: {}'.format(b[b < 10]))

# More difficult -- iterate over elements and append to list.  Note that 'append' is a method for a list object
d = []
for x in a:
    if x < 10:
        d.append(x)

print('Elements less than 10: {}'.format(d))

# Less difficult -- use list comprehension
e = [x for x in a if x < 10]
print('Elements less than 10: {}'.format(e))