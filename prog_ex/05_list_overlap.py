# Practice Python Exercise 05 -- List overlap
#
# This script takes two lists and identifies the elements that are common to both (ignoring duplicates).  The lists can
# be different sizes.
#
# Extras:
#   - Generate the lists randomly
#   - Write in a single line of Python

# Import packages
import random

# Define lists to compare
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# [METHOD 1] Difficult way 1: iterate over lists

# Iterate over a, find matching elements in b.  The resultant list could contain duplicates
c = []
for x in a:
    for y in b:
        if x == y:
            c.append(x)

# Now remove any duplicates.  Loop over each element in c.  If another identical element exists, don't include it in
# the unique list.

# Initialize empty list to contain unique numbers
d = []

# Loop over all elements in c
for x in c:
    # Define counter to keep track of duplicates
    ctr = 0
    # Loop over current elements in the unique list
    for y in d:
        # If the element of c is already in the list, increment the counter
        if x == y:
            ctr += 1

    # If the number does not yet exist in the list, append it
    if ctr == 0:
        d.append(x)

# Display the result
print('[Method 1] Unique numbers common to both sets: {}'.format(d))

# [METHOD 2] Somewhat easier way: use the 'in' operator

c = [x for x in a if x in b]  # Will find duplicates
d = []
for x in c:
    if x not in d:
        d.append(x)

print('[Method 2] Numbers in both list a and b: {}'.format(d))

# [METHOD 3] Nested conditionals

c = []
for x in a:
    if x in b:
        if x not in c:
            c.append(x)

print('[Method 3] Unique numbers: {}'.format(c))

# [METHOD 4] Easier way:  convert elements to sets, find intersection

# Convert lists to sets.  Note that by doing this, any duplicates are removed.  A set is a unique sequence of objects
# that cannot be iterated over.
sa = set(a)
sb = set(b)

# Find the intersection between the two sets
c = sa.intersection(sb)
print('[Method 4] Intersection between two sets: {}'.format(c))

# Extra -- define random sets
max_len = 30  # Define maximum length of two lists
max_num = 100  # Define maximum number

# Randomly sample to determine length of each list
list_len = random.sample(range(1, max_len + 1), 2)

# Randomly sample to define the two lists
a2 = random.sample(range(max_num), list_len[0])
b2 = random.sample(range(max_num), list_len[1])

# Print random lists
print('Randomly-generated list 1: {}'.format(a2))
print('Randomly-generated list 2: {}'.format(b2))

sa = set(a2)
sb = set(b2)

# Find the intersection between the two sets
c = sa.intersection(sb)
print('Intersection between random sets: {}'.format(c))
