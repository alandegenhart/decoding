# Python programming exercise 1 -- character input
#
# This function will prompt the user for their age and calculate the date when they will turn 100.

# Import modules
import datetime as dt

# Prompt user for their age -- this should be a string
date_str = input('Enter date of birth (mm/dd/yyyy): ')

# Convert age to some sort of date number and add 100
d = dt.datetime.strptime(date_str, '%m/%d/%Y')
# d2 = d + dt.timedelta(100 * 365)  # 'timedelta' does not support years -- Note this doesn't work due to leap years
d2 = d.replace(year = d.year + 100)  # Note that datetime.replace returns a copy - the original is not modified

# Re-format date number as a string and display to the user
date_str = dt.datetime.strftime(d2, '%m/%d/%Y')
print('You will turn 100 years old on: {}'.format(date_str))