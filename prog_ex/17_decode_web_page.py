""" Practice Python Exercise 17 -- Decode a web page

This exercise prints out all of the titles on a particular web page.  The function pulls the HTML from the site and
then grabs the specific elements of choice.

The current version of the code grabs all span elements and prints them.  It seems both the 'span' and 'h2' tags define
the headers for the website.  Obviously this could be improved to search these, but without knowing more about how the
website is structured this is probably good enough to get the point across.

"""

# Import packages
import requests
from bs4 import BeautifulSoup

# Pull all HTML from the website
url = 'https://www.nytimes.com'
r = requests.get(url)
r_text = r.text

# Print raw text -- do this to identify the tag of interest.  Note that this is not really that helpful, as the HTML is
# buried in a bunch of other markup.
# print(r_text)

# Now get elements based on HTML tag.  The 'span' elements seems to contain the article titles
soup = BeautifulSoup(r_text, 'html.parser')
class_id = 'esl82me0'
span_elements = soup.find_all('span')  # Returns the set of all span elements

# Alternatively, the following works better (the 'esl82me0' class seems to be a hidden element around all stories:
title_elements = soup.find_all(class_=class_id)

# Get list of all span elements that aren't empty
article_titles = [s.string for s in title_elements if s.string is not None]

# Print list
print_string = '\n'.join(article_titles)  # The 'join' method will join a list of strings (the input is an iterable)
print('\nAll article titles (class = {}): \n\n{}\n'.format(class_id, print_string))
