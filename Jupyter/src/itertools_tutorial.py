# Ref: http://jmduke.com/posts/a-gentle-introduction-to-itertools/

import itertools

letters = ['a', 'b', 'c', 'd', 'e', 'f']
booleans = [1, 0, 1, 0, 0, 1]
numbers = [23, 20, 44, 32, 7, 12]
decimals = [0.1, 0.7, 0.4, 0.4, 0.5]

print(list(itertools.chain(letters,decimals)))
print(list(itertools.chain(letters, letters[3:])))

