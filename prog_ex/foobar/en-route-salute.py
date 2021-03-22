def solution(s):

	# The dash characters are meaningless, so we can remove these
	s = ''.join([si for si in s if si != '-'])

	count = 0
	for i in range(len(s)):
		# Get the current item to compare
		si = s[i]

		# If the item is '<', do nothing
		if si == '<':
			continue

		# If the item is '>', count the number of times '<' is present
		# in the rest of the string, multiply by 2, and add to the
		# count.
		if si == '>':
			count += s.count('<', i) * 2

	return count


if __name__ == '__main__':
	s = '--->-><-><-->-'
	o_expected = 10
	o = solution(s)
	print(f'Input: {s} -> Output: {o}, Expected: {o_expected}')

	s = '>----<'
	o_expected = 2
	o = solution(s)
	print(f'Input: {s} -> Output: {o}, Expected: {o_expected}')

	s = '<<>><'
	o_expected = 4
	o = solution(s)
	print(f'Input: {s} -> Output: {o}, Expected: {o_expected}')
