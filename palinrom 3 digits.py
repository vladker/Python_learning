# Largest palindrome product
# Show HTML problem content
# Problem 4
# A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 × 99.
#
# Find the largest palindrome made from the product of two 3-digit numbers.
def palindrome_multiply_find(first, second, razryadi):
    palinrome_list = []
    x, y = first, second
    while x > razryadi:
        y = second
        while y > razryadi:
            string1 = str(x * y)
            string2 = string1[::-1]
            if string1 == string2:
                palinrome_list.append(x * y)
                # print('Palindrome ', x*y, ' ',x , ' * ', y)
                break
            y -= 1
        x -= 1
    return palinrome_list

list_to_show=max(palindrome_multiply_find(999, 999, 100))

print(list_to_show)

