# Задача 9
# Особая тройка Пифагора
# Тройка Пифагора - три натуральных числа a < b < c, для которых выполняется равенство
#  ** возведение в степень
# a**2 + b**2 = c**2
# Например, 3**2 + 4**2 = 9 + 16 = 25 = 5**2.
#
# Существует только одна тройка Пифагора, для которой a + b + c = 1000.
# Найдите произведение abc.
a, b, c = 1, 2, 3
while (a+b+c)!=1000:
    for b in range(c):
        for a in range(b):
            if (a ** 2) + (b ** 2) == c ** 2 and (a + b + c == 1000):
                print(a, b, c, a * b * c)
                break
    c += 1