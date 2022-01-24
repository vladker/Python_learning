# 2520 - самое маленькое число, которое делится без остатка на все числа от 1 до 10.
#
# Какое самое маленькое число делится нацело на все числа от 1 до 20?

# надо постараться использовать модуль перебора числа и для делимого и для делителя
# 232792560
n = 20
divider_list = list(_ for _ in range(1, n + 1))
divider_count = 0
divided = 1
while divider_count != n:
    print(divided)
    for divider in divider_list:
        if divided % divider == 0:
            divider_count += 1
    if divider_count==n:
        print(f"Наименьшее кратное {divided}")
        break
    divider_count = 0
    divided += 1
