# Задача 7
# 10001-е простое число
# Выписав первые шесть простых чисел, получим 2, 3, 5, 7, 11 и 13. Очевидно, что 6-е простое число - 13.
#
# Какое число является 10001-м простым числом?
divided= 1
divider_total = 0
while divided>0:
    divider = divided
    divider_count = 0
    step = 1
    while divider>0:
        if divided % divider == 0:
            divider_count += 1
        if step == 1:
            divider = divider // 2
            step += 1
        else:
            divider -= 1
        if divider_count == 3:
            break
    if divider_count == 2:
        divider_total += 1
    if divider_total == 10001:
        print("10001 simple divider is", divided)
        break
    divided+=1


