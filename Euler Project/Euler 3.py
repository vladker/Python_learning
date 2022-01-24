# App to show max simple divider of defined number (start ot end)
def max_simle_divider_straight(number):
    i = 1
    dividers_list=[]
    while i < number:
        if number % i == 0:
            number_of_dividers = 0
            j = 1
            while j < i:
                if i % j == 0:
                    number_of_dividers += 1
                if number_of_dividers > 1:
                    break
                j += 1

            if number_of_dividers == 1:
                dividers_list.append(i)
        i += 1
    return max(dividers_list)

def max_simle_divider_reverse(number):
    i = number//2
    dividers_list=[]
    while i > 0:
        if number % i == 0:
            number_of_dividers = 0
            j = 1
            while j < i//2:
                if i % j == 0:
                    number_of_dividers += 1
                if number_of_dividers > 1:
                    break
                j += 1

            if number_of_dividers == 1:
                dividers_list.append(i)
        i -= 1
    return max(dividers_list)
print(max_simle_divider_reverse(600851475143))
