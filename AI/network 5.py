import random
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]
tema = 5
n_sensor = 15
weights = [0 for i in range(n_sensor)]

def perceptron(Sensor):
    b = 7
    s = 0
    for i in range(n_sensor):
        s += int(Sensor[i]) * weights[i]
        print(s)
        if s >=b:
            return True
        else:
            return False

def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] -= 1

def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:
            weights[i] += 1

def learn(n):
    for i in range(n):
        j = random.randint(0,9)
        r = perceptron(nums[j])

        if j != tema:
            if r:
                decrease(nums[j])
        else:
            if not r:
                increase(nums[tema])
    print(weights)

def print_matrix_3x5(digit):
    counter = 0
    for elements in digit:
        if counter<2:
            print(elements, end=" ")
            counter += 1
        else:
            print(elements)
            counter = 0
learn(100000)
print(perceptron(num5))
# for elements in nums:
#     print()
#     print_matrix_3x5(elements)
#     print()
#     print(perceptron(elements))
#     print('_____________')
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

print(perceptron(num51))
print(perceptron(num52))
print(perceptron(num53))
print(perceptron(num54))
print(perceptron(num55))
print(perceptron(num56))