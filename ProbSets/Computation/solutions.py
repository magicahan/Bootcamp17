# solutions.py
"""Volume IB: Testing.
<Name>
<Date>
"""
import math
import os.path
import numpy as np

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.


def addition(a, b):
    return a + b


def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a + b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other):
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here
# Problem 5: Write code for the Set game here
def read_sets(filename):
    #test filename
    if filename[-3:] != 'txt':
        raise ValueError('Wrong file name')
    if not os.path.isfile('hands/'+filename):
        raise ValueError('Wrong file name')
    
    card_list = []
    dup_list = []
    with open('hands/' + filename, 'r') as f:
        for line in f:
            line_list = list(line.strip())
            line_list = [int(i) for i in line_list]
            card_list.append(line_list)
            dup_list.append(line)
    card_array = np.array(card_list)
    #test invalid input
    valid_input = (card_array > 3) + (card_array < 1)
    if valid_input.sum() > 0:
        raise ValueError('Invalid Input')

    #test duplicate:
    card_sets = set(dup_list)
    if len(card_sets) != len(card_list):
        raise ValueError('Duplicate Cards')

    #test number of cards
    if len(card_list) != 12:
        raise ValueError('Insufficient amount of cards')
    
    return card_array

def compute_set(filename):
    card_array = read_sets(filename)
    card_list = card_array.tolist()
    card_comb = []
    sets_list = []
    #compute each possible combination
    for i, card1 in enumerate(card_list):
        if i + 2 >= len(card_list):
            break
        for j,card2 in enumerate(card_list[i+1:]):
            if i + 1 + j + 1 >= len(card_list):
                break
            for k, card3 in enumerate(card_list[i + 1 + j+1:]):
                card_comb.append([card1, card2, card3])
    for comb in card_comb:
        status = True
        for i in range(4):
            sum_col = comb[0][i] + comb[1][i] + comb[2][i]
            if sum_col%3 != 0:
                status = False
        if status:
            sets_list.append(comb)
    return len(sets_list)
