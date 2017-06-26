# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""

import solutions as soln
import pytest

# Problem 1: Test the addition and fibonacci functions from solutions.py
def test_addition():
    assert soln.addition(2, 3) == 5, 'Addtion failed on adding two positive integers.'
    assert soln.addition(-8, -10) == -18, 'Addition faild on adding two negative integers.'
    assert soln.addition(-8, 6) == -2, 'Addition Failed on adding positive and negative integers.'
    
def test_smallest_factor():
    assert soln.smallest_factor(1) == 1, 'Smallests factor fails on {}.'.format(1)
    assert soln.smallest_factor(5) == 5, 'Smallest factor fails on {}.'.format(5)
    assert soln.smallest_factor(21) == 3, 'Smallest factor fails on {}.'.format(21) 

# Problem 2: Test the operator function from solutions.py
def test_operator():
    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 2, 1)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper should be a string"

    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 2, 'ab')
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper should be one character"

    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 0, '/')
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "You can't divide by zero!"

    assert soln.operator(1, 2, '+') == 3, 'Addition operation is wrong'
    assert soln.operator(4, 2, '/') == 2, 'Division Operation is Wrong'
    assert soln.operator(2, 3, '-') == -1, 'Subtraction operation is wrong.'
    assert soln.operator(4, 4, '*') == 16, 'Multiplication operation is wrong'
    

    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 2, '{')
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper can only be: '+', '/', '-', or '*'"   

# Problem 3: Finish testing the complex number class
@pytest.fixture
def set_up_complex_nums():
    number_1 = soln.ComplexNumber(1, 2)
    number_2 = soln.ComplexNumber(5, 5)
    number_3 = soln.ComplexNumber(2, 9)
    return number_1, number_2, number_3

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1 + number_2 == soln.ComplexNumber(6, 7)
    assert number_1 + number_3 == soln.ComplexNumber(3, 11)
    assert number_2 + number_3 == soln.ComplexNumber(7, 14)
    assert number_3 + number_3 == soln.ComplexNumber(4, 18)

def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1 * number_2 == soln.ComplexNumber(-5, 15)
    assert number_1 * number_3 == soln.ComplexNumber(-16, 13)
    assert number_2 * number_3 == soln.ComplexNumber(-35, 55)
    assert number_3 * number_3 == soln.ComplexNumber(-77, 36)

def test_complex_subtraction(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1 - number_2 == soln.ComplexNumber(-4, -3)
    assert number_1 - number_3 == soln.ComplexNumber(-1, -7)
    assert number_2 - number_3 == soln.ComplexNumber(3, -4)
    assert number_3 - number_3 == soln.ComplexNumber(0, 0)

def test_complex_truediv(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1 / number_2 == soln.ComplexNumber(3/10, --1/10)
    assert number_1 / number_3 == soln.ComplexNumber(4/17, -1/17)
    assert number_2 / number_3 == soln.ComplexNumber(11/17, -7/17)
    assert number_3 / number_3 == soln.ComplexNumber(1, 0)

def test_complex_conjugate(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1.conjugate() == soln.ComplexNumber(1, -2)
    assert number_2.conjugate() == soln.ComplexNumber(5, -5)
    assert number_3.conjugate() == soln.ComplexNumber(2, -9)

def test_complex_norm(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums()
    assert number_1.norm() == (1**2 + 2**2)**0.5
    assert number_2.norm() == (5**2 + 5**2)**0.5
    assert number_3.norm() == (2**2 + 9**2)**0.5

# Problem 4: Write test cases for the Set game.
