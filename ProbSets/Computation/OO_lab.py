#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:13:13 2017

@author: luxihan
"""

class Backpack():
    def __init__(self, name, color, size = 5):
        self.name = name
        self.color = color
        self.max_size = size
        self.contents = []
        
    def put(self, item):
        if len(self.contents) < self.max_size:
            self.contents.append(item)
            
        else:
            print('No Room!')
    
            
    def dump(self):
        self.contents = []
        
    def take(self, item):
        self.contents.remove(item)
        
    def __add__(self, other):
        return len(self.contents) + len(other.contents)
    
    def __lt__(self, other):
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        return self.name == other.name and self.color == other.color and \
        len(self.contetns) == len(other.contents)
    
    def __str__(self):
        doc_str = 'Owner:\t\t{}\nColor:\t{}\nSize:\t\t{}\nMax Size:\t\t{}\nContents:\t'.format(\
                           self.name, self.color, len(self.contents), self.max_size,\
                                                     self.contents)
        return doc_str
        
class Knackpack(Backpack):
    def __init__(self, name, color, max_size = 3):
        Backpack.__init__(name, color, max_size)
        self.closed = True
    
    def put(self, item):
        if self.closed:
            print("I'm closed!")
        elif len(self.contents >= self.max_size):
            print('No room!')
        else:
            self.contents.append(item)
            
    def take(self, item):
        if self.closed:
            print("I'm closed!")
        else:
            self.contents.remove(item)
            
    def fly(self, fuel_burn):
        if self.a_fuel < fuel_burn:
            print('Not enough fuel!')
        else:
            self.a_fuel -= fuel_burn
    
    def dump(self):
        self.contents = []
        self.a_fuel = 0
    
            
class Jetpack(Backpack):
    def __init__(self, name, color, max_size = 2, a_fuel = 10):
        Backpack.__init__(name, color, max_size)
        self.a_fuel = a_fuel

def testbackpack():
    testpack = Backpack('Barry', 'black')
    if testpack.max_size != 5:
        print('Wrong Default max_size')
    test_content = ['a', 'b', 'c', 'd', 'e', 'f']
    for item in ['a', 'b', 'c', 'd', 'e', 'f']:
        Backpack.put(item)
        if len(Backpack.contents > Backpack.max_size):
            print('max_size is not working')
    print(testpack.contents)
    if testpack.contents != test_content[:5]:
        print('Put in wrong items')
        
        
class ComplexNumber():
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
     
    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)
                             
    def __abs__(self):
        absolute = (self.real**2 + self.imag**2)**0.5
        return absolute
    
    def __lt__(self, other):
        return abs(self) < abs(other)
    
    def __gt__(self, other):
        return abs(self) > abs(other)
    
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag
    
    def __ne__(self, other):
        return not (self.real == other.real and self.imag == other.imag)
    
    def __add__(self, other):
        #define addtiion between complex number and int/float as well
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        else:
            other_new = ComplexNumber(other, 0)
            return ComplexNumber(self.real + other_new.real, \
                                 self.imag + other_new.imag)
    
    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        else:
            other_new = ComplexNumber(other, 0)
            return ComplexNumber(self.real - other_new.real, \
                                 self.imag - other_new.imag)
        
    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexNumber(real, imag)
        else:
            other_new = ComplexNumber(other, 0)
            real = self.real * other_new.real - self.imag * other_new.imag
            imag = self.real * other_new.imag + self.imag * other_new.real
            return ComplexNumber(real, imag)
            
    
    def __div__(self, other):
        if isinstance(other, ComplexNumber):
            denom = other * other.conjugate()
            num = self * other.conjugate()
            rv = ComplexNumber(num.real / denom, num.imag / denom)
            return rv
        else:
            other_new = ComplexNumber(other, 0)
            denom = other_new * other_new.conjugate()
            num = self * other_new.conjugate()
            rv = ComplexNumber(num.real / denom, num.imag / denom)
            return rv
        
    def __str__(self):
        doc_str = 'real:\t{}\nimag:\t{}'.format(self.real, self.imag)
        return doc_str
            
    
    
    
    
    
    
    
    
    
