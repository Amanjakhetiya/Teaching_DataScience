
# coding: utf-8

# Python is an interpreter. Expressions and variables can be directly typed in.

# In[1]:

5+9

# no semicolons needed!
# is the comment symbol


# # Whitespace Formatting
# 
# Python uses indentation to delimit blocks of code. No more curly braces!

# In[2]:

for i in [1,2,3,4,5]:
    print(i)
    for j in [6,7,8,9,10]:
        print(j)
        print(i+j)
    print(i)
print('done')


# ## Strings
# 
# Strings can be delimited by either single or double quotation marks.

# In[4]:

x = 'rich is a string'
y = "Richie is still a string"
print(y)


# ### String Functions

# In[11]:

len(x)     # length of a string


# ### Lists
# 
# The list data structure is really useful and cool in Python. Technically, it's an ordered collection. 
# 
# (Arrays with some added functionality.)

# In[6]:

x = [1,2,3]
y = [1,'a',2.3,False]
z = [1,x]      # lists can have different types in them
w = range(50)  # is the list [0,1,...,49]


# In[9]:

# Exploring types
print(type(1))
print(type('a'))
print(type(2.3))
print(type(False))
print(type(z))


# In[13]:

# Index a list using square brackets
# lists are zero-indexed
print( x[1] )
print( y[2] )
print( x )
z[0] = 999
print (z)


# In[14]:

# Slicing lists
# square bracket syntax with : operator
w = range(50)
print(w[10:20]) # everything from item 10 up to, but not including item 20
print(w[35:])   # everything from item 35 to the end of the list
print(w[:46])   # everything from the beginning, up to but not including 46

# negative indicies
print(w[-1])    # last item in the list
w[-3]    # next to, next to last item in the list


# In[16]:

# Checking for membership in a list
# using the **in** operator, returns a boolean

# little weird syntax...
print(1 in [1,2,3])
1 in [4,5,6]


# In[22]:

# Concatenating lists
x = [1,2,3]
x


# In[23]:

y = x + [4,5,6]
y


# In[24]:

z = y * 2
z


# In[25]:

z.append(0)
z


# In[26]:

z.extend([99,'done'])
z


# ### Tuples
# 
# A tuple is immutable. Very similar to a list.
# Uses parentheses instead of the square brackets of a list.

# In[27]:

t = (1,2,3)
w = [1,2,3]


# In[28]:

print (type(t))
print (type(w))


# In[29]:

w[0]


# In[30]:

t[0]


# In[31]:

w[0] = 5


# In[32]:

w


# In[33]:

t[0] = 5


# ### Dictionaries
# 
# Associates keys with values. Curly braces.

# In[34]:

empty_dict = {}
d = {}
d1 = { 'age':50, 'weight':250, 'height':"5'6"}   # initializing a dictionary


# In[35]:

d1.keys()   # list of keys in the dictionary


# In[36]:

d1.values() # list of values


# In[37]:

d1.items()  # list of (key,value) tuples in the dictionary


# In[39]:

# checking for existance of a key
print ('age' in d1)
print ('marital_status' in d1)


# In[40]:

# getting a value from a dict using the key
d1['age']


# In[42]:

# assignment
d1['age'] = d1['age']+1
d1['age']


# ### If Statements
# 
# elif instead of else if
# Also note the :, no parentheses, and the tabbing.

# In[45]:

avg = 95
if avg == 100:
    print ("super, you get an A")
elif avg > 90:
    print ("a-")
elif avg >= 80:
    print ("BBBBBB")
else:
    print ("you fail")


# ## List Comprehensions
# 
# Really neat.
# Can transform a list into another list, or only select certain elements.

# In[47]:

evens = [x for x in range(5) if x % 2 == 0]
squares = [x * x for x in range(5)]
even_squares = [x * x for x in evens]

# naming convensions in python, note use of _ for spaces

print(evens)
print(squares)
print(even_squares)


# In[49]:

# nested loops permitted in list comprehensions

pairs = [(x,y)
        for x in range (5)
        for y in range (3)]
pairs


# ### Random Number Generation
# 
# Need to first import the random library.
# Produces deterministic numbers (if you want) for reproducible results.

# In[55]:

import random

y = [random.random() for x in range(3)]
print(y)

y = [random.random() for x in range(3)]
print(y)

    
# now reproducible
random.seed(5)
y = [random.random() for x in range(3)]
print(y)

random.seed(5)
y = [random.random() for x in range(3)]
print(y)  


# ## Functions

# In[56]:

def double(x):
    """documentation about the function
    this function multiplies its input by 2"""
    x = x * 2
    return x

# the triple quoted string can be extended over multiple lines!   


# In[57]:

y = [1,2,3]
print(double(y))


# In[58]:

# multiple parameters
def another_method(x,y):
    x = x + 5
    y += 3
    return (y,x)   # can return multiple values


# In[59]:

a,b = another_method(10,100)
print (a)
print (a,b)  # printing multiple values on same line


# In[61]:

# optional arguments
def one_final_method(x,y,z=100):  # z has a default value if it is unspecified
    x = y+z
    return z


# In[62]:

print (one_final_method(3,4,5))
print (one_final_method(3,4))


# ## Object-Oriented Programming
# 
# `__init__` is the constructor (note the double underscore)
# 
# Note the static variable (one per class).
# Note the instance variables (one per instantiated object).
# 
# Note the use of `self` to refer to the instance.

# In[90]:

class Employee:
    'Common base class for all employees'
    empCount = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def display_count(self):
        print ("Total Employee ", Employee.empCount)

    def display_employee(self):
        print ("Name : ", self.name,  ", Salary: ", self.salary)
        
    def modify_salary(self, newsalary):
        self.salary = newsalary


# In[91]:

worker1 = Employee('burns', 90000)
worker2 = Employee('wyatt', 60000)

print(Employee.empCount)
worker1.display_count()
worker2.display_employee()

worker1.modify_salary(100000)

worker1.display_employee()
print(worker1.name)


# # The Biggest Topics to Know (I think)
# 
# * Assignments
# * If Statements
# * Loops
# * Lists
# * Tuples
# * Dictionaries
# * List Comprehensions
# * Random Number Generation
# * Functions
# * OO
# 
# Read Chapter 2 of the Grus book and look at the official Python tutorial.
# 
# ### Other topics
# 
# * Regular Expressions
# * While Loops
# * break, continue statements
# * Inheritance
# * ...
# 
# 
# 
