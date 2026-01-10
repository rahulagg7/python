"""
================================================================================
COMPREHENSIVE PYTHON MASTERY GUIDE FOR AI/ML/DS ENGINEERS
================================================================================
This file contains everything you need to master Python from scratch to advanced
level, specifically tailored for AI/ML/Data Science roles ($125k+ positions).

Structure: Progressive learning from basics → intermediate → advanced → AI/ML/DS
================================================================================
"""

# ============================================================================
# PART 1: PYTHON FUNDAMENTALS
# ============================================================================

# ----------------------------------------------------------------------------
# 1.1 VARIABLES AND DATA TYPES
# ----------------------------------------------------------------------------

# Numbers
integer_var = 42
float_var = 3.14
complex_var = 3 + 4j

# Strings
string_var = "Hello, Python!"
multiline_string = """This is a
multiline string"""
f_string = f"Value: {integer_var}"  # f-strings (Python 3.6+)
formatted = "Value: {}".format(integer_var)

# Boolean
is_true = True
is_false = False

# None (null equivalent)
none_var = None

# Type checking and conversion
print(type(integer_var))  # <class 'int'>
print(isinstance(integer_var, int))  # True
converted = str(integer_var)  # "42"
converted_back = int(converted)  # 42


# ----------------------------------------------------------------------------
# 1.2 OPERATORS
# ----------------------------------------------------------------------------

# Arithmetic
a, b = 10, 3
print(a + b)  # 13
print(a - b)  # 7
print(a * b)  # 30
print(a / b)  # 3.333...
print(a // b)  # 3 (floor division)
print(a % b)  # 1 (modulo)
print(a ** b)  # 1000 (exponentiation)

# Comparison
print(a == b)  # False
print(a != b)  # True
print(a > b)   # True
print(a < b)   # False
print(a >= b)  # True
print(a <= b)  # False

# Logical
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# Identity and membership
x, y = [1, 2], [1, 2]
print(x is y)  # False (different objects)
print(x == y)  # True (same values)
print(1 in x)  # True


# ----------------------------------------------------------------------------
# 1.3 DATA STRUCTURES
# ----------------------------------------------------------------------------

# Lists (mutable, ordered)
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # [1, 2, 3, 4, 5, 6]
my_list.insert(0, 0)  # [0, 1, 2, 3, 4, 5, 6]
my_list.remove(0)  # [1, 2, 3, 4, 5, 6]
popped = my_list.pop()  # 6, list becomes [1, 2, 3, 4, 5]
sliced = my_list[1:4]  # [2, 3, 4]
reversed_list = my_list[::-1]  # [5, 4, 3, 2, 1]

# List methods
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()  # [1, 1, 2, 3, 4, 5, 6, 9]
sorted_copy = sorted(numbers, reverse=True)  # [9, 6, 5, 4, 3, 2, 1, 1]
print(numbers.count(1))  # 2
print(numbers.index(4))  # 3

# Tuples (immutable, ordered)
my_tuple = (1, 2, 3)
single_item = (42,)  # Note the comma!
unpacked = my_tuple  # Can unpack: a, b, c = my_tuple

# Dictionaries (mutable, key-value pairs)
my_dict = {"name": "Python", "version": 3.11, "type": "language"}
my_dict["creator"] = "Guido van Rossum"
value = my_dict.get("name", "default")  # "Python"
keys = list(my_dict.keys())  # ['name', 'version', 'type', 'creator']
values = list(my_dict.values())
items = list(my_dict.items())  # [('name', 'Python'), ...]

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Sets (mutable, unordered, unique elements)
my_set = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
my_set.add(5)
my_set.remove(2)
set1, set2 = {1, 2, 3}, {3, 4, 5}
union = set1 | set2  # {1, 2, 3, 4, 5}
intersection = set1 & set2  # {3}
difference = set1 - set2  # {1, 2}

# Set comprehension
evens = {x for x in range(10) if x % 2 == 0}  # {0, 2, 4, 6, 8}


# ----------------------------------------------------------------------------
# 1.4 CONTROL FLOW
# ----------------------------------------------------------------------------

# If-elif-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator
status = "pass" if score >= 70 else "fail"
1
# For loops
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for item in [1, 2, 3]:
    print(item)

# Enumerate (get index and value)
for idx, value in enumerate(['a', 'b', 'c']):
    print(f"{idx}: {value}")

# Zip (iterate multiple sequences)
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age}")

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break  # Stop at 7
    print(i)


# ----------------------------------------------------------------------------
# 1.5 FUNCTIONS
# ----------------------------------------------------------------------------

# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default arguments
def power(base, exponent=2):
    return base ** exponent

# Function with *args (variable positional arguments)
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# Function with **kwargs (variable keyword arguments)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Python", version=3.11)

# Lambda functions (anonymous functions)
square = lambda x: x ** 2
add = lambda x, y: x + y

# Lambda with map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)  # 15

# Type hints (Python 3.5+)
def calculate_total(items: list[float], tax: float = 0.1) -> float:
    subtotal = sum(items)
    return subtotal * (1 + tax)


# ============================================================================
# PART 2: INTERMEDIATE PYTHON
# ============================================================================

# ----------------------------------------------------------------------------
# 2.1 LIST COMPREHENSIONS AND GENERATOR EXPRESSIONS
# ----------------------------------------------------------------------------

# List comprehension (faster and more Pythonic)
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(20) if x % 2 == 0]  # [0, 2, 4, ..., 18]
nested = [[i*j for j in range(3)] for i in range(3)]  # [[0,0,0], [0,1,2], [0,2,4]]

# Generator expressions (memory efficient)
squares_gen = (x**2 for x in range(10))  # Generator object
print(list(squares_gen))  # Convert to list if needed

# Dictionary comprehension
word_counts = {word: len(word) for word in ['python', 'is', 'great']}
# {'python': 6, 'is': 2, 'great': 5}


# ----------------------------------------------------------------------------
# 2.2 STRING MANIPULATION
# ----------------------------------------------------------------------------

text = "Python Programming"
print(text.upper())  # "PYTHON PROGRAMMING"
print(text.lower())  # "python programming"
print(text.split())  # ['Python', 'Programming']
print(text.replace("Python", "Java"))  # "Java Programming"
print(text.startswith("Python"))  # True
print(text.endswith("ing"))  # True
print(" ".join(["Hello", "World"]))  # "Hello World"
print(text.strip())  # Remove whitespace
print(text.find("Prog"))  # 7

# String formatting (multiple ways)
name, age = "Alice", 30
print(f"{name} is {age} years old")  # f-string (preferred)
print("{} is {} years old".format(name, age))
print("%s is %d years old" % (name, age))


# ----------------------------------------------------------------------------
# 2.3 FILE I/O
# ----------------------------------------------------------------------------

# Writing to file
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")
    f.write("This is a test file.\n")

# Reading from file
with open("example.txt", "r") as f:
    content = f.read()  # Read entire file
    # OR
    lines = f.readlines()  # Read as list of lines
    # OR
    for line in f:  # Read line by line (memory efficient)
        print(line.strip())

# Reading with encoding
with open("data.json", "r", encoding="utf-8") as f:
    data = f.read()


# ----------------------------------------------------------------------------
# 2.4 ERROR HANDLING
# ----------------------------------------------------------------------------

# Try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred")
finally:
    print("This always executes")

# Raising exceptions
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b

# Custom exceptions
class CustomError(Exception):
    pass

class ValidationError(Exception):
    def __init__(self, message, field):
        self.message = message
        self.field = field
        super().__init__(f"{field}: {message}")


# ----------------------------------------------------------------------------
# 2.5 MODULES AND PACKAGES
# ----------------------------------------------------------------------------

# Importing modules
import math
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import json
import csv

# Using imported modules
print(math.sqrt(16))  # 4.0
print(math.pi)  # 3.14159...
print(os.getcwd())  # Current working directory

# JSON handling (critical for APIs and data)
data = {"name": "Python", "version": 3.11}
json_string = json.dumps(data)  # Convert to JSON string
parsed_data = json.loads(json_string)  # Parse JSON string

# Working with JSON files
with open("data.json", "w") as f:
    json.dump(data, f)  # Write to file

with open("data.json", "r") as f:
    loaded_data = json.load(f)  # Read from file


# ============================================================================
# PART 3: OBJECT-ORIENTED PROGRAMMING
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 CLASSES AND OBJECTS
# ----------------------------------------------------------------------------

class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    # Class method
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = datetime.now().year - birth_year
        return cls(name, age)
    
    # Static method
    @staticmethod
    def is_adult(age):
        return age >= 18
    
    # String representation
    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Creating objects
person1 = Person("Alice", 30)
person2 = Person.from_birth_year("Bob", 1995)
print(person1.introduce())
print(Person.is_adult(25))  # True


# ----------------------------------------------------------------------------
# 3.2 INHERITANCE
# ----------------------------------------------------------------------------

class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement this method")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
print(dog.speak())  # "Buddy says Woof!"


# ----------------------------------------------------------------------------
# 3.3 ENCAPSULATION AND PROPERTIES
# ----------------------------------------------------------------------------

class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance  # Protected (convention)
    
    @property
    def balance(self):
        return self._balance
    
    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
        else:
            raise ValueError("Insufficient funds")


# ----------------------------------------------------------------------------
# 3.4 SPECIAL METHODS (Dunder methods)
# ----------------------------------------------------------------------------

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)
v3 = v1 + v2  # Vector(4, 6)
v4 = v1 * 2   # Vector(6, 8)


# ============================================================================
# PART 4: ADVANCED PYTHON CONCEPTS
# ============================================================================

# ----------------------------------------------------------------------------
# 4.1 DECORATORS
# ----------------------------------------------------------------------------

# Simple decorator
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")


# ----------------------------------------------------------------------------
# 4.2 GENERATORS
# ----------------------------------------------------------------------------

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using generator
for num in fibonacci(10):
    print(num)  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# Generator for reading large files
def read_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()


# ----------------------------------------------------------------------------
# 4.3 ITERATORS
# ----------------------------------------------------------------------------

class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1


# ----------------------------------------------------------------------------
# 4.4 CONTEXT MANAGERS
# ----------------------------------------------------------------------------

# Using with statement (built-in)
with open("file.txt", "r") as f:
    content = f.read()
# File automatically closed

# Custom context manager
class Timer:
    def __init__(self):
        self.start = None
    
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        print(f"Elapsed time: {time.time() - self.start:.4f} seconds")

with Timer():
    # Do something
    time.sleep(1)


# ----------------------------------------------------------------------------
# 4.5 REGULAR EXPRESSIONS
# ----------------------------------------------------------------------------

import re

text = "Contact: email@example.com or phone: 123-456-7890"

# Search
email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
if email_match:
    print(email_match.group())  # email@example.com

# Find all
phone_numbers = re.findall(r'\d{3}-\d{3}-\d{4}', text)
print(phone_numbers)  # ['123-456-7890']

# Replace
masked = re.sub(r'\d', 'X', text)
print(masked)  # Contact: email@example.com or phone: XXX-XXX-XXXX

# Compile pattern (for reuse)
pattern = re.compile(r'\b\w+@\w+\.\w+\b')
matches = pattern.findall(text)


# ----------------------------------------------------------------------------
# 4.6 COLLECTIONS MODULE
# ----------------------------------------------------------------------------

from collections import Counter, defaultdict, deque, namedtuple

# Counter (count occurrences)
counter = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counter)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(counter.most_common(2))  # [('a', 3), ('b', 2)]

# Defaultdict (dict with default values)
dd = defaultdict(int)
dd['key1'] += 1  # No KeyError, defaults to 0
print(dd['key1'])  # 1

# Deque (double-ended queue)
dq = deque([1, 2, 3])
dq.appendleft(0)  # [0, 1, 2, 3]
dq.append(4)      # [0, 1, 2, 3, 4]
dq.popleft()      # 0

# Namedtuple (tuple with named fields)
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)  # 1 2


# ----------------------------------------------------------------------------
# 4.7 DATETIME AND TIME HANDLING
# ----------------------------------------------------------------------------

from datetime import datetime, timedelta, date, time

# Current time
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))  # "2024-01-15 14:30:00"

# Parsing dates
date_str = "2024-01-15"
parsed_date = datetime.strptime(date_str, "%Y-%m-%d")

# Time arithmetic
future = now + timedelta(days=7, hours=3)
difference = future - now
print(difference.days)  # 7

# Timezone (using pytz - install separately)
# from pytz import timezone
# utc = timezone('UTC')
# local = now.astimezone(utc)


# ============================================================================
# PART 5: AI/ML/DS ESSENTIAL LIBRARIES AND PATTERNS
# ============================================================================

# ----------------------------------------------------------------------------
# 5.1 NUMPY BASICS (Numerical Computing)
# ----------------------------------------------------------------------------

import numpy as np

# Creating arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 3))
ones = np.ones((2, 2))
identity = np.eye(3)
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# Array operations
arr = np.array([1, 2, 3, 4, 5])
print(arr * 2)  # [2, 4, 6, 8, 10]
print(arr + 1)  # [2, 3, 4, 5, 6]
print(arr ** 2)  # [1, 4, 9, 16, 25]

# Array indexing and slicing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 1])      # 2
print(matrix[:, 0])      # [1, 4, 7] (first column)
print(matrix[1, :])      # [4, 5, 6] (second row)
print(matrix[0:2, 1:3])  # [[2, 3], [5, 6]]

# Array operations
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
print(np.dot(a, b))      # 32 (dot product)
print(a @ b)             # 32 (matrix multiplication operator)
print(np.sum(a))         # 6
print(np.mean(a))        # 2.0
print(np.std(a))         # 0.816...
print(np.max(a))         # 3
print(np.min(a))         # 1

# Reshaping
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()

# Broadcasting
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3])
print(a + b)  # Broadcasting: [[2, 4, 6], [5, 7, 9]]

# Random numbers
random_arr = np.random.rand(3, 3)  # Uniform [0, 1)
normal_arr = np.random.randn(3, 3)  # Standard normal
integers = np.random.randint(0, 10, size=(3, 3))


# ----------------------------------------------------------------------------
# 5.2 PANDAS BASICS (Data Manipulation)
# ----------------------------------------------------------------------------

import pandas as pd

# Creating DataFrames
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'LA', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)

# Reading data
# df = pd.read_csv('data.csv')
# df = pd.read_excel('data.xlsx')
# df = pd.read_json('data.json')

# Basic operations
print(df.head())        # First 5 rows
print(df.tail())        # Last 5 rows
print(df.info())        # Data types and info
print(df.describe())    # Statistical summary
print(df.shape)         # (rows, columns)
print(df.columns)      # Column names

# Selecting data
print(df['name'])                    # Single column
print(df[['name', 'age']])          # Multiple columns
print(df.iloc[0])                   # First row by index
print(df.iloc[0:2, 1:3])            # Rows 0-1, columns 1-2
print(df.loc[df['age'] > 30])      # Filtering

# Adding/Modifying columns
df['salary'] = [50000, 60000, 70000, 80000]
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Old')

# Grouping and aggregation
grouped = df.groupby('age_group')['age'].mean()
print(grouped)

# Handling missing values
df_missing = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
df_cleaned = df_missing.dropna()           # Remove rows with NaN
df_filled = df_missing.fillna(0)          # Fill NaN with 0
df_filled_mean = df_missing.fillna(df_missing.mean())  # Fill with mean

# Merging DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})
merged = pd.merge(df1, df2, on='key', how='inner')  # inner, left, right, outer

# Writing data
# df.to_csv('output.csv', index=False)
# df.to_excel('output.xlsx', index=False)


# ----------------------------------------------------------------------------
# 5.3 MATPLOTLIB BASICS (Visualization)
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
# plt.show()  # Uncomment to display
# plt.savefig('plot.png')  # Save figure

# Scatter plot
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
plt.scatter(x_scatter, y_scatter, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
# plt.show()

# Histogram
data_hist = np.random.normal(100, 15, 1000)
plt.hist(data_hist, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
# plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x_scatter, y_scatter)
axes[1, 0].hist(data_hist, bins=30)
axes[1, 1].bar(['A', 'B', 'C'], [3, 7, 2])
# plt.show()


# ----------------------------------------------------------------------------
# 5.4 DATA PROCESSING PATTERNS
# ----------------------------------------------------------------------------

# List of dictionaries to DataFrame
data_list = [
    {'name': 'Alice', 'score': 85},
    {'name': 'Bob', 'score': 90},
    {'name': 'Charlie', 'score': 78}
]
df = pd.DataFrame(data_list)

# Processing with list comprehensions
scores = [item['score'] for item in data_list if item['score'] > 80]

# Data transformation pipeline
def process_data(data):
    # Clean
    cleaned = [d for d in data if d.get('score', 0) > 0]
    # Transform
    transformed = [{'name': d['name'].upper(), 'score': d['score'] * 1.1} 
                   for d in cleaned]
    # Filter
    filtered = [d for d in transformed if d['score'] > 85]
    return filtered


# ----------------------------------------------------------------------------
# 5.5 WORKING WITH APIs
# ----------------------------------------------------------------------------

import requests

# GET request
response = requests.get('https://api.github.com/users/octocat')
if response.status_code == 200:
    data = response.json()
    print(data['name'])

# POST request
payload = {'key': 'value'}
response = requests.post('https://httpbin.org/post', json=payload)
print(response.json())

# With error handling
try:
    response = requests.get('https://api.example.com/data', timeout=5)
    response.raise_for_status()  # Raises exception for bad status codes
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")


# ----------------------------------------------------------------------------
# 5.6 CSV PROCESSING
# ----------------------------------------------------------------------------

import csv

# Reading CSV
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])

# Writing CSV
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', '25', 'NYC'])

# Using pandas (easier)
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)


# ----------------------------------------------------------------------------
# 5.7 FUNCTIONAL PROGRAMMING PATTERNS
# ----------------------------------------------------------------------------

from functools import reduce, partial, lru_cache

# Map, Filter, Reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
sum_all = reduce(lambda x, y: x + y, numbers)

# Partial functions
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # 10

# Memoization (caching)
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# ----------------------------------------------------------------------------
# 5.8 MULTITHREADING AND MULTIPROCESSING (for data processing)
# ----------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# Threading (I/O bound tasks)
def fetch_data(url):
    # Simulate API call
    time.sleep(1)
    return f"Data from {url}"

urls = ['url1', 'url2', 'url3', 'url4']
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_data, urls))

# Multiprocessing (CPU bound tasks)
def cpu_intensive_task(n):
    return sum(i*i for i in range(n))

numbers = [1000000, 2000000, 3000000]
with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(cpu_intensive_task, numbers))


# ----------------------------------------------------------------------------
# 5.9 ENVIRONMENT VARIABLES AND CONFIGURATION
# ----------------------------------------------------------------------------

import os
from dotenv import load_dotenv  # pip install python-dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('API_KEY', 'default_value')
database_url = os.environ.get('DATABASE_URL')


# ----------------------------------------------------------------------------
# 5.10 TYPE HINTS FOR BETTER CODE (Python 3.5+)
# ----------------------------------------------------------------------------

from typing import List, Dict, Tuple, Optional, Union, Callable

def process_data(
    items: List[Dict[str, Union[int, str]]],
    threshold: int = 10,
    callback: Optional[Callable] = None
) -> Tuple[List[Dict], int]:
    filtered = [item for item in items if isinstance(item.get('value'), int)]
    count = len(filtered)
    if callback:
        callback(count)
    return filtered, count


# ----------------------------------------------------------------------------
# 5.11 DATA VALIDATION
# ----------------------------------------------------------------------------

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_data(data: Dict) -> Tuple[bool, List[str]]:
    errors = []
    if 'name' not in data or not data['name']:
        errors.append("Name is required")
    if 'age' in data and (not isinstance(data['age'], int) or data['age'] < 0):
        errors.append("Age must be a positive integer")
    if 'email' in data and not validate_email(data['email']):
        errors.append("Invalid email format")
    return len(errors) == 0, errors


# ============================================================================
# PART 6: BEST PRACTICES AND PATTERNS
# ============================================================================

# ----------------------------------------------------------------------------
# 6.1 CODE ORGANIZATION
# ----------------------------------------------------------------------------

# Use meaningful variable names
# Bad: x, y, z
# Good: user_name, total_score, data_points

# Follow PEP 8 style guide
# - Use 4 spaces for indentation
# - Max line length: 79 characters (or 99 for modern code)
# - Use snake_case for functions and variables
# - Use PascalCase for classes
# - Use UPPER_CASE for constants

# Docstrings
def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary containing mean, median, and standard deviation
        
    Example:
        >>> stats = calculate_statistics([1, 2, 3, 4, 5])
        >>> print(stats['mean'])
        3.0
    """
    mean = sum(data) / len(data)
    sorted_data = sorted(data)
    n = len(sorted_data)
    median = (sorted_data[n//2] + sorted_data[(n-1)//2]) / 2 if n % 2 == 0 else sorted_data[n//2]
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return {'mean': mean, 'median': median, 'std_dev': std_dev}


# ----------------------------------------------------------------------------
# 6.2 EFFICIENT DATA PROCESSING
# ----------------------------------------------------------------------------

# Use vectorized operations (NumPy/Pandas) instead of loops
# Bad (slow):
result = []
for x in range(1000000):
    result.append(x * 2)

# Good (fast):
result = np.arange(1000000) * 2

# Use generators for large datasets
def process_large_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield process_line(line)  # Process one line at a time


# ----------------------------------------------------------------------------
# 6.3 ERROR HANDLING PATTERNS
# ----------------------------------------------------------------------------

def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers, returning None on error."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Warning: Division by zero")
        return None
    except TypeError:
        print("Error: Invalid input types")
        return None


# ----------------------------------------------------------------------------
# 6.4 TESTING PATTERNS (Structure for unit tests)
# ----------------------------------------------------------------------------

# Example test structure (use pytest or unittest)
def test_calculate_statistics():
    data = [1, 2, 3, 4, 5]
    stats = calculate_statistics(data)
    assert stats['mean'] == 3.0
    assert stats['median'] == 3.0


# ============================================================================
# PART 7: COMMON AI/ML/DS WORKFLOWS
# ============================================================================

# ----------------------------------------------------------------------------
# 7.1 DATA LOADING AND PREPROCESSING
# ----------------------------------------------------------------------------

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load data and perform basic preprocessing."""
    # Load
    df = pd.read_csv(filepath)
    
    # Clean
    df = df.dropna()  # Remove missing values
    df = df.drop_duplicates()  # Remove duplicates
    
    # Transform
    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].str.lower().str.strip()
    
    # Filter
    df = df[df['value'] > 0]  # Remove invalid values
    
    return df


# ----------------------------------------------------------------------------
# 7.2 FEATURE ENGINEERING PATTERNS
# ----------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""
    # Numerical transformations
    df['log_value'] = np.log1p(df['value'])  # Log transform
    df['sqrt_value'] = np.sqrt(df['value'])  # Square root
    
    # Categorical encoding
    df = pd.get_dummies(df, columns=['category'], prefix='cat')
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Aggregations
    df['value_mean'] = df.groupby('category')['value'].transform('mean')
    df['value_std'] = df.groupby('category')['value'].transform('std')
    
    return df


# ----------------------------------------------------------------------------
# 7.3 DATA VALIDATION PIPELINE
# ----------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate dataset quality."""
    errors = []
    
    # Check for required columns
    required_cols = ['id', 'value', 'date']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check for empty dataset
    if df.empty:
        errors.append("Dataset is empty")
    
    # Check for duplicates
    if df.duplicated().any():
        errors.append("Dataset contains duplicate rows")
    
    # Check data types
    if 'value' in df.columns and not pd.api.types.is_numeric_dtype(df['value']):
        errors.append("'value' column must be numeric")
    
    return len(errors) == 0, errors


# ============================================================================
# SUMMARY: KEY TAKEAWAYS FOR AI/ML/DS ENGINEERS
# ============================================================================

"""
ESSENTIAL SKILLS CHECKLIST:

✓ Python Fundamentals
  - Variables, data types, operators
  - Control flow (if/else, loops)
  - Functions and lambda expressions
  - List/dict/set comprehensions

✓ Data Structures
  - Lists, tuples, dictionaries, sets
  - When to use each (mutable vs immutable, ordered vs unordered)

✓ Object-Oriented Programming
  - Classes, inheritance, encapsulation
  - Special methods (__init__, __str__, etc.)

✓ Advanced Python
  - Decorators, generators, iterators
  - Context managers (with statements)
  - Error handling (try/except)

✓ Essential Libraries
  - NumPy: Numerical computing, arrays
  - Pandas: Data manipulation, DataFrames
  - Matplotlib: Data visualization
  - Requests: API interactions

✓ Data Processing
  - Reading/writing CSV, JSON, Excel files
  - Data cleaning and preprocessing
  - Feature engineering
  - Data validation

✓ Best Practices
  - Code organization and PEP 8
  - Type hints
  - Docstrings
  - Error handling
  - Efficient data processing (vectorization)

✓ Real-World Patterns
  - API integration
  - Large file processing (generators)
  - Multithreading/multiprocessing
  - Configuration management

NEXT STEPS:
1. Practice with real datasets (Kaggle, UCI ML Repository)
2. Build projects combining these concepts
3. Learn ML libraries: scikit-learn, TensorFlow/PyTorch
4. Master SQL for data extraction
5. Learn cloud platforms (AWS, GCP, Azure)
6. Study statistics and linear algebra
7. Practice system design for ML systems

REMEMBER: 
- Practice is key - code daily
- Build projects to reinforce learning
- Read other people's code (GitHub)
- Contribute to open source
- Stay updated with Python and ML ecosystem
"""

print("\n" + "="*80)
print("CONGRATULATIONS! You've completed the comprehensive Python guide.")
print("Practice these concepts daily and build real projects.")
print("="*80)

