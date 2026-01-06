# This is my first python program

print("I Like Pizza")
# variables
full_name = "Rahul Aggarwal"
age = 26

gpa = 3.5
is_student = False
print(f"Hello {full_name}")
print(f"my age is {age}")
print(f"my cgpa is {gpa}")
print(f"am i a student?: {is_student}")

if is_student:
    print("I am a student")
else:
    print("I am not a student")

friends = 5
# friends += 1
# friends -= 2
# friends *= 2
# friends /= 2
# friends %= 2
print(friends)

# type casting str(), int(), float(), bool()
# name = "Rahulagg"
name = ""
age = 26
gpa = 3.8
is_student = True

# age = str(age)
# print(type(age))
name = bool(name)
print(name)

# accept user input
name = input("what is your name?")
print(f"Hello {name}")

#logical operators and, or, not
temp = 25
if temp > 30 and temp < 50:
    print("it is warm")
elif temp < 0:
    print("it is cold")
else:
    print("it is ok")