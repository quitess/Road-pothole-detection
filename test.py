filename = "f1.txt"
def modify_name():
    filename = "f2.txt"
    print("local filename:", filename)
modify_name()
print("global filename:", filename)  