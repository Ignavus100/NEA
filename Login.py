from DatabaseAccess import *

def hash(password):
    final = 0
    for i in range(len(password)):
        final += ord(password[i]) * (11**i)
    return final