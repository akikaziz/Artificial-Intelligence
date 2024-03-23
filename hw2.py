import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=[0]*26

    Letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    with open (filename,encoding='utf-8') as f:
       for line in f:
        for char in line:
            if (set(char.upper()).issubset(Letters)):
                char = char.upper()
                ascii_value = ord(char)
                i = ascii_value-65
                X[i] += 1

            else:
                continue
    
    print("Q1")

    for i in range(0,26):
        ascii_char = chr(i + 65)

        print(ascii_char + " " + str(X[i]))
    
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def q2(letters, probabilities):
    print("Q2")

    val1 = letters[0] * math.log(probabilities[0][0])
    val2 = letters[0] * math.log(probabilities[1][0])
    print("%.4f" % val1)
    print("%.4f" % val2)

def q3(letters, probabilities):
    print("Q3")

    f_English = math.log(probabilityOfEnglish)

    for i in range(0,26):
        t = letters[i]*math.log(probabilities[0][i])

        f_English += t
    print("%.4f" % f_English)

    f_Spanish = math.log(probabilityOfSpanish)

    for i in range(0,26):
        t = letters[i]*math.log(probabilities[1][i])

        f_Spanish += t
    print("%.4f" % f_Spanish)

    return(f_English, f_Spanish)

def q4(F_Vals):
    print("Q4")

    val = 0

    if(F_Vals[1] - F_Vals[0] >= 100):
        val = 0
    elif(F_Vals[1] - F_Vals[0] <= -100):
        val = 1
    else:
        val = 1 / (1 + math.exp(F_Vals[1] - F_Vals[0]))

    print("%.4f" % val)


letters = shred("letter.txt")

probabilities = get_parameter_vectors()

probabilityOfEnglish = 0.6
probabilityOfSpanish = 0.4

q2(letters, probabilities)

F_Vals= q3(letters, probabilities)

q4(F_Vals)