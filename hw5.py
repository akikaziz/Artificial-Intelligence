import sys
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

def main():
    
    filename = sys.argv[1]

    try:
        dataset = pd.read_csv(filename)

        x = dataset['year']
        y = dataset['days']

        plt.plot(x, y)
        plt.title('Year vs Number of Frozen Days')

        plt.xlabel('Year')
        plt.ylabel('Number of Frozen Days')

        plt.savefig("plot.jpg")

        plt.show() #remove for submission

        X = np.column_stack((np.ones_like(x), x))

        print("Q3a:")
        print(X.astype(np.int64))

        Y = np.array(y)

        print("Q3b:")
        print(Y.astype(np.int64))

        Z = np.dot(X.T, X)
        I = np.linalg.inv(Z)

        print("Q3c:")
        print(Z.astype(np.int64))

        print("Q3d:")
        print(I)

        PI = np.dot(I, X.T)

        EstBeta = np.dot(PI, Y)

        print("Q3e:")
        print(PI)

        print("Q3f:")
        print(EstBeta)

        x_test = 2022  

        y_test = EstBeta[0] + EstBeta[1] * x_test

        
        print("Q4: " + str(y_test))

        if EstBeta[1] > 0:
            symbol = '>'
            shortAns = "A positive sign indicates that as the year increases, the number of ice days also increases."

        elif EstBeta[1] < 0:
            symbol = '<'
            shortAns = "A negative sign indicates that as the year increases, the number of ice days decreases."

        else:
            symbol = '=='
            shortAns = "A zero sign indicates that the number of ice days does not change with the year."

        print("Q5a: " + symbol)
        print("Q5b: " + shortAns)
        
        xAsterisk = -EstBeta[0]/EstBeta[1]

        print("Q6a: " + str(xAsterisk))
        print("Q6b: The prediction made that Lake Mendota will no longer freeze by the year " + str(xAsterisk) + " is based on a simple linear regression model assuming a linear relationship between the year and the corresponding number of ice days. This makes sense if a solution to Global Warming isn't fixed, however, it might not be true in reality since several other factors like Global Warming could also affect the number of ice days in a year.")
            
    except FileNotFoundError:
        print("ERROR: File not found")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__=="__main__":
    main()