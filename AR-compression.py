import numpy as np
import matplotlib.pyplot as plt


def YuleWalkerACoefs (x, window):
    r = []
    r.append (np.mean (x[0:] * x[0:]))
    for i in range (1, window+1):
        r.append (np.mean (x[0:-i] * x[i:]))

    R = np.transpose (r[1:])

    FI = ConstructMatrixFi (window, r)

    aCoefs = np.linalg.inv(FI) * R

    return aCoefs

def YuleWalkerACoefsInverse (x, aCoefs, window):
    xhad = [0 for i in range (1, len(x)+1)]

    for m in range (window, len(x)):
        a_temp = np.array ([])
        a_2 = 0
        for j in range (window):
            a_temp = (aCoefs[j] * x[m - j - 1])
            a_2 += a_temp
        xhad[m] = np.sum (a_2)

    return xhad

def ConstructMatrixFi (n, r):
    FI = np.zeros ((n, n))
    for i in range (n):
        for j in range (n):
            distance = abs (i - j)
            FI[i, j] = r[distance]
    return FI

def ClosestSquare (array):
    a = np.floor (np.sqrt (array.shape[0]))
    while array.shape[0] % a:
        a -= 1
    return (int(a), int(array.shape[0]/a))

def PrintImageArray (array):
    print ("shape: ", array.reshape (ClosestSquare (array)).shape)
    plt.imshow (array.reshape (ClosestSquare (array)))
    plt.colorbar ()
    plt.show ()

def PrintArray (array):
    plt.plot (array)
    plt.show ()



# n = np.arange (0, 100)
n = np.linspace (0, 4*np.pi, 200)

# x = (.5)**n + (.25)**n
# x = n**2
x = np.sin (n * 4)


window = 4

xhad = YuleWalkerACoefsInverse (x, YuleWalkerACoefs (x, window), window)

plt.plot (x[window:])
plt.plot (xhad[window:])
plt.show ()