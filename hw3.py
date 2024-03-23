from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!

    x = np.load(filename)
    mean = np.mean(x, axis = 0)
    
    return x - mean

def get_covariance(dataset):
    # Your implementation goes here!

    data_transposed = np.transpose(dataset)
    val = np.dot(data_transposed, dataset)

    res = val*(1/(len(dataset)-1))

    return res

def get_eig(S, m):
    # Your implementation goes here!

    eigenValues, eigenVectors = eigh(S, subset_by_index = [1023-m+1, 1023])

    diagonalMatrix = [[0]*m for i in range(m)]

    i = 0

    for ind in reversed(range(0, len(eigenValues))):
                        diagonalMatrix[i][i] = eigenValues[ind]
                        i+=1
    
    for ind in (range(0, len(eigenVectors))):
          eigenVectors[ind] = eigenVectors[ind] [::-1]
    
    diagonalMatrix = np.array(diagonalMatrix)

    return diagonalMatrix, eigenVectors

def get_eig_prop(S, prop):
    # Your implementation goes here!

    eigenValues, eigenVectors = eigh(S)
    sumEig = sum(eigenValues)

    eigenValues, eigenVectors = eigh(S, subset_by_value = (prop*sumEig, np.inf))

    m = len(eigenValues)

    diagonalMatrix = [[0]*m for i in range(m)]

    i = 0

    for ind in reversed(range(0, len(eigenValues))):
        diagonalMatrix[i][i] = eigenValues[ind]
        i+=1
    
    for ind in (range(0, len(eigenVectors))):
          eigenVectors[ind] = eigenVectors[ind] [::-1]
    
    diagonalMatrix = np.array(diagonalMatrix)

    return diagonalMatrix, eigenVectors

def project_image(image, U):
    # Your implementation goes here!

    U_transpose = np.transpose(U)

    a_ij = np.dot(U_transpose, image)

    for i in range(0, len(a_ij)):

        for j in range(0, len(U)):
               U[j][i]*= a_ij[i]

    x = []

    for k in range(0, len(U)):

        val = 0

        for l in range(0, len(U[k])):
            val += U[k][l]

        x.append(val)

    x = np.array(x)

    return x

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2

    orig = np.reshape(orig, (32,32))
    proj = np.reshape(proj, (32,32))

    orig = np.transpose(orig)
    proj = np.transpose(proj)

    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    
    ax1.set_title('Original')
    ax2.set_title('Projection')

    originalImg = ax1.imshow(orig, aspect='equal')
    projectionImg = ax2.imshow(proj, aspect='equal')

    fig.colorbar(originalImg, ax=ax1)
    fig.colorbar(projectionImg, ax=ax2)

    return fig, ax1, ax2

x = load_and_center_dataset('YaleB_32x32.npy') 
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
fig, (ax1, ax2) = display_image(x[0], projection)
plt.show()