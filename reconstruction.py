import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import DataframeCreator as dfc
import pandas as pd
import os

def divideArrayIntoBatches (array, num_divisions) -> list:
    """
    Divides a 2D array into 16 batches.
    
    Args:
        array: a 2D numpy array.
        num_divisions: The number of divisions to use.

    Returns:
    A list of 16 2D numpy arrays, each of which is a batch of the input array.
    """

    if len (array.shape) != 2:
        raise ValueError ("Input array must be 2D")

    num_rows, num_cols = array.shape

    if num_rows == 0 or num_cols == 0:
        return []
    
    batch_size_rows = (num_rows) // num_divisions
    batch_size_cols = (num_cols) // num_divisions

    batches = []
    for i in range (num_divisions):
        for j in range (num_divisions):
            batch = array [i * batch_size_rows: (i + 1) * batch_size_rows,
                           j * batch_size_cols: (j + 1) * batch_size_cols]
            batches.append (batch)

    return batches

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def calculateACoefs (pic):
    '''
        Calculates the aCoefs using the given 2D numpy array (pic)

        Args:
            pic: a 2D numpy array representing the picture.

        Returns:
            aCoefs: a 1D numpy array containing the calculated aCoefs.
    '''
    import numpy as np
    from numpy import mean

    R22 = R33 = R44 = R55 = mean (mean (pic * pic))
    R12 = R34 = R45 = mean ( mean (pic[: , 1:-1] * pic[: , 2:]))
    R13 = mean (mean (pic[1:-1, 1:-1] * pic[2:, 2:]))
    R14 = R23 = mean (mean (pic[1:-1 , :] * pic[2:, :]))
    R15 = R24 = mean (mean (pic[2:, 1:-1] * pic[1:-1, 2:]))
    R25 = mean (mean (pic[1:-1, 3:] * pic[2:, 1:-2]))
    R35 = mean (mean (pic[:, 3:] * pic[:, 1:-2]))

    Phi = np.array ([[R22, R23, R24, R25],
                    [R23, R22, R34, R35],
                    [R24, R34, R22, R45],
                    [R25, R35, R45, R22]])

    R = np.array ([R12, R13, R14, R15])

    aCoefs = np.linalg.solve (Phi, R)

    return aCoefs

def reconstructPicfromACoefs (aCoefs, pic):
    '''
    Reconstructs the image using the given 2D numpy array (pic) and aCoefs.

    Args:
        aCoefs: a 1D numpy array containing aCoefs.
        pic: a 2D numpy array representing the picture. 

    Returns:
        pic_recons_frompic: a 2D numpy array representing the recontructed image using the original image.

    '''

    pic_recons_fromzero = np.zeros (pic.shape, dtype='double')
    pic_recons_fromzero[:,0] = pic[:,0]
    pic_recons_fromzero[0,:] = pic[0,:]

    for i in range (1 , pic.shape[0]-1):
        for j in range (1, pic.shape[1]-1):
            pic_recons_fromzero [i, j] = aCoefs[0]*pic_recons_fromzero[i-1,j] + aCoefs[1]*pic_recons_fromzero[i-1,j+1] + aCoefs[2]*pic_recons_fromzero[i,j+1] + aCoefs[3]*pic_recons_fromzero[i+1,j+1]

    pic_recons_frompic = np.zeros (pic.shape, dtype='double')

    pic_recons_frompic[1:-1, 1:-1] = aCoefs[0] * pic[:-2, 1:-1] + aCoefs[1] * pic[:-2, 2:] + aCoefs[2] * pic[1:-1, 2:] + aCoefs[3] * pic[2:, 2:]

    return pic_recons_frompic

def optimizeAcoefs (aCoefs, pic, reconstructed_pic, max_iterations=100) :
    
    '''
        Optimizes the given prediction coefficients and their corresponding images and optimizes them using hillclimbing algorithm.
    '''

    def calculateDiff (aCoefs, pic, reconstructed_pic):

        diff = np.sum (np.abs (pic - reconstructPicfromACoefs (aCoefs, pic)))
        return diff
    
    step_size = 0.01
    tolerance = 1e-6

    diff = calculateDiff (aCoefs, pic, reconstructed_pic)

    for i in range (max_iterations):
        new_aCoefs = aCoefs + np.random.normal (0, step_size, size=4)
        new_diff = calculateDiff (new_aCoefs, pic, reconstructed_pic)

        if new_diff < diff:
            aCoefs = new_aCoefs
            diff = new_diff

        if diff < tolerance:
            break
    
    return aCoefs

def mse(img1, img2):
    '''
        Measures the MSE score
    '''
    h, w = img1.shape
    diff = abs (img1 - img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse

def createDiffMatrixFromMatrices (pic, pic_recons):
    return pic - pic_recons

def createHist (pic, pic_address, filename, pic_recons_frompic, pic_recons_frompic_optimized):
    fig = plt.figure (figsize= (10, 8))
    plt.subplot (2,2,1)
    plt.hist (createDiffMatrixFromMatrices(pic, pic_recons_frompic).ravel(), 256, [-127, 127])
    plt.title ('diff')
    plt.grid ()
    plt.subplot (2,2,2)
    plt.hist (createDiffMatrixFromMatrices(pic, pic_recons_frompic).ravel(), 256, [-127, 127], log=True)
    plt.title ('Log diff')
    plt.grid ()

    plt.subplot (2,2,3)
    plt.hist (createDiffMatrixFromMatrices(pic, pic_recons_frompic_optimized).ravel(), 256, [-127, 127])
    plt.title ('diff (optimized)')
    plt.grid ()
    plt.subplot (2,2,4)
    plt.hist (createDiffMatrixFromMatrices(pic, pic_recons_frompic_optimized).ravel(), 256, [-127, 127], log=True)
    plt.title ('Log diff (optimized)')
    plt.grid ()

    plt.figtext (0, 0, 'pic: {}'.format (pic_address))

    fig.savefig ('output/' + filename)

def createExcelFromPics (file_path, filename, dataframe):
    pic_src = cv.imread (file_path)
    pic = cv.cvtColor(pic_src, cv.COLOR_BGR2GRAY) 

    pic = image_resize (pic, height= 512)

    aCoefs = calculateACoefs (pic)
    aCoefs_optimized = optimizeAcoefs (aCoefs, pic, reconstructPicfromACoefs (aCoefs, pic))

    pic_recons_frompic = reconstructPicfromACoefs (aCoefs, pic)
    pic_recons_frompic_optimized = reconstructPicfromACoefs (aCoefs_optimized, pic)

    createHist (pic, file_path, filename, pic_recons_frompic, pic_recons_frompic_optimized)
    dfc.appendRow (dataframe, filename, aCoefs, aCoefs_optimized)

def createBoxplot (data):
    pass

def acoef_str2array (str_representation:str) -> list:
    
    str_values = str_representation [1:-1].split ()
    values = [float(value) for value in str_values]
    array = np.array (values, dtype='double')
    
    return array

def createPicArrayFromPath (file_dir:str, pic_name:str, pic_size:int=512):
    pic_path = file_dir + pic_name

    pic_src = cv.imread (pic_path)
    pic = cv.cvtColor (pic_src, cv.COLOR_BGR2GRAY)

    pic = image_resize (pic, height= 512)

    return pic

def createDiffFromPath (file_dir:str, pic_name:str, acoefs:list):
    pic = createPicArrayFromPath (file_dir, pic_name)

    pic_reconstructed = reconstructPicfromACoefs (acoefs, pic)
    diff = createDiffMatrixFromMatrices (pic, pic_reconstructed)

    return diff

def gaussianPars (diff) -> tuple:
    diff_flat = diff.ravel ()

    hist = np.histogram (diff_flat, bins=256, range=[-127, 127])
    data = hist [0]

    return (np.mean (data), np.std (data))

def appendColumnToDataframe (df:pd.DataFrame, column_name:str, added_data:any = None):
    df[column_name] = added_data

def addGaussianParsToDataFrame (df:pd.DataFrame, files_dir:str):
    if 'mean_value' not in df.columns:
        appendColumnToDataframe (df, 'mean_value')
    if 'std_value' not in df.columns:
        appendColumnToDataframe (df, 'std_value')

    i = 0
    for row in df.itertuples ():
        if i > df.shape [0] + 1:
            break
        pars = gaussianPars (createDiffFromPath (files_dir, row.file_name, acoef_str2array (row.aCoefs_optimized)))
        df.loc [row.Index, 'mean_value'] =  pars [0]
        df.loc [row.Index, 'std_value'] =  pars [1]
        i += 1

    return df

def convertExcel2Dataframe (excel_file_path:str) -> pd.DataFrame:
    df = pd.read_excel (excel_file_path)

    return df