import numpy as np

def gallery(array, ncols=3, nrows=3, pad=1, pad_value=0):
    
    array = np.pad(array,[[0,0],[1,1],[1,1]],'constant',constant_values=pad_value)

    nindex, height, width = array.shape
    #nrows = int(np.ceil(nindex/ncols))
    
    n_extra = ncols*nrows - nindex
    array = np.concatenate((array,pad_value*np.ones((n_extra,height, width))))
        
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
              
    return result