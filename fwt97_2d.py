'''
2D CDF 9/7 Wavelet Forward and Inverse Transform (lifting implementation)

This code is provided "as is" and is given for educational purposes.
2008 - Kris Olhovsky - code.inquiries@olhovsky.com
'''

import numpy as np
        
                           
def fwt97_2d(m, nlevels=1):
    ''' Perform the CDF 9/7 transform on a 2D matrix signal m.
    nlevel is the desired number of times to recursively transform the 
    signal. '''
    
    w = len(m[0])
    h = len(m)
    for i in range(nlevels):
        m = fwt97(m, w, h) # cols
        m = fwt97(m, w, h) # rows
        w = int(w/2)
        h = int(h/2)
    
    return m


def iwt97_2d(m, nlevels=1):
    ''' Inverse CDF 9/7 transform on a 2D matrix signal m.
        nlevels must be the same as the nlevels used to perform the fwt.
    '''
    
    w = len(m[0])
    h = len(m)

    # Find starting size of m:
    for i in range(nlevels-1):
        h = int(h/2)
        w = int(w/2)
        
    for i in range(nlevels):
        m = iwt97(m, w, h) # rows
        m = iwt97(m, w, h) # cols
        h *= 2
        w *= 2
    
    return m


def fwt97(s, width, height):
    ''' Forward Cohen-Daubechies-Feauveau 9 tap / 7 tap wavelet transform   
    performed on all columns of the 2D n*n matrix signal s via lifting.
    The returned result is s, the modified input matrix.
    The highpass and lowpass results are stored on the left half and right
    half of s respectively, after the matrix is transposed. '''
        
    # 9/7 Coefficients:
    a1 = -1.586134342
    a2 = -0.05298011854
    a3 = 0.8829110762
    a4 = 0.4435068522
    
    # Scale coeff:
    k1 = 0.81289306611596146 # 1/1.230174104914
    k2 = 0.61508705245700002 # 1.230174104914/2
    # Another k used by P. Getreuer is 1.1496043988602418
        
    for col in range(width): # Do the 1D transform on all cols:
        ''' Core 1D lifting process in this loop. '''
        ''' Lifting is done on the cols. '''
        
        # Predict 1. y1
        for row in range(1, height-1, 2):
            s[row][col] += a1 * (s[row-1][col] + s[row+1][col])   
        s[height-1][col] += 2 * a1 * s[height-2][col] # Symmetric extension
        
        # Update 1. y0
        for row in range(2, height, 2):
            s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
        s[0][col] +=  2 * a2 * s[1][col] # Symmetric extension
        
        # Predict 2.
        for row in range(1, height-1, 2):
            s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a3 * s[height-2][col]
        
        # Update 2.
        for row in range(2, height, 2):
            s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
        s[0][col] += 2 * a4 * s[1][col]
                
    # de-interleave
    temp_bank = np.zeros_like(s)#[[0]*width for i in range(height)]
    for row in range(height):
        for col in range(width):
            # k1 and k2 scale the vals
            # simultaneously transpose the matrix when deinterleaving
            if row % 2 == 0: # even
                temp_bank[int(col)][int(row/2)] = k1 * s[int(row)][int(col)]
            else:           # odd
                temp_bank[int(col)][int(row/2 + height/2)] = k2 * s[int(row)][int(col)]
                
    # write temp_bank to s:
    for row in range(width):
        for col in range(height):
            s[row][col] = temp_bank[row][col]
                
    return s


def iwt97(s, width, height):
    ''' Inverse CDF 9/7. '''
    
    # 9/7 inverse coefficients:
    a1 = 1.586134342
    a2 = 0.05298011854
    a3 = -0.8829110762
    a4 = -0.4435068522
    
    # Inverse scale coeffs:
    k1 = 1.230174104914
    k2 = 1.6257861322319229
    
    # Interleave:
    temp_bank = [[0]*width for i in range(height)]
    for col in range(int(width/2)):
        for row in range(height):
            # k1 and k2 scale the vals
            # simultaneously transpose the matrix when interleaving
            temp_bank[col * 2][row] = k1 * s[row][col]
            temp_bank[col * 2 + 1][row] = k2 * s[row][int(col + width/2)]
                
    # write temp_bank to s:
    for row in range(width):
        for col in range(height):
            s[row][col] = temp_bank[row][col]

                
    for col in range(width): # Do the 1D transform on all cols:
        ''' Perform the inverse 1D transform. '''
        
        # Inverse update 2.
        for row in range(2, height, 2):
            s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
        s[0][col] += 2 * a4 * s[1][col]
        
        # Inverse predict 2.
        for row in range(1, height-1, 2):
            s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a3 * s[height-2][col]

        # Inverse update 1.
        for row in range(2, height, 2):
            s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
        s[0][col] +=  2 * a2 * s[1][col] # Symmetric extension
        
        # Inverse predict 1.
        for row in range(1, height-1, 2):
            s[row][col] += a1 * (s[row-1][col] + s[row+1][col])   
        s[height-1][col] += 2 * a1 * s[height-2][col] # Symmetric extension
                
    return s


def seq_to_img(m, pix):
    ''' Copy matrix m to pixel buffer pix.
    Assumes m has the same number of rows and cols as pix. '''
    for row in range(len(m)):
        for col in range(len(m[row])):
            pix[col,row] = m[row][col]
