import numpy as np
        
                           
def fwt97_2d(m, nlevels=1):
    
    w = len(m[0])
    h = len(m)
    for i in range(nlevels):
        m = fwt97(m, w, h)
        m = fwt97(m, w, h) 
        w = int(w/2)
        h = int(h/2)
    
    return m


def iwt97_2d(m, nlevels=1):
    
    w = len(m[0])
    h = len(m)

    for i in range(nlevels-1):
        h = int(h/2)
        w = int(w/2)
        
    for i in range(nlevels):
        m = iwt97(m, w, h) 
        m = iwt97(m, w, h) 
        h *= 2
        w *= 2
    
    return m


def fwt97(s, width, height):
        
    # 9/7 Coefficients:
    a1 = -1.586134342
    a2 = -0.05298011854
    a3 = 0.8829110762
    a4 = 0.4435068522
    
    # Scale coeff:
    k1 = 0.81289306611596146 
    k2 = 0.61508705245700002 
        
    for col in range(width): 

        for row in range(1, height-1, 2):
            s[row][col] += a1 * (s[row-1][col] + s[row+1][col])   
        s[height-1][col] += 2 * a1 * s[height-2][col] 

        for row in range(2, height, 2):
            s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
        s[0][col] +=  2 * a2 * s[1][col] 
        

        for row in range(1, height-1, 2):
            s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a3 * s[height-2][col]
        
        for row in range(2, height, 2):
            s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
        s[0][col] += 2 * a4 * s[1][col]
                
    temp_bank = np.zeros_like(s)
    for row in range(height):
        for col in range(width):
            if row % 2 == 0: 
                temp_bank[int(col)][int(row/2)] = k1 * s[int(row)][int(col)]
            else:      
                temp_bank[int(col)][int(row/2 + height/2)] = k2 * s[int(row)][int(col)]
                
    for row in range(width):
        for col in range(height):
            s[row][col] = temp_bank[row][col]
                
    return s


def iwt97(s, width, height):

    # 9/7 inverse coefficients:
    a1 = 1.586134342
    a2 = 0.05298011854
    a3 = -0.8829110762
    a4 = -0.4435068522
    
    # Inverse scale coeffs:
    k1 = 1.230174104914
    k2 = 1.6257861322319229

    temp_bank = [[0]*width for i in range(height)]
    for col in range(int(width/2)):
        for row in range(height):

            temp_bank[col * 2][row] = k1 * s[row][col]
            temp_bank[col * 2 + 1][row] = k2 * s[row][int(col + width/2)]

    for row in range(width):
        for col in range(height):
            s[row][col] = temp_bank[row][col]

                
    for col in range(width): 
        
        for row in range(2, height, 2):
            s[row][col] += a4 * (s[row-1][col] + s[row+1][col])
        s[0][col] += 2 * a4 * s[1][col]

        for row in range(1, height-1, 2):
            s[row][col] += a3 * (s[row-1][col] + s[row+1][col])
        s[height-1][col] += 2 * a3 * s[height-2][col]

        for row in range(2, height, 2):
            s[row][col] += a2 * (s[row-1][col] + s[row+1][col])
        s[0][col] +=  2 * a2 * s[1][col] 
        
        for row in range(1, height-1, 2):
            s[row][col] += a1 * (s[row-1][col] + s[row+1][col])   
        s[height-1][col] += 2 * a1 * s[height-2][col] 
                
    return s
