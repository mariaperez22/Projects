import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    distancia = levenshtein_matriz(x, y)
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    secuencia = []
    iteraciones = distancia + (lenX + lenY - 2 * distancia)
    posX,posY = lenX, lenY
    #print(D[posX][posY])
    for k in range(iteraciones):
        if posX > 0 and posY > 0:
            dia= D[posX-1][posY-1]
            bX = D[posX][posY-1]
            bY = D[posX-1][posY]
            minimo = min([bX,bY,dia])
            #si el elemento anterior de la diagonal es el minimo y es igual, las letras coinciden
            if dia == minimo and D[posX][posY] == D[posX-1][posY-1]:
                secuencia.append((str(x[posX-1]), str(y[posY-1])))
                posX, posY = posX-1, posY-1
            else:
                #insercion
                if bX == minimo:
                    secuencia.append(('',str(y[posY-1])))
                    posX,posY = posX, posY-1

                #borrado
                elif bY == minimo:
                    secuencia.append((str(x[posX-1]),''))
                    posX,posY = posX-1, posY
                #cambio
                else:
                    secuencia.append((str(x[posX-1]), str(y[posY-1])))
                    posX,posY = posX-1, posY-1 
    if posX > 0:
        primeros = x[:posX]
        primeros = primeros[::-1]
        for k in primeros:
            tupla = (k,'')
            secuencia.append(tupla)
    secuencia.reverse()

    return distancia,secuencia

def levenshtein_reduccion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    vprev = np.zeros((lenX + 1), dtype=int) # 1xlenX
    for m in range(1, lenX + 1): vprev[m] = vprev[m-1] + 1
    vcurrent = np.zeros((lenX + 1), dtype=int) # 1xlenX
    
    # [i][j] fila x columna
    for j in range(1, lenY + 1):
        vcurrent[0] = vprev[0] + 1
        for i in range(1, lenX + 1):
            vcurrent[i] = min(
                    vprev[i] + 1, # izq.
                    vcurrent[i - 1] + 1, # arriba
                    vprev[i - 1] + (x[i - 1] != y[j - 1]), # diagonal (arriba e izq.)
            )
        vprev, vcurrent = vcurrent, vprev
        
    return vprev[lenX]

def levenshtein(x, y, threshold):
    # # completar versión reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    vprev = np.zeros((lenX + 1), dtype=int) # 1xlenX
    for m in range(1, lenX + 1): vprev[m] = vprev[m-1] + 1
    vcurrent = np.zeros((lenX + 1), dtype=int) # 1xlenX
    
    # [i][j] fila x columna
    for j in range(1, lenY + 1): 
        vcurrent[0] = vprev[0] + 1
        for i in range(1, lenX + 1):
            vcurrent[i] = min(
                    vprev[i] + 1, # izq.
                    vcurrent[i - 1] + 1, # arriba
                    vprev[i - 1] + (x[i - 1] != y[j - 1]), # diagonal (arriba e izq.)
            )
        if min(vcurrent) > threshold:
            return threshold + 1    
        vprev, vcurrent = vcurrent, vprev
    return vprev[lenX]

def levenshtein_cota_optimista(x, y, threshold):
        conteo_exceso_1 = {}  #Exceso para str1
        conteo_exceso_2 = {}  #Exceso para str2

        for char in x:
            if char in conteo_exceso_1:
                conteo_exceso_1[char] += 1
            else:
                conteo_exceso_1[char] = 1

        for char in y:
            if char in conteo_exceso_2:
                conteo_exceso_2[char] += 1
            else:
                conteo_exceso_2[char] = 1

        #Suma de conteos en exceso para cada cadena
        suma_exceso_1 = sum(max(conteo - conteo_exceso_2.get(char, 0), 0) for char, conteo in conteo_exceso_1.items())
        suma_exceso_2 = sum(max(conteo - conteo_exceso_1.get(char, 0), 0) for char, conteo in conteo_exceso_2.items())

        #Calcular la cota optimista
        cota_optimista = max(suma_exceso_1, suma_exceso_2)
        
        #Comprobar si la cota optimista supera el umbral
        if cota_optimista > threshold:
            return threshold + 1  #Threshold superado, devolver umbral + 1
        else:
            #Continuar con el cálculo normal de la distancia de Levenshtein
            return levenshtein(x, y, threshold)

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            if (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]):
                D[i][j] = min(
                    D[i - 1][j] + 1,
                    D[i][j - 1] + 1,
                    D[i-2][j-2] + 1,
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                )
            else:
                D[i][j] = min(
                    D[i - 1][j] + 1,
                    D[i][j - 1] + 1,
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                )
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    distancia = damerau_restricted_matriz(x, y)
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            if (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]):
                D[i][j] = min(
                    D[i - 1][j] + 1,
                    D[i][j - 1] + 1,
                    D[i-2][j-2] + 1,
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                )
            else:
                D[i][j] = min(
                    D[i - 1][j] + 1,
                    D[i][j - 1] + 1,
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
                )
    secuencia = []
    iteraciones = distancia + (lenX + lenY - 2 * distancia)
    posX,posY = lenX, lenY
    #print(D[posX][posY])
    for k in range(iteraciones):
        if posX > 0 and posY > 0:
            dia= D[posX-1][posY-1]
            bX = D[posX][posY-1]
            bY = D[posX-1][posY]
            minimo = min([bX,bY,dia])
            #si el elemento anterior de la diagonal es el minimo y es igual, las letras coinciden
            
            if (posX > 1 and posY > 1) and (x[posX - 2] == y[posY - 1] and x[posX - 1] == y[posY - 2]) and D[posX][posY] == D[posX - 2][posY - 2] + 1:
                secuencia.append((str(x[posX - 2] + str(x[posX - 1])), str(x[posX - 1] + str(x[posX - 2]))))
                posX, posY = posX - 2, posY - 2
            elif dia == minimo and D[posX][posY] == D[posX-1][posY-1]:
                secuencia.append((str(x[posX-1]), str(y[posY-1])))
                posX, posY = posX-1, posY-1
            else:
                #insercion
                if bX == minimo:
                    secuencia.append(('',str(y[posY-1])))
                    posX,posY = posX, posY-1

                #borrado
                elif bY == minimo:
                    secuencia.append((str(x[posX-1]),''))
                    posX,posY = posX-1, posY

                else:
                    secuencia.append((str(x[posX-1]), str(y[posY-1])))
                    posX,posY = posX-1, posY-1
    if posX > 0:
        primeros = x[:posX]
        primeros = primeros[::-1]
        for k in primeros:
            tupla = (k,'')
            secuencia.append(tupla)
    secuencia.reverse()

    return distancia,secuencia
    return 0, []

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    vprev = np.zeros((lenX + 1), dtype=int)
    vcurrent = np.zeros((lenX + 1), dtype=int)
    vprev2 = np.zeros((lenX + 1), dtype=int)

    for m in range(1, lenX + 1):
        vprev[m] = vprev[m-1] + 1
        vprev2[m] = vprev2[m-1] + 1

    for j in range(1, lenY + 1):
        vcurrent[0] = vprev[0] + 1
        
        for i in range(1, lenX + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            
            vcurrent[i] = min(
                vprev[i] + 1,                   # Eliminación
                vcurrent[i - 1] + 1,            # Inserción
                vprev[i - 1] + cost,            # Reemplazo
            )
            
            if i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]:
                vcurrent[i] = min(vcurrent[i], vprev2[i - 2] + 1)  # Transposición

            
        if min(vcurrent) > threshold:
            return threshold + 1    
            
        vprev2, vprev, vcurrent = vprev, vcurrent, vprev2
    return vprev[lenX]
    


def damerau_intermediate_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein intermedia con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        
    #Llena la matriz utilizando la distancia de Damerau-Levenshtein intermedia
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            if (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]):
                # Transposición de ab ↔ ba con coste 1
                D[i][j] = min(
                    D[i - 1][j] + 1,                #Eliminación 
                    D[i][j - 1] + 1,                #Inserción
                    D[i-2][j-2] + 1,               #Transposición 
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),  #Reemplazo
                )
            elif (i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]):
                #Sustitución de acb ↔ ba con coste 2
                D[i][j] = min(
                    D[i - 1][j] + 1,                
                    D[i][j - 1] + 1,             
                    D[i-3][j-2] + 2,               
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),  
                )
            elif (i > 1 and j > 2 and y[j-3] == x[i-1] and y[j-1] == x[i-2]):
                #Sustitución de ab ↔ bca con coste 2
                D[i][j] = min(
                    D[i - 1][j] + 1,               
                    D[i][j - 1] + 1,                
                    D[i-2][j-3] + 2,              
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), 
                )
            else:
                #Ninguna operación especial
                D[i][j] = min(
                    D[i - 1][j] + 1,          
                    D[i][j - 1] + 1,             
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), 
                )
    return D[lenX, lenY]  # Devuelve el valor de distancia en la última celda de la matriz

def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        
    #Llena la matriz utilizando la distancia de Damerau-Levenshtein intermedia
    for i in range(1, lenX + 1):
        for j in range(1, lenY + 1):
            if (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]):
                # Transposición de ab ↔ ba con coste 1
                D[i][j] = min(
                    D[i - 1][j] + 1,                #Eliminación 
                    D[i][j - 1] + 1,                #Inserción
                    D[i-2][j-2] + 1,               #Transposición 
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),  #Reemplazo
                )
            elif (i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]):
                #Sustitución de acb ↔ ba con coste 2
                D[i][j] = min(
                    D[i - 1][j] + 1,                
                    D[i][j - 1] + 1,             
                    D[i-3][j-2] + 2,               
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),  
                )
            elif (i > 1 and j > 2 and y[j-3] == x[i-1] and y[j-1] == x[i-2]):
                #Sustitución de ab ↔ bca con coste 2
                D[i][j] = min(
                    D[i - 1][j] + 1,               
                    D[i][j - 1] + 1,                
                    D[i-2][j-3] + 2,              
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), 
                )
            else:
                #Ninguna operación especial
                D[i][j] = min(
                    D[i - 1][j] + 1,          
                    D[i][j - 1] + 1,             
                    D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), 
                )
    matriz = D
    res = []
    i = len(x)
    j = len(y)
    while i > 0 or j > 0:
        if i == 0:
            res.append(('', y[j-1]))
            j -= 1
        elif j == 0:
            res.append((x[i-1], ''))
            i -= 1
        else:
            if matriz[i][j] == matriz[i-1][j]+1:
                res.append((x[i-1], ''))
                i -= 1
            elif matriz[i][j] == matriz[i][j-1]+1:
                res.append(('', y[j-1]))
                j -= 1
            elif (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2] and
                  matriz[i][j] == matriz[i-2][j-2]+1):
                res.append((x[i-2]+x[i-1], y[j-2]+y[j-1]))
                i -= 2
                j -= 2
            elif (i > 2 and j > 1 and x[i-1] == y[j-2] and x[i-3] == y[j-1] and
                  matriz[i][j] == matriz[i-3][j-2]+2):
                res.append((x[i-3]+x[i-2]+x[i-1], y[j-2]+y[j-1]))
                i -= 3
                j -= 2
            elif (i > 1 and j > 2 and x[i-1] == y[j-3] and x[i-2] == y[j-1] and
                  matriz[i][j] == matriz[i-2][j-3]+2):
                res.append((x[i-2]+x[i-1], y[j-3]+y[j-2]+y[j-1]))
                i -= 2
                j -= 3
            else:
                res.append((x[i-1], y[j-1]))
                i -= 1
                j -= 1

    res.reverse()

    return D[lenX, lenY], res  # Devuelve el valor de distancia en la última celda de la matriz
    
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    
    # Inicializa los cuatro vectores columna
    v0 = [0] * (lenY + 1)
    v1 = [0] * (lenY + 1)
    v2 = [0] * (lenY + 1)
    v3 = [0] * (lenY + 1)

    #Inicializa el primer vector columna
    for j in range(lenY + 1):
        v0[j] = j

    for i in range(1, lenX + 1):
        #Inicializa el primer elemento del vector actual
        v1[0] = i
        
        for j in range(1, lenY + 1):
            if (i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]):
                # Transposición de ab ↔ ba con coste 1
                transpose_cost = v2[j-2] + 1
                v1[j] = min(
                    v0[j] + 1,              
                    v1[j-1] + 1,              
                    transpose_cost,           
                    v0[j-1] + (x[i-1] != y[j-1])  
                )
            elif (i > 2 and j > 1 and x[i-3] == y[j-1] and x[i-1] == y[j-2]):
                #Sustitución de acb ↔ ba con coste 2
                v1[j] = min(
                    v0[j] + 1,          
                    v1[j-1] + 1,             
                    v3[j-2] + 2,          
                    v0[j-1] + (x[i-1] != y[j-1])  
                )
            elif (i > 1 and j > 2 and y[j-3] == x[i-1] and y[j-1] == x[i-2]):
                #Sustitución de ab ↔ bca con coste 2
                v1[j] = min(
                    v0[j] + 1,                 
                    v1[j-1] + 1,                 
                    v2[j-3] + 2,
                    v0[j-1] + (x[i-1] != y[j-1]) 
                )
            else:
                #Ninguna operación especial
                v1[j] = min(
                    v0[j] + 1,                   
                    v1[j-1] + 1,            
                    v0[j-1] + (x[i-1] != y[j-1])  
                )

        #Intercambia los vectores v0 y v1 para la próxima iteración
        v0, v1 = v1, v0

        # Comprueba el umbral
        if threshold is not None:
            min_dist = min(v0[1:])
            if min_dist > threshold:
                return threshold + 1

    return v0[lenY] 

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

