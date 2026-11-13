import numpy as np 
from operator import itemgetter

# chcemy zliczac zera pojawiające się na początku macierzy 
def rows_permutations(A):
    """permutuje wiersze tak żeby eliminacja gaussa działała gdy są zera na 1. miejscach. 
        zwraca: 
                new_order - kolejność wierszy po jej zamianie (lista), 
                old_order - permutację odwrotną(lista) """
    zero_ocurrences = []
    rows, columns = A.shape 
    for i in range(rows):
        zero_count = 0 
        for j in range(columns):

            if np.isclose(A[i,j], 0, atol=1e-10):
                zero_count +=1  
            else:
                break 
        zero_ocurrences.append([zero_count, i])

    zero_ocurrences = sorted(zero_ocurrences, key= lambda x: x[0])
    new_order = list(map(itemgetter(1), zero_ocurrences))
    old_order = np.argsort(new_order)

    return new_order, old_order

def gauss_solver(matrix,end_matrix):
    # zmieniamy kopię macierzy

    matrix_1 = matrix.copy()
    end_matrix_1 = end_matrix.copy()

    for i in range(matrix_1.shape[0]):

        sub_matrix = matrix_1[i:, :]
        new_order,_ = rows_permutations(sub_matrix)
        full_order = [i + idx for idx in new_order]

        matrix_1[i:] = matrix_1[full_order]
        end_matrix_1[i:] = end_matrix_1[full_order]

        other_rows=list(range(i))
        other_rows.extend(list(range(i+1,matrix_1.shape[0])))

        div=1/matrix_1[i,i]
        matrix_1[i]=matrix_1[i]*div
        end_matrix_1[i]=end_matrix_1[i]*(div)
        
        for j in other_rows:
            div=matrix_1[j,i]
            matrix_1[j]=matrix_1[j]-(div*matrix_1[i])
            end_matrix_1[j]=end_matrix_1[j]-(div*end_matrix_1[i])
    return end_matrix_1
