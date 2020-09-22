### Sample for calculating matrix determinant ###
matrix = [[1,2,3],[1,0,1],[2,1,3]]

def del_row_col(a, row, col):
    a_ = []
    for r in range(len(a)):
        r_ = []
        if(r == row):
            continue
        else:
            for c in range(len(a[r])):
                if(c == col):
                    continue
                else:
                    r_.append(a[r][c])
    
        a_.append(r_)

    print(a_)
    return a_

def det(a):
    if(len(a) == 2):
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    else:
        determinant = 0
        for i in range(len(a)):
            determinant += ((-1) ** i) * a[0][i] * det(del_row_col(a, 0, i))

        return determinant

print(det(matrix))
