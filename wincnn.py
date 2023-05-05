from __future__ import print_function
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint
from operator import mul
from functools import reduce
from pathlib import Path


def At(a,m,n):
    return Matrix(m, n, lambda i,j: a[i]**j)

def A(a,m,n):
    return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def T(a,n):
    return Matrix(Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i,j: -a[i]**n)))

def Lx(a,n):
    x=symbols('x')
    return Matrix(n, 1, lambda i,j: Poly((reduce(mul, ((x-a[k] if k!=i else 1) for k in range(0,n)), 1)).expand(basic=True), x))

def F(a,n):
    return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))

def Fdiag(a,n):
    f=F(a,n)
    return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))

def FdiagPlus1(a,n):
    f = Fdiag(a,n-1)
    f = f.col_insert(n-1, zeros(n-1,1))
    f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0)))
    return f

def L(a,n):
    lx = Lx(a,n)
    f = F(a, n)
    return Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T

def Bt(a,n):
    return L(a,n)*T(a,n)

def B(a,n):
    return Bt(a,n-1).row_insert(n-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

FractionsInG=0
FractionsInA=1
FractionsInB=2
FractionsInF=3

def cookToomFilter(a,n,r,fractionsIn=FractionsInG):
    alpha = n+r-1
    f = FdiagPlus1(a,alpha)
    if f[0,0] < 0:
        f[0,:] *= -1
    if fractionsIn == FractionsInG:
        AT = A(a,alpha,n).T
        G = (A(a,alpha,r).T*f**(-1)).T
        BT = f * B(a,alpha).T
    elif fractionsIn == FractionsInA:
        BT = f * B(a,alpha).T
        G = A(a,alpha,r)
        AT = (A(a,alpha,n)).T*f**(-1)
    elif fractionsIn == FractionsInB:
        AT = A(a,alpha,n).T
        G = A(a,alpha,r)
        BT = B(a,alpha).T
    else:
        AT = A(a,alpha,n).T
        G = A(a,alpha,r)
        BT = f * B(a,alpha).T
    return (AT,G,BT,f)


def filterVerify(n, r, AT, G, BT):

    alpha = n+r-1

    di = IndexedBase('d')
    gi = IndexedBase('g')
    d = Matrix(alpha, 1, lambda i,j: di[i])
    g = Matrix(r, 1, lambda i,j: gi[i])

    V = BT*d
    U = G*g
    M = U.multiply_elementwise(V)
    Y = simplify(AT*M)

    return Y

def convolutionVerify(n, r, B, G, A):

    di = IndexedBase('d')
    gi = IndexedBase('g')

    d = Matrix(n, 1, lambda i,j: di[i])
    g = Matrix(r, 1, lambda i,j: gi[i])

    V = A*d
    U = G*g
    M = U.multiply_elementwise(V)
    Y = simplify(B*M)

    return Y

def showCookToomFilter(a,n,r,fractionsIn=FractionsInG):

    AT,G,BT,f = cookToomFilter(a,n,r,fractionsIn)

    print ("AT = ")
    pprint(AT)
    print ("")

    print ("G = ")
    pprint(G)
    print ("")

    print ("BT = ")
    pprint(BT)
    print ("")

    if fractionsIn != FractionsInF:
        print ("FIR filter: AT*((G*g)(BT*d)) =")
        pprint(filterVerify(n,r,AT,G,BT))
        print ("")

    if fractionsIn == FractionsInF:
        print ("fractions = ")
        pprint(f)
        print ("")

def showCookToomConvolution(a,n,r,fractionsIn=FractionsInG):

    AT,G,BT,f = cookToomFilter(a,n,r,fractionsIn)

    B = BT.transpose()
    A = AT.transpose()

    print ("A = ")
    pprint(A)
    print ("")

    print ("G = ")
    pprint(G)
    print ("")

    print ("B = ")
    pprint(B)
    print ("")

    if fractionsIn != FractionsInF:
        print ("Linear Convolution: B*((G*g)(A*d)) =")
        pprint(convolutionVerify(n,r,B,G,A))
        print ("")

    if fractionsIn == FractionsInF:
        print ("fractions = ")
        pprint(f)
        print ("")

def cRow(m):
    n = len(m)
    s = "\t(const float [%d]){"%(n)
    for i in range(len(m)):
        v = m[i]
        row_string = str(v) + ".f"
        if(i != (len(m) - 1)):
            row_string +=","
        s += "{:>10}".format(row_string)
    s += "},\n"
    return s

def cMatrix(m,name):
    s = "static const float * %s[]={\n"%(name)
    for i in range(m.shape[0]):
        s+= cRow(m[i,:])
    s+= "};\n\n"
    return s

def generateDefinition(a,n,r):
    AT,G,BT,f = cookToomFilter(a,n,r,0)
    dim = "%d_%d"%(n,r)
    s = "/* Winograd Transformation Matrices for F(%d,%d) */\n" %(n,r)
    s += cMatrix(AT,"At" + dim)
    s += cMatrix(G,"G" + dim)
    s += cMatrix(BT,"Bt" + dim)
    s += "WINOGRAD_STRUCT Wino_F%s={G%s,Bt%s,At%s,%d,%d,%d,%d};\n\n"%(
            dim,dim,dim,dim,r + n - 1,n,r,n)
    return s


def generateCHeader(l,outfile):
    fname = Path(outfile).stem.upper()
    s = "#ifndef %s_H\n#define %s_H 0\n\n"%(fname,fname)
    # Define the Structure
    s += ("typedef const struct WSTRUCT{ \n"
        "\tconst float ** G;\n"   
        "\tconst float ** Bt;\n" 
        "\tconst float ** At;\n" 
        "\tconst int tile_size;\n"
        "\tconst int tile_stride;\n"
        "\tconst int kernel_size;\n"
        "\tconst int out_size;\n"
        "}WINOGRAD_STRUCT;\n\n\n")
    for e in l:
        s += generateDefinition(e[0],e[1],e[2])
    s += "\n\n#endif"
    f = open(outfile,"w")
    f.write(s)
    f.close()

