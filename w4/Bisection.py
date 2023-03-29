import numpy as np

def Bisection(f,a,b,nmax,tol):
    """ approximates a root, c, of f bounded
        by a and b to within tolerance
        |f(m)| < tol with m being the midpoint
        between a and b, Recursive implementation """
    
    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        print('a = %0.1f b= %0.1f f(a) = %1.2e f(b) = %1.2e' % (a, b, fa, fb))
        print('funcion has same signs at a and b')
        return
    
    error = b - a

    for n in range(0, nmax):
        error = error/2
        c = a + error
        fc = f(c)

        print('n = %02d c = %0.7f f(c) = %1.2e error = %1.2e' % (n, c, fc, error))

        if np.abs(error) < tol:
            print('convergence')
            return c
        if np.sign(fa) != np.sign(fc):
            b = c
            fb = fc
        else:
            a = c
            fa = fc