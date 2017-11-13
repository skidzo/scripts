import cython

@cython.locals(t = cython.int, i = cython.int)
cpdef int dostuff(int n)

cpdef int c_function(int x, int y=*)
cdef double _helper(double a)

cdef class A:
    cdef public double a,b
    cpdef foo(self, double x)
