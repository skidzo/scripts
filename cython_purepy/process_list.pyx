cimport cython
from libc.stdlib cimport malloc, free

def process(a, int len):

    cdef int *my_ints

    my_ints = <int *>malloc(len(a)*cython.sizeof(int))
    if my_ints is NULL:
        raise MemoryError()

    for i in xrange(len(a)):
        my_ints[i] = a[i]

    with nogil:
        #Once you convert all of your Python types to C types,
        #then you can release the GIL and do the real work
        ...
        free(my_ints)

    #convert back to python return type
    return value
