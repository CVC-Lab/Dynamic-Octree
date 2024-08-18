cdef class Object:
    cdef double x
    cdef double y
    cdef double z
    cdef bint fixed
    cdef int id
    cdef int node_id

    cpdef double getX(self)
    cpdef double getY(self)
    cpdef double getZ(self)
    cpdef bint is_fixed(self)
    cpdef void set_fixed(self, bint value)
    cpdef void set_id(self, int id_value)
    cpdef double distance(self, Object other)
    cpdef void set_position(self, tuple position)
