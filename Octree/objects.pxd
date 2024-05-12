cdef class Object:
    cdef public double x, y, z
    cdef public bint fixed
    cdef public int id
    cdef public int node_id
    cdef public object data

    cpdef double getX(self)
    cpdef double getY(self)
    cpdef double getZ(self)
    cpdef bint is_fixed(self)
    cpdef void set_fixed(self, bint value)
    cpdef void set_id(self, int id_value)
    cpdef int get_id(self)
    cpdef void set_position(self, tuple position)
    cpdef void setData(self, object data)
    cpdef object getData(self)
    cpdef void setNodeID(self, int id)
    cpdef int getNodeID(self)