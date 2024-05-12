cimport objects

cdef class Object:
    def __init__(self, position, id=-1):
        self.x, self.y, self.z = position
        self.fixed = False
        self.id = id
        self.node_id = 0
        self.data = None

    cpdef double getX(self):
        return self.x

    cpdef double getY(self):
        return self.y

    cpdef double getZ(self):
        return self.z

    cpdef bint is_fixed(self):
        return self.fixed

    cpdef void set_fixed(self, bint value):
        self.fixed = value

    cpdef void set_id(self, int id_value):
        self.id = id_value

    cpdef int get_id(self):
        return self.id

    cpdef void set_position(self, tuple position):
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])

    cpdef void setData(self, object data):
        self.data = data

    cpdef object getData(self):
        return self.data

    cpdef void setNodeID(self, int id):
        self.node_id = id

    cpdef int getNodeID(self):
        return self.node_id