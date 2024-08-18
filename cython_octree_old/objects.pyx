import numpy as np
cimport numpy as np

cdef class Object:
    """
    Represents an object in the octree.
    """
    
    def __init__(self, tuple position, int id=-1):
        """
        Initialize an object with a given position.
        
        Args:
            position (tuple): Tuple containing the x, y, and z coordinates of the object.
            id (int): Unique identifier for the object (default is -1).
        """
        self.x, self.y, self.z = position
        self.node_id = -1        
        self.fixed = False
        self.id = id
    
    cpdef double getX(self):
        """
        Get the x-coordinate of the object's position.
        
        Returns:
            float: X-coordinate of the object.
        """
        return self.x

    cpdef double getY(self):
        """
        Get the y-coordinate of the object's position.
        
        Returns:
            float: Y-coordinate of the object.
        """
        return self.y

    cpdef double getZ(self):
        """
        Get the z-coordinate of the object's position.
        
        Returns:
            float: Z-coordinate of the object.
        """
        return self.z

    cpdef bint is_fixed(self):
        """
        Check if the object is fixed.
        
        Returns:
            bool: True if the object is fixed, False otherwise.
        """
        return self.fixed
    
    cpdef void set_fixed(self, bint value):
        """
        Set the fixed attribute of the object.
        
        Args:
            value (bool): New value for the fixed attribute.
        """
        self.fixed = value
        
    cpdef void set_id(self, int id_value):
        """
        Set the id of the object.
        
        Args:
            id_value (int): New identifier for the object.
        """
        self.id = id_value
        
    cpdef double distance(self, Object other):
        """
        Calculate the distance between this object and another object.
        
        Args:
            other (Object): The other object to calculate the distance to.
            
        Returns:
            double: The distance between the two objects.
        """
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)
    
    cpdef void set_position(self, tuple position):
        """
        Set the position of the object.
        
        Args:
            position (tuple): Tuple containing the new x, y, and z coordinates of the object.
        """
        self.x, self.y, self.z = position
