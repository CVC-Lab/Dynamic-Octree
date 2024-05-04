import random

class Object:
    """
    Represents an object in the octree.

    Each object is assumed to be a sphere for simplicity,
    but this representation can be extended to other shapes
    using techniques like the union of spheres.

    Attributes:
        x (float): X-coordinate of the object's position.
        y (float): Y-coordinate of the object's position.
        z (float): Z-coordinate of the object's position.
        fixed (bool): Indicates whether the object is fixed in place.
        id (int or None): Unique identifier for the object.
        object_to_node_map (dictionary): Mapping between Object and node_id
    """
    def __init__(self, position):
        """
        Initialize an object with a given position.

        Args:
            position (tuple): Tuple containing the x, y, and z coordinates of the object.
        """
        self.x, self.y, self.z = position
        self.fixed = False  # Initialize fixed attribute to False
        self.id = None  # Initialize the id attribute
        self.node_id = None  # Initialize the node_id attribute
        # self.object_to_node_map = {}  # Initialize the object to node mapping
        
    def getX(self):
        """
        Get the x-coordinate of the object's position.

        Returns:
            float: X-coordinate of the object.
        """
        return self.x

    def getY(self):
        """
        Get the y-coordinate of the object's position.

        Returns:
            float: Y-coordinate of the object.
        """
        return self.y

    def getZ(self):
        """
        Get the z-coordinate of the object's position.

        Returns:
            float: Z-coordinate of the object.
        """
        return self.z
    
    def is_fixed(self):
        """
        Check if the object is fixed.

        Returns:
            bool: True if the object is fixed, False otherwise.
        """
        return self.fixed
    
    def set_fixed(self, value):
        """
        Set the fixed attribute of the object.

        Args:
            value (bool): New value for the fixed attribute.
        """
        self.fixed = value
        
    def set_id(self, id_value):
        """
        Set the id of the object.

        Args:
            id_value (int or None): New identifier for the object.
        """
        self.id = id_value

    def get_id(self):
        return self.id
    
    def set_position(self, position):
        r"""
        Sets the position of the object given a position tuple

        args:
            position(tuple(x, y, z)): New position of the objec
        """
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])