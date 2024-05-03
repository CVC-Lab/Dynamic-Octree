import sys, math, random
from objects import Object   # Import the Object class from the objects module

INIT_NUM_OCTREE_NODES = 10  # Initial number of octree nodes
LOW_BITS = 14  # Low bits used for generating the Octree IDs

# Class for storing octree scores
class OctreeScore:
    def __init__(self):
        self.score = 0

    def set_score(self, d):
        self.score = d

    def get_score(self):
        return self.score

# Class for construction parameters of the octree
class OctreeConstructionParams:
    def __init__(self, max_leaf_size=5, max_leaf_dim=6, slack_factor=1.0):
        self.max_leaf_size = max_leaf_size
        self.max_leaf_dim = max_leaf_dim
        self.slack_factor = slack_factor

    # Setter methods
    def set_max_leaf_size(self, max_leaf_size):
        self.max_leaf_size = max_leaf_size

    def set_max_leaf_dim(self, max_leaf_dim):
        self.max_leaf_dim = max_leaf_dim

    def set_slack_factor(self, slack_factor):
        self.slack_factor = slack_factor

    # Getter methods
    def get_max_leaf_size(self):
        return self.max_leaf_size

    def get_max_leaf_dim(self):
        return self.max_leaf_dim

    def get_slack_factor(self):
        return self.slack_factor

    # Method to print construction parameters
    def print_params(self):
        print(f"Max leaf size: {self.max_leaf_size}, and max leaf dim: {self.max_leaf_dim}.")


# params = OctreeConstructionParams()
# params.print_params()  # Output: Max leaf size: 50, and max leaf dim: 6.
# params.set_max_leaf_size(100)
# params.set_max_leaf_dim(8)
# params.set_slack_factor(0.5)
# params.print_params()  # Output: Max leaf size: 100, and max leaf dim: 8.

# Class for storing attributes of an octree node
class DynamicOctreeNodeAttr:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.sx = x
        self.sy = y
        self.sz = z
    
    # Method to combine attributes
    def combine_s(self, all_attribs):
        self.sx = sum(attr.sx for attr in all_attribs)
        self.sy = sum(attr.sy for attr in all_attribs)
        self.sz = sum(attr.sz for attr in all_attribs)
    
    # Method to compute attributes
    def compute_s(self, atoms):
        self.sx = sum(atom.x for atom in atoms)
        self.sy = sum(atom.y for atom in atoms)
        self.sz = sum(atom.z for atom in atoms)
    
    # Method to update attributes
    def update_s(self, atm, add):
        if add:
            self.sx += atm.x
            self.sy += atm.y
            self.sz += atm.z
        else:
            self.sx -= atm.x
            self.sy -= atm.y
            self.sz -= atm.z

# Class for representing an octree node
class DynamicOctreeNode:
    def __init__(self, node_id=None):
        # Initialize attributes of the node
        self.lx = 0.0  # Position of the node
        self.ly = 0.0  # Position of the node
        self.lz = 0.0  # Position of the node
        self.dim = 0.0  # Dimension of the node
        self.num_atoms = 0  # Number of objects contained in the node
        self.n_fixed = 0  # Number of fixed objects in the node
        self.id_cap = 5  # Capacity for IDs
        self.id_num = 0  # ID number
        self.atom_indices = []  # Indices of objects contained in the node
        self.parent_pointer = -1  # Pointer to the parent node
        self.child_pointer = [-1] * 8  # Pointers to child nodes
        self.leaf = True  # Flag indicating if the node is a leaf
        self.attribs = DynamicOctreeNodeAttr()  # Attributes of the node
    
    # Method to initialize node attributes
    def init_node(self):
        self.lx = self.ly = self.lz = 0.0
        self.dim = 0.0
        self.num_atoms = 0
        self.n_fixed = 0
        self.id_cap = 0
        self.id_num = 0
        self.atom_indices = []
        self.parent_pointer = -1
        self.child_pointer = [-1] * 8
        self.leaf = True
        self.id = None  # Add an id attribute to the node
        self.attribs = DynamicOctreeNodeAttr()
        
    # Method to check if the node is a leaf
    def is_leaf(self):
        """
        Check if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self.leaf
        
    # Setter methods for node attributes
    def set_child_pointer(self, loc, ptr):
        if 0 <= loc < 8:
            self.child_pointer[loc] = ptr
        else:
            print("Error: Invalid child pointer index")
        
    def set_parent_pointer(self, i):
        self.parent_pointer = i
    
    def get_parent_pointer(self):
        return self.parent_pointer
    
    def set_id(self, node_id):
        self.id = node_id
    
    def set_lx(self, value):
        self.lx = value

    def set_ly(self, value):
        self.ly = value

    def set_lz(self, value):
        self.lz = value
    
    def set_dim(self, value):
        self.dim = value
        
    def set_num_atoms(self, i):
        self.num_atoms = i
        
    def set_atom_indices(self, i):
        self.atom_indices = i
        
    def set_leaf(self, i):
        self.leaf = i
        
    def set_IdCap(self, i):
        self.id_cap = i
        
    def set_num_fixed(self, i):
        self.n_fixed = i
        
    def set_atom_index(self, loc, index):
        # Check if the location is valid
        if loc < 0:
            raise ValueError("Location index must be non-negative")
        
        # If the location is beyond the current size, extend the list with None values
        while loc >= len(self.atom_indices):
            self.atom_indices.append(None)
        
        # Assign the index value
        self.atom_indices[loc] = index
    
    # Getter method for node dimension
    def get_dim(self):
        return self.dim
    
    # Method to combine and set node attributes
    def combine_and_set_attribs(self, all_child_attribs):
        self.attribs.combine_s(all_child_attribs)
    
    # Method to compute own attributes
    def compute_own_attribs(self, atoms):
        self.attribs.compute_s(atoms)
    
    # Method to update attributes
    def update_attribs(self, obj, add):
        if add:
            self.attribs.update_s(obj, add=True)
            self.num_atoms += 1  # Increment num_atoms when adding an object
        else:
            self.attribs.update_s(obj, add=False)
            self.num_atoms -= 1  # Decrement num_atoms when removing an object

class DynamicOctree:
    def __init__(self, atoms, n_atoms, cons_par, verbose=True, max_nodes=None):
        """
        Initialize a DynamicOctree object.

        Args:
            atoms: List of atoms.
            n_atoms: Number of atoms.
            cons_par: Octree construction parameters.
            verbose: Verbosity flag (default is False).
            max_nodes: Maximum number of nodes allowed in the octree.
        """
        self.nodes = []  # List to store octree nodes
        self.atoms = atoms  # List of atoms
        self.num_atoms = n_atoms  # Number of atoms
        self.num_nodes = 0  # Number of nodes in the octree
        self.next_free_node = -1  # Index of the next free node
        self.octree_built = False  # Flag indicating if the octree is built
        self.verbose = verbose  # Verbosity flag
        self.construction_params = cons_par  # Octree construction parameters
        self.max_nodes = max_nodes  # Maximum number of nodes allowed in the octree
        self.scoring_params = None  # Scoring parameters for the octree
        self.object_to_node_map = {} # Initialize object to node mapping dictionary
        
    def set_node_id(self, obj, node_id):
        """
        Set the node ID for the object.

        Args:
            obj (Object): The object for which to set the node ID.
            node_id (int): Node ID to associate with the object.
        """
        self.object_to_node_map[obj] = node_id
        
    def get_node_id(self, obj):
        """
        Get the node ID associated with the object.

        Args:
            obj (Object): The object for which to get the node ID.

        Returns:
            int or None: Node ID associated with the object, or None if not mapped.
        """
        return self.object_to_node_map.get(obj)

    def build_octree(self):
        """
        Build the octree.

        Returns:
            bool: True if the octree is built successfully, False otherwise.
        """
        if self.verbose:
            print("Starting DynamicOctree::buildOctree")

        # Allocate temporary storage for indices
        indices = [i for i in range(self.num_atoms)]
        print("Number of Atoms: ", self.num_atoms)
        indices_temp = [0] * self.num_atoms

        try:
            if not self.init_free_node_server():
                print("ERROR: Could not create free node server")
                return False

            octree_root = self.get_next_free_node()
            
            print("Root Node: ", octree_root)

            if octree_root == -1:
                print("ERROR: Could not get next node")
                return False

            # Compute root bounding box
            self.compute_root_bounding_box(octree_root, self.construction_params.get_slack_factor(), indices, 0, self.num_atoms - 1)

            self.nodes[octree_root].set_parent_pointer(-1)

            print("Numeber of atoms considered while expanding octree: ", indices)
            # Expand octree node
            self.octree_built = self.expand_octree_node(octree_root, indices, indices_temp, 0, self.num_atoms - 1)
            """
            # Update object to node mapping
            for atom_id in indices:
                atom = self.atoms[atom_id]
                node_id = self.get_node_id(atom)
                if node_id is not None:
                    self.object_to_node_map[atom] = octree_root  # Update node ID to the root node
            """
        finally:
            # Free temporary storage
            del indices
            del indices_temp

        print("Octree built")
        print("Number of Nodes:", self.num_nodes)

        if self.verbose:
            self.print_octree()

        return self.octree_built
    
    def create_octree_ptr(self, a, b):
        """
        Create an octree pointer from two integers.

        Parameters:
            a (int): First integer.
            b (int): Second integer.

        Returns:
            int: Octree pointer.
        """
        return (a << LOW_BITS) + b
    
    def get_index_in_node(self, c):
        """
        Get the index within the node from the given value.

        Parameters:
        c (int): The value containing the index within the node.

        Returns:
        int: The index within the node.
        """
        return c & 0x3FFF
    
    def needs_dynamic_expansion(self, node):
        """
        Checks whether a node needs dynamic expansion.

        Args:
            node (DynamicOctreeNode): The node to check.

        Returns:
            bool: True if the node needs dynamic expansion, False otherwise.
        """
        return node.num_atoms > (self.construction_params.max_leaf_size << 1)
    
    def needs_dynamic_contraction(self, node):
        """
        Check if dynamic contraction is needed for a node.

        Parameters:
        node (DynamicOctreeNode): The node to check.

        Returns:
        bool: True if dynamic contraction is needed, False otherwise.
        """
        return node.num_atoms < (self.construction_params.max_leaf_size >> 1)

    def allocate_nodes(self, new_num_nodes):
        """
        Allocate new nodes for the octree.

        Args:
            new_num_nodes (int): Number of new nodes to allocate.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("Inside DynamicOctree::allocateNodes({})".format(new_num_nodes))

        nodes = []
        for _ in range(new_num_nodes):
            node = DynamicOctreeNode()
            nodes.append(node)

        self.nodes = nodes

        if self.verbose:
            print("Allocated {} nodes".format(new_num_nodes))

        return True


    def reallocate_nodes(self, new_num_nodes):
        """
        Reallocate nodes for the octree.

        Args:
            new_num_nodes (int): Number of new nodes to reallocate.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("Inside DynamicOctree::reallocateNodes({})".format(new_num_nodes))

        self.nodes = self.nodes[:new_num_nodes] + [DynamicOctreeNode() for _ in range(new_num_nodes - len(self.nodes))]

        if self.verbose:
            print("Allocated {} nodes".format(new_num_nodes))

        return True

    def init_free_node_server(self):
        """
        Initialize the free node server.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::initFreeNodeServer")

        self.num_nodes = 0
        self.next_free_node = -1

        self.allocate_nodes(INIT_NUM_OCTREE_NODES)

        if self.nodes is None:
            return False

        self.num_nodes = INIT_NUM_OCTREE_NODES

        # Set parent pointers and update next free node
        for i in range(self.num_nodes - 1, -1, -1):
            self.nodes[i].set_parent_pointer(self.next_free_node)
            self.next_free_node = i

        if self.verbose:
            print("Allocated {} new nodes".format(INIT_NUM_OCTREE_NODES))
        
        # Print parent nodes
        print("Parent Nodes:")
        for i in range(self.num_nodes):
            parent_node_id = self.nodes[i].get_parent_pointer()
            if parent_node_id != -1:
                print(f"Node ID: {i}, Parent Node ID: {parent_node_id}")

        return True

    # def get_next_free_node(self):  TO ALLOCATE ONLY ONE NODE WHEN THERE ARE NO FREE NODES
    #     """
    #     Get the index of the next free node.

    #     Returns:
    #         int: Index of the next free node.
    #     """
    #     if self.verbose:
    #         print("In DynamicOctree::getNextFreeNode")

    #     if self.next_free_node == -1:
    #         self.reallocate_nodes(1)  # Allocate only one new node

    #         if self.nodes is None:
    #             return -1

    #         # Set parent pointer for the new node
    #         self.nodes[-1].set_parent_pointer(self.next_free_node)
    #         self.next_free_node = len(self.nodes) - 1

    #     next_node = self.next_free_node
    #     self.next_free_node = self.nodes[next_node].get_parent_pointer()

    #     if self.verbose:
    #         print("Next node is", next_node)

    #     return next_node

    def get_next_free_node(self):
        """
        Get the index of the next free node.

        Returns:
            int: Index of the next free node.
        """
        if self.verbose:
            print("In DynamicOctree::getNextFreeNode")

        if self.next_free_node == -1:
            new_num_nodes = 2 * self.num_nodes

            if new_num_nodes <= 0:
                new_num_nodes = INIT_NUM_OCTREE_NODES

            self.reallocate_nodes(new_num_nodes)

            if self.nodes is None:
                return -1

            # Set parent pointers and update next free node
            for i in range(new_num_nodes - 1, self.num_nodes, -1):
                self.nodes[i] = DynamicOctreeNode(i)  # Initialize node with ID
                self.nodes[i].set_parent_pointer(self.next_free_node)
                self.next_free_node = i

            self.num_nodes = new_num_nodes

        next_node = self.next_free_node

        self.next_free_node = self.nodes[next_node].get_parent_pointer()

        if self.verbose:
            print("Next node is", next_node)

        return next_node

    def compute_root_bounding_box(self, node_id, slack_factor, indices, start_id, end_id):
        """
        Compute the bounding box for the root node.

        Args:
            node_id: ID of the node.
            slack_factor: Slack factor.
            indices: List of indices.
            start_id: Start index.
            end_id: End index.
        """
        if self.verbose:
            print("In DynamicOctree::computeRootBoundingBox")

        node = self.nodes[node_id]

        s = indices[start_id]

        minX, minY, minZ = maxX, maxY, maxZ = self.atoms[s].getX(), self.atoms[s].getY(), self.atoms[s].getZ()

        for i in range(start_id + 1, end_id + 1):
            j = indices[i]

            minX = min(minX, self.atoms[j].getX())
            maxX = max(maxX, self.atoms[j].getX())

            minY = min(minY, self.atoms[j].getY())
            maxY = max(maxY, self.atoms[j].getY())

            minZ = min(minZ, self.atoms[j].getZ())
            maxZ = max(maxZ, self.atoms[j].getZ())

        cx, cy, cz = (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2

        dim = max(maxX - minX, maxY - minY, maxZ - minZ)
        dim *= slack_factor

        node.set_lx(cx - dim * 0.5)
        node.set_ly(cy - dim * 0.5)
        node.set_lz(cz - dim * 0.5)

        node.set_dim(dim)

        if self.verbose:
            print("root dim =", dim)
            
    def needs_expansion(self, node):
        """
        Checks whether a node needs expansion based on the maximum leaf size and dimension.

        Args:
            node (DynamicOctreeNode): The node to check.

        Returns:
            bool: True if the node needs expansion, False otherwise.
        """
        # # Check if the node is the root node and if it contains more atoms than the maximum leaf size
        # if node.parent_pointer == -1 and node.num_atoms > self.construction_params.get_max_leaf_size():
        #     return True
        
        # # Check if the dimension of the node exceeds the maximum leaf dimension
        # return node.dim > self.construction_params.get_max_leaf_dim()
        
        # Check if the number of atoms in the node exceeds the maximum leaf size
        # or if the dimension of the node exceeds the maximum leaf dimension
        return not (node.num_atoms <= self.construction_params.get_max_leaf_size() or node.dim <= self.construction_params.get_max_leaf_dim())


    def expand_octree_node(self, node_id, indices, indices_temp, start_id, end_id):
        """
        Expand the octree node.

        Args:
            node_id: ID of the node.
            indices: List of indices.
            indices_temp: Temporary list of indices.
            start_id: Start index.
            end_id: End index.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::expandOctreeNode")

        node = self.nodes[node_id]
        print("Start and End IDs: ", start_id, end_id)
        nAtoms = end_id - start_id + 1
        node.set_num_atoms(nAtoms)
        print("nAtoms: ", nAtoms)
        
        # self.compute_leaf_attributes(node_id, indices, start_id, end_id)
        dim = node.get_dim()

        # The node is a leaf. If the atom is fixed, it is placed at the beginning of the list of atom indices.
        if not self.needs_expansion(node):
            node.set_leaf(True)

            new_indices = [-1] * (2 * nAtoms)
            
            if new_indices is None:
                print("Failed to allocate leaf node memory for octree!")
                return False
            
            node.set_atom_indices(new_indices)
            node.set_IdCap(2*nAtoms)
            
            nfixed = 0

            for i in range(start_id, end_id + 1):
                j = indices[i]

                if self.atoms[j].is_fixed():
                    node.set_atom_index(nfixed, j)
                    self.atoms[j].node_id = node_id  # Update node_id attribute
                    nfixed += 1
                else:
                    new_indices[i - start_id + nfixed] = j
                    
            node.set_num_fixed(nfixed)

            if nfixed < nAtoms:
                k = nfixed
                for i in range(start_id, end_id + 1):
                    j = indices[i]

                    if not self.atoms[j].is_fixed():
                        node.set_atom_index(k, j)
                        self.atoms[j].node_id = node_id  # Update node_id attribute
                        k += 1

        # If the node is not a leaf (i.e., it needs expansion), it proceeds to subdivide the node.
        else:
            node.set_leaf(0)

            count = [0] * 8

            for i in range(start_id, end_id + 1):
                j = indices[i]
                k = self.get_child_id(node, self.atoms[j])
                count[k] += 1

            start_index = [0] * 8
            cur_index = [0] * 8

            cur_index[0] = start_index[0] = start_id

            for i in range(1, 8):
                cur_index[i] = start_index[i] = start_index[i - 1] + count[i - 1]

            for i in range(start_id, end_id + 1):
                j = indices[i]
                k = self.get_child_id(node, self.atoms[j])
                indices_temp[cur_index[k]] = j
                cur_index[k] += 1

            nfxd = 0

            for i in range(8):
                if count[i] > 0:
                    j = self.get_next_free_node()

                    node.set_child_pointer(i, j)
                    self.nodes[j].set_parent_pointer(node_id)

                    self.compute_non_root_bounding_box(j, i)

                    if not self.expand_octree_node(j, indices_temp, indices, start_index[i], start_index[i] + count[i] - 1):
                        return False

                    nfxd += self.nodes[j].n_fixed

                    node.set_num_fixed(nfxd)

                else:
                    node.set_child_pointer(i, -1)

        # Update object_to_node_map for each atom in the expanded node
        for i in range(start_id, end_id + 1):
            j = indices[i]
            print(f"Mapping {self.atoms[j]} to {node_id}")
            self.object_to_node_map[self.atoms[j]] = node_id
        return True

    def print_octree(self):
        """
        Print the octree.

        This method traverses the octree and prints its contents, including objects in each node.
        """
        if self.verbose:
            print("In DynamicOctree::print")
        print("\n")
        # self.traverse_and_print(0, 3)
        self.print_node_details(0)
        print("\n")
        
    def print_node_details(self, node_id, indent=""):
        """
        Print details of a node and its children recursively.

        Args:
            node_id (int): The ID of the node to print.
            indent (str): Indentation string for better visualization.
        """
        node = self.nodes[node_id]
        print(indent + f"Node ID: {node_id}")
        print(indent + f"Parent Pointer: {node.parent_pointer}")
        print(indent + f"Node Attributes: {node.attribs}")  # Print any other relevant node attributes
        
        if node.leaf:
            print(indent + "Leaf Node")
            print(indent + f"Atom Indices: {node.atom_indices}")
        else:
            print(indent + "Non-Leaf Node")
            print(indent + f"Child Pointers: {node.child_pointer}")

            # Recursively print details of children nodes
            for child_id in node.child_pointer:
                if child_id != -1:
                    self.print_node_details(child_id, indent + "  ")

    def traverse_and_print(self, node_id, depth):
        """
        Traverse the octree and print the contents of each node.

        Args:
            node_id: ID of the current node.
            depth: Depth of the current node in the octree.
        """
        node = self.nodes[node_id]
        prefix = "  " * depth
        print(prefix + f"Node ID: {node_id}")
        print(prefix + f"Position: ({node.lx}, {node.ly}, {node.lz})")
        print(prefix + f"Dimension: {node.dim}")
        print(prefix + f"Number of Atoms: {node.num_atoms}")
        print(prefix + f"Number of Fixed Atoms: {node.n_fixed}")
        print(prefix + f"Atom Indices: {node.atom_indices}")
        print(prefix + f"Parent Pointer: {node.parent_pointer}")
        print(prefix + f"Child Pointers: {node.child_pointer}")
        print(prefix + f"Is Leaf: {node.leaf}")
        print(prefix + "Attributes:")
        print(prefix + f"  sx: {node.attribs.sx}")
        print(prefix + f"  sy: {node.attribs.sy}")
        print(prefix + f"  sz: {node.attribs.sz}")

        if not node.leaf:
            for i, child_id in enumerate(node.child_pointer):
                if child_id != -1:
                    print(prefix + f"Child {i}:")
                    self.traverse_and_print(child_id, depth + 1)
        print("\n")

    def compute_non_root_bounding_box(self, node_id, child_id=None):
        """
        Compute the bounding box for a non-root node.

        Args:
            node_id (int): The ID of the node.
            child_id (int): The ID of the child node.

        This method computes the bounding box for a non-root node based on its parent's bounding box.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeNonRootBoundingBox(node_id={node_id}, child_id={child_id})")

        node = self.nodes[node_id]

        if node.parent_pointer < 0:
            return

        pnode = self.nodes[node.parent_pointer]

        if child_id is None:
            for i in range(8):
                if pnode.child_pointer[i] == node_id:
                    child_id = i
                    break

            if child_id is None:
                return

        lx = pnode.lx
        ly = pnode.ly
        lz = pnode.lz
        dim = pnode.dim

        dim *= 0.5

        if child_id & 1:
            lx += dim
        if child_id & 2:
            ly += dim
        if child_id & 4:
            lz += dim

        node.lx = lx
        node.ly = ly
        node.lz = lz
        node.dim = dim
        
        # Assign the parent node's ID to the child node's ID
        node.set_id(node_id)  # Assign the ID to the node

        if self.verbose:
            print(f"nonroot dim = {dim}")

    def compute_non_leaf_attributes(self, node_id):
        """
        Compute the attributes for a non-leaf node.

        Args:
            node_id (int): The ID of the node.

        This method computes the attributes for a non-leaf node by combining the attributes of its child nodes.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeNonLeafAttributes(node_id={node_id})")

        sumX, sumY, sumZ = 0, 0, 0
        sumQ = 0

        child_attribs = []

        for i in range(8):
            child_id = self.nodes[node_id].child_pointer[i]
            if child_id >= 0:
                child_attribs.append(self.nodes[child_id].attribs)

        self.nodes[node_id].combine_and_set_attribs(child_attribs)

    def compute_leaf_attributes(self, node_id, indices, start_id, end_id):
        """
        Compute the attributes for a leaf node.

        Args:
            node_id (int): The ID of the node.
            indices (list): List of atom indices.
            start_id (int): Start index of atoms.
            end_id (int): End index of atoms.

        This method computes the attributes for a leaf node based on the provided atom indices.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeLeafAttributes(node_id={node_id}, indices[start_id]={indices[start_id]}, indices[end_id]={indices[end_id]})")

        node_atoms = [self.atoms[j] for j in indices[start_id:end_id + 1]]

        # Update node_id attribute for each object
        for atom in node_atoms:
            atom.node_id = node_id

        print("prepared list of atoms")

        self.nodes[node_id].compute_own_attribs(node_atoms)

        print("computed attributes")

    def get_child_id(self, node, atom):
        """
        Get the child ID of a node based on the position of an object.

        Args:
            node (DynamicOctreeNode): The node.
            atom (Object): The object.

        Returns:
            int: The child ID.

        This method returns the child ID of a node based on the position of the given object.
        """
        dim = 0.5 * node.dim
        cx, cy, cz = node.lx + dim, node.ly + dim, node.lz + dim

        k = ((atom.getZ() >= cz) << 2) + ((atom.getY() >= cy) << 1) + (atom.getX() >= cx)

        return k
        
    def collect_atoms_from_leaves(self, node_id, indices, start_id):
        """
        Recursively collects atoms from leaf nodes.

        Args:
            node_id (int): The ID of the current node.
            indices (List[int]): The list to store atom indices.
            start_id (int): The starting index in the indices list.

        This method recursively collects atoms from leaf nodes and stores their indices in the indices list.
        """
        if self.nodes[node_id].is_leaf():
            for i in range(self.nodes[node_id].num_atoms):
                indices[start_id + i] = self.nodes[node_id].atom_indices[i]
        else:
            for i in range(8):
                child_id = self.nodes[node_id].child_pointers[i]
                if child_id >= 0:
                    self.collect_atoms_from_leaves(child_id, indices, start_id)
                    start_id += self.nodes[child_id].num_atoms
                    self.free_node(child_id)


    def contract_octree_node(self, node_id):
        """
        Contracts the octree node if necessary.

        Args:
            node_id (int): The ID of the node to contract.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::contractOctreeNode")

        node = self.nodes[node_id]

        n_atoms = node.num_atoms

        if not self.needs_contraction(node):
            return True

        self.compute_non_leaf_attributes(node_id)

        new_indices = [0] * (2 * n_atoms)

        self.collect_atoms_from_leaves(node_id, new_indices, 0)

        k = 0
        for i in range(n_atoms):
            j = new_indices[i]
            if self.atoms[j].is_fixed():
                new_indices[k], new_indices[i] = new_indices[i], new_indices[k]
                k += 1

        for i in range(n_atoms):
            j = new_indices[i]
            self.atoms[j].set_id(self.create_octree_ptr(node_id, i))

        node.is_leaf = True
        node.id_cap = 2 * n_atoms
        node.atom_indices = new_indices

        return True

    def traverse_octree(self, node_id):
        """
        Traverses the octree recursively and prints information about each node.

        Args:
            node_id (int): The ID of the current node.
        """
        if self.verbose:
            print("In DynamicOctree::traverseOctree")

        node = self.nodes[node_id]

        print("%d ( %d, %lf ): " % (node_id, node.num_atoms, node.dim), end="") # REMOVED num_fixed from here, CHECK IF THAT IS USEFUL

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    print("%d " % node.child_pointers[i], end="")

        print()

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    self.traverse_octree(node.child_pointers[i])

    def get_subtree_size(self, node_id):
        """
        Calculates the size of the subtree rooted at the given node.

        Args:
            node_id (int): The ID of the root node of the subtree.

        Returns:
            int: The size of the subtree.
        """
        if self.verbose:
            print("In DynamicOctree::getSubtreeSize (Recursive)")

        node = self.nodes[node_id]

        s = sys.getsizeof(DynamicOctreeNode)

        s += node.id_cap * sys.getsizeof(int)

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    s += self.get_subtree_size(node.child_pointers[i])

        return s

    def get_octree_size(self):
        """
        Calculates the size of the entire octree.

        Returns:
            int: The size of the octree.
        """
        if self.verbose:
            print("In DynamicOctree::getOctreeSize")

        return self.get_subtree_size(0)

    def remove_atom_from_leaf(self, node_id, atom_id):
        """
        Removes an atom from a leaf node.

        Args:
            node_id (int): The ID of the leaf node.
            atom_id (int): The ID of the atom to remove.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::removeAtomFromLeaf")

        atom = self.atoms[atom_id]

        j = self.get_index_in_node(atom.id)

        node = self.nodes[node_id]

        n = node.num_atoms
        index_nf, index_j = 0, 0

        if atom.is_fixed():
            nf = node.n_fixed - 1

            node.atom_indices[j] = node.atom_indices[nf]
            node.atom_indices[nf] = node.atom_indices[n - 1]

            index_nf = node.atom_indices[nf]
            index_j = node.atom_indices[j]

            self.atoms[index_nf].id = self.atoms[index_j].id
            self.atoms[index_j].id = atom.id

            node.n_fixed = nf
        else:
            node.atom_indices[j] = node.atom_indices[n - 1]
            self.atoms[node.atom_indices[j]].id = atom.id

        # Update node_id attribute of the removed atom
        self.atoms[atom_id].node_id = None
        
        # Remove the mapping from the object to node map
        if atom_id in self.object_to_node_map:
            del self.object_to_node_map[atom_id]

        node.num_atoms -= 1

        if n <= (node.id_cap >> 2):
            new_indices = node.atom_indices[:node.id_cap >> 1]

            if new_indices is None:
                print("Failed to contract leaf storage for octree!")
                return False

            node.atom_indices = new_indices
            node.id_cap = node.id_cap >> 1

        node.update_attribs(atom, False)
        atom.id = -1

        return True

    def remove_atom_from_non_leaf(self, node_id, atom_id):
        """
        Removes an atom from a non-leaf node.

        Args:
            node_id (int): The ID of the non-leaf node.
            atom_id (int): The ID of the atom to remove.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::removeAtomFromNonLeaf")

        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        # # Find the index of the atom in the node's atom_indices list
        # atom_index = None
        # for i in range(node.num_atoms):
        #     if node.atom_indices[i] == atom_id:
        #         atom_index = i
        #         break

        # if atom_index is None:
        #     # Atom not found in the node's atom_indices list
        #     return False

        # # Remove the atom from the node's atom_indices list
        # del node.atom_indices[atom_index]

        node.num_atoms -= 1

        if atom.is_fixed():
            node.num_fixed -= 1

        node.update_attribs(atom, False)

        # Update the node_id attribute of the removed atom to None
        atom.node_id = None

        # Remove the atom from the object_to_node_map dictionary
        del self.object_to_node_map[atom]

        # Check if the node needs dynamic contraction and contract if necessary
        if self.needs_dynamic_contraction(node):
            return self.contract_octree_node(node_id)

        return True

    def add_atom_to_non_leaf(self, node_id, atom_id):
        """
        Adds an atom to a non-leaf node.

        Args:
            node_id (int): The ID of the non-leaf node.
            atom_id (int): The ID of the atom to add.
        """
        if self.verbose:
            print("In DynamicOctree::addAtomToNonLeaf")

        # Ensure nodes and atoms lists have sufficient capacity
        if node_id >= len(self.nodes):
            self.nodes += [DynamicOctreeNode() for _ in range(node_id - len(self.nodes) + 1)]
        if atom_id >= len(self.atoms):
            self.atoms += [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(atom_id - len(self.atoms) + 1)]

        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        # Increase the number of atoms in the node
        node.num_atoms += 1

        # Update the node attributes based on the added atom
        node.update_attribs(atom, True)

        # If the added atom is fixed, increase the number of fixed atoms in the node
        if atom.is_fixed():
            node.num_fixed += 1

        # Update the node_id attribute of the added atom
        atom.node_id = node_id

        # Update the object_to_node_map
        self.object_to_node_map[atom] = node_id

    def add_atom_to_leaf(self, node_id, atom_id):
        """
        Adds an atom to a leaf node.

        Args:
            node_id (int): The ID of the leaf node.
            atom_id (int): The ID of the atom to add.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::addAtomToLeaf")

        # Ensure nodes and atoms lists have sufficient capacity
        if node_id >= len(self.nodes):
            self.nodes += [DynamicOctreeNode() for _ in range(node_id - len(self.nodes) + 1)]
        if atom_id >= len(self.atoms):
            self.atoms += [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(atom_id - len(self.atoms) + 1)]

        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        n = node.num_atoms

        # Ensure capacity for atom indices in the leaf node
        if n == node.id_cap:
            if node.id_cap == 0:
                node.id_cap = 1
                node.atom_indices = [None] * (node.id_cap << 1)
            else:
                node.atom_indices += [None] * (node.id_cap << 1)
            if node.atom_indices is None:
                print("Failed to expand leaf storage for octree!")
                return False

            node.id_cap = node.id_cap << 1

        # Add atom to the leaf node
        if atom.is_fixed():
            nf = node.num_fixed

            # Swap if there are non-fixed atoms before the added fixed atom
            if n > 0:
                node.atom_indices[n] = node.atom_indices[nf]
                self.atoms[node.atom_indices[n]].id = self.create_octree_ptr(node_id, n)

            # Add the fixed atom to the end
            node.atom_indices[nf] = atom_id
            atom.id = self.create_octree_ptr(node_id, nf)
            atom.node_id = node_id  # Update node_id attribute

            node.num_fixed = nf + 1
        else:
            # Ensure node.atom_indices has sufficient capacity
            if n >= len(node.atom_indices):
                node.atom_indices += [-1] * (n - len(node.atom_indices) + 1)

            # Add the non-fixed atom to the end
            node.atom_indices[n] = atom_id
            atom.id = self.create_octree_ptr(node_id, n)
            atom.node_id = node_id  # Update node_id attribute

        # Update node attributes
        # node.num_atoms += 1
        node.update_attribs(atom, True)

        # Update object_to_node_map
        self.object_to_node_map[atom] = node_id

        # Check if dynamic expansion is needed and expand if necessary
        if self.needs_dynamic_expansion(node):
            temp = [None] * node.num_atoms

            if temp is None:
                print("Failed to allocate temporary storage for octree!")
                return False

            print("Number of atoms: ", node.num_atoms)
            print("Indices: ", len(node.atom_indices))
            print("Node ID: ", node_id)
            done = self.expand_octree_node(node_id, node.atom_indices, temp, 0, node.num_atoms - 1)

            return done
        else:
            return True

    def pull_up(self, node_id, atom_id):
        """
        Pulls up an atom in the octree.

        Args:
            node_id (int): The ID of the current node.
            atom_id (int): The ID of the atom to pull up.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::pullUp")

        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        if not self.inside_node(node, atom):
            if node.parent_pointer < 0:
                print("Atom has moved outside the root bounding box!")
                return False

            if node.is_leaf():
                self.remove_atom_from_leaf(node_id, atom_id)
            else:
                self.remove_atom_from_non_leaf(node_id, atom_id)

            return self.pull_up(node.parent_pointer, atom_id)
        else:
            return self.push_down(node_id, atom_id)


    def push_down(self, node_id, atom_id):
        """
        Pushes down an atom in the octree.

        Args:
            node_id (int): The ID of the current node.
            atom_id (int): The ID of the atom to push down.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::pushDown")

        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        if node.is_leaf():
            return self.add_atom_to_leaf(node_id, atom_id)
        else:
            self.add_atom_to_non_leaf(node_id, atom_id)

            for i in range(8):
                if node.child_pointer[i] >= 0:
                    cnode = self.nodes[node.child_pointer[i]]
                    if self.inside_node(cnode, atom):
                        return self.push_down(node.child_pointer[i], atom_id)
                else:
                    lx, ly, lz = node.lx, node.ly, node.lz
                    hdim = 0.5 * node.dim

                    if i & 1:
                        lx += hdim
                    if i & 2:
                        ly += hdim
                    if i & 4:
                        lz += hdim

                    if not (lx <= atom.x < lx + hdim and ly <= atom.y < ly + hdim and lz <= atom.z < lz + hdim):
                        continue

                    j = self.get_next_free_node()
                    node = self.nodes[node_id]
                    node.child_pointer[i] = j
                    cnode = self.nodes[j]
                    cnode.parent_pointer = node_id
                    cnode.lx = lx
                    cnode.ly = ly
                    cnode.lz = lz
                    cnode.dim = hdim
                    cnode.is_leaf = True
                    return self.push_down(j, atom_id)

            return False

    def destroy_octree(self):
        """
        Destroys the octree, freeing memory.
        """
        if self.verbose:
            print("In DynamicOctree::destroyOctree")

        if self.octree_built:
            self.free_subtree_nodes(0)

        self.free_mem(self.nodes)


    def free_subtree_nodes(self, node_id):
        """
        Frees the memory occupied by the subtree rooted at the given node.

        Args:
            node_id (int): The ID of the node to start the subtree from.
        """
        if self.verbose:
            print("In DynamicOctree::freeSubtreeNodes")

        if node_id >= self.num_nodes or node_id < 0:
            print("Node_id is out of bounds in freeSubtreeNodes")
            return

        node = self.nodes[node_id]

        if not node.is_leaf():
            for i in range(8):
                if node.get_child_pointer(i) >= 0:
                    self.free_subtree_nodes(node.get_child_pointer(i))

        self.free_node(node_id)


    def reorganize_octree(self, batch_update):
        """
        Reorganizes the octree by updating its structure.

        Args:
            batch_update (bool): Flag indicating whether to perform batch updates.

        Returns:
            bool: True if reorganization succeeds, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::reorganizeOctree")

        if batch_update:
            emp = 0
            if not self.batch_pull_up(0, emp):
                return False
            if not self.batch_push_down(0):
                return False
        else:
            for i in range(self.num_atoms):
                atom = self.atoms[i]
                if not atom.is_fixed():
                    self.update_octree(atom)

        return True

    def update_octree(self, obj):
        """
        Updates the octree structure with the given object.

        Args:
            obj (Object): The object to update the octree with.
        """
        if self.verbose:
            print("In DynamicOctree::updateOctree")

        node_id = self.get_node_id(obj)  # Modify the function call here
        node = self.nodes[node_id]

        if not self.inside_node(node, obj):
            self.pull_up(node_id, obj.id)

    def compute_score_recursive(self, octree_moving, static_node_id, moving_node_id, score_par, scores):
        """
        Recursively computes scores between static and moving octree nodes.

        Args:
            octree_moving (DynamicOctree): The moving octree.
            static_node_id (int): The ID of the static octree node.
            moving_node_id (int): The ID of the moving octree node.
            score_par (OctreeScoringParams): Scoring parameters.
            scores (OctreeScore): Scores container.

        Returns:
            bool: True if the computation is successful, False otherwise.
        """
        static_node = self.nodes[static_node_id]
        moving_node = octree_moving.nodes[moving_node_id]

        if moving_node.get_num_fixed() == moving_node.get_num_atoms():
            return True

        if score_par.get_trans_other() is None and score_par.get_trans_self() is None:
            if not self.within_distance_cutoff_untransformed(static_node, moving_node, score_par):
                return True

        if not self.within_distance_cutoff_transformed(static_node, moving_node, score_par):
            self.compute_far_score(static_node, moving_node, score_par, scores, octree_moving)
            return True

        if static_node.is_leaf():
            if moving_node.is_leaf():
                if self.verbose:
                    print(f"Computing score for static node {static_node_id} (with {static_node.get_num_atoms()} atoms), "
                        f"and moving node {moving_node_id} (with {moving_node.get_num_atoms()} atoms).")
                self.score(static_node, moving_node, score_par, scores, octree_moving)
            else:
                for j in range(8):
                    if moving_node.get_child_pointer(j) >= 0:
                        self.compute_score_recursive(octree_moving, static_node_id, moving_node.get_child_pointer(j),
                                                    score_par, scores)
        else:
            if moving_node.is_leaf():
                for i in range(8):
                    if static_node.get_child_pointer(i) >= 0:
                        self.compute_score_recursive(octree_moving, static_node.get_child_pointer(i), moving_node_id,
                                                    score_par, scores)
            else:
                for i in range(8):
                    if static_node.get_child_pointer(i) >= 0:
                        for j in range(8):
                            if moving_node.get_child_pointer(j) >= 0:
                                self.compute_score_recursive(octree_moving, static_node.get_child_pointer(i),
                                                            moving_node.get_child_pointer(j), score_par, scores)

        return True


    def compute_score(self, octree_moving, score_par, scores):
        """
        Computes scores between static and moving octrees.

        Args:
            octree_moving (DynamicOctree): The moving octree.
            score_par (OctreeScoringParams): Scoring parameters.
            scores (OctreeScore): Scores container.

        Returns:
            bool: True if computation is successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::computeScore")

        if not self.octree_built or not octree_moving.octree_built:
            print("Cannot compute scores. Octrees are not built yet.")
            return False

        return self.compute_score_recursive(octree_moving, 0, 0, score_par, scores)


    def score(self, static_node, moving_node, score_par, scores, moving_octree):
        """
        Computes the score between static and moving octree nodes.

        Args:
            static_node (DynamicOctreeNode): The static octree node.
            moving_node (DynamicOctreeNode): The moving octree node.
            score_par (OctreeScoringParams): Scoring parameters.
            scores (OctreeScore): Scores container.
            moving_octree (DynamicOctree): The moving octree.

        Returns:
            bool: True if scoring is successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::score")

        scores.set_score(scores.get_score() + 1)
        return True


    def compute_far_score(self, static_node, moving_node, score_par, scores, moving_octree):
        """
        Computes the score for nodes that are beyond the distance cutoff.

        Args:
            static_node (DynamicOctreeNode): The static octree node.
            moving_node (DynamicOctreeNode): The moving octree node.
            score_par (OctreeScoringParams): Scoring parameters.
            scores (OctreeScore): Scores container.
            moving_octree (DynamicOctree): The moving octree.

        Returns:
            bool: True if scoring is successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::computeFarScore")

        scores.set_score(scores.get_score() + 1)
        return True
