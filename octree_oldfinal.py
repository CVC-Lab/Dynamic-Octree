## TODO: 1. After creating INIT_NUM_OCTREE_NODES when we need them again, it becomes a new octree i.e. with parent id -1. CORRECT IT:
##       2. Is leaf or not, this issue needs to be clarified
## THESE MIGHT HELP SOLVE THE BIG QUESTION: Not all objects are considered while creating nb_lists

import sys, math, random
from objects import Object   # Import the Object class from the objects module
import numpy as np

INIT_NUM_OCTREE_NODES = 8  # Initial number of octree nodes
LOW_BITS = 14  # Low bits used for generating the Octree IDs

# Class for construction parameters of the octree
class OctreeConstructionParams:
    def __init__(self, max_leaf_size, max_leaf_dim, slack_factor=1.0):
        self.max_leaf_size = max_leaf_size
        self.max_leaf_dim = max_leaf_dim
        self.slack_factor = slack_factor

    # Setter methods
    def set_max_leaf_size(self, max_leaf_size):
        self.max_leaf_size = max_leaf_size ### CAN SET IT AS self.alpha * self.K

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
        self.atom_indices = []  # Indices of objects contained in the node
        self.parent_pointer = -1  # Pointer to the parent node
        self.child_pointer = [-1] * 8  # Pointers to child nodes
        self.leaf = False  # Flag indicating if the node is a leaf
        self.attribs = DynamicOctreeNodeAttr()  # Attributes of the node
        self.id = node_id  # Add id to the node for printing
    
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
        self.leaf = False
        self.id = None  # Add an id attribute to the node
        self.attribs = DynamicOctreeNodeAttr()
        
    def distance(self, other):
        center_self = np.array([self.lx + self.dim / 2, self.ly + self.dim / 2, self.lz + self.dim / 2])
        center_other = np.array([other.lx + other.dim / 2, other.ly + other.dim / 2, other.lz + other.dim / 2])
        return np.linalg.norm(center_self - center_other)
        
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
    
    def get_lx(self):
        return self.lx

    def get_ly(self):
        return self.ly

    def get_lz(self):
        return self.lz
    
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
    def __init__(self, atoms, n_atoms, cons_par, verbose=True, max_nodes=None, interaction_distance=20):
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
        self.nb_lists = [[] for _ in range(self.num_atoms)] # Initialize neighbourhood lists
        self.nb_lists_with_dist = [[] for _ in range(self.num_atoms+1)] # Initialize neighbourhood lists which stores distances as well
        self.interaction_distance = interaction_distance
        self.root_node_id = 0
        
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
        # print("OBJ: ", obj)
        # print("obj_to_node map: ", self.object_to_node_map)
        return self.object_to_node_map.get(obj)
    
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
    
    def print_children(self, node_id):
        """
        Print the children of the given node.

        Args:
            node_id: ID of the node.
        """
        # print("--------")
        print(f"Node_id: {node_id}, all nodes: {len(self.nodes)}, parent pointer of node_id: {self.nodes[node_id].parent_pointer}")
        children = [i for i, node in enumerate(self.nodes) if node is not None and node.parent_pointer == node_id]
        if self.verbose:
            print(f"Children of node {node_id}: {children}")

    def build_nb_lists(self, interaction_range):
        """
        Build neighborhood lists for all atoms within the specified interaction range.
        
        Args:
            interaction_range: The distance within which atoms are considered neighbors.
        """
        if self.verbose:
            print("\nBuilding neighbor lists...\n")
        # self.nb_lists = [[] for _ in range(self.num_atoms)]
        self._accum_inter(self.nodes[self.root_node_id], self.nodes[self.root_node_id], interaction_range)
        # for index, particle in enumerate(self.atoms):
        #     print(f"Processing atom {index} with position ({particle.x}, {particle.y}, {particle.z})")
        #     neighbours = self._find_neighbours(self, self, interaction_range)
        #     print(f"Found neighbors for particle {index}: {neighbours}")
        #     self.nb_lists[index].extend(neighbours)
        if self.verbose:
            print("Neighbor lists construction completed.\n")
        
    def _accum_inter(self, u, v, d):
        """
        Accumulate interactions between atoms in nodes u and v within the distance d.
        
        Args:
            u: First node.
            v: Second node.
            d: Interaction distance.
        """
        if u.distance(v) > d:
            return
        elif u.leaf and v.leaf:
            for p_idx in u.atom_indices:
                p = self.atoms[p_idx]
                for q_idx in v.atom_indices:
                    q = self.atoms[q_idx]
                    print(f"Build: Considering Atom {p_idx} and Atom {q_idx}")
                    if p != q and p.distance(q) <= d:
                        if q_idx not in self.nb_lists[p_idx]:
                            self.nb_lists[p_idx].append(q_idx)
                            self.nb_lists_with_dist[p_idx].append((q_idx, p.distance(q)))
                        if p_idx not in self.nb_lists[q_idx]:
                            self.nb_lists[q_idx].append(p_idx)
                            self.nb_lists_with_dist[q_idx].append((p_idx, p.distance(q)))
        elif u.leaf:
            for v_child_idx in v.child_pointer:
                if v_child_idx != -1:
                    print(f"Build: Considering Atom {p_idx} and Atom {q_idx}")
                    v_child = self.nodes[v_child_idx]
                    self._accum_inter(u, v_child, d)
        elif v.leaf:
            for u_child_idx in u.child_pointer:
                if u_child_idx != -1:
                    print(f"Build: Considering Atom {p_idx} and Atom {q_idx}")
                    u_child = self.nodes[u_child_idx]
                    self._accum_inter(u_child, v, d)
        else:
            for u_child_idx in u.child_pointer:
                if u_child_idx != -1:
                    u_child = self.nodes[u_child_idx]
                    for v_child_idx in v.child_pointer:
                        if v_child_idx != -1:
                            print(f"Build: Considering Atom {p_idx} and Atom {q_idx}")
                            v_child = self.nodes[v_child_idx]
                            self._accum_inter(u_child, v_child, d)

    def _distance(self, p1, p2):
        distance = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
        # print(f"\nCalculating distance between ({p1.id}) and ({p2.id}): {distance}\n")
        return distance
    
#---------------BUILD OCTREE---------------

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
        # print("Number of Atoms: ", self.num_atoms)
        indices_temp = [0] * self.num_atoms

        try:
            if not self.init_free_node_server():
                print("ERROR: Could not create free node server")
                return False

            octree_root = self.get_next_free_node()
            self.root_node_id = octree_root
            
            if self.verbose:
                print("\nRoot Node: ", octree_root)

            if octree_root == -1:
                print("ERROR: Could not get next node")
                return False

            # Compute root bounding box
            self.compute_root_bounding_box(octree_root, self.construction_params.get_slack_factor(), indices, 0, self.num_atoms - 1)

            self.nodes[octree_root].set_parent_pointer(-1)

            if self.verbose:
                print("Number of atoms considered while expanding octree: ", indices)
            # Expand octree node
            if self.num_nodes < self.max_nodes:
                self.octree_built = self.expand_octree_node(octree_root, indices, indices_temp, 0, self.num_atoms - 1)
            
            if self.verbose:
                print("\nobject to node map after expanding the octree: ", self.object_to_node_map)
                print("\n")
            
                for i in range(self.num_nodes):
                    self.print_children(i)
            
                print("-----------Atoms in the respective Nodes------------")
                for i in range(self.num_nodes):
                    print(f"The indices of the atoms in Node {i}: {self.nodes[i].atom_indices}")
                    
            # Update neighborhood lists for all atoms
            for atom_index in range(self.num_atoms):
                node = self.get_node_containing_point(self.atoms[atom_index])
                self.update_nb_lists(atom_index, node)
                
            # Ensure no None nodes are left behind
            # self.nodes = [node for node in self.nodes if node is not None]
                                
        finally:
            # Free temporary storage
            del indices
            del indices_temp

        return self.octree_built
    
    def init_free_node_server(self):
        """
        Initialize the free node server. 
        The initialization process in the init_free_node_server function is not 
        intended to set up a complete octree structure with parent-child relationships. 
        Instead, it's primarily focused on initializing a pool of free nodes 
        that can be used later during octree construction.

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
        # for i in range(self.num_nodes):
            self.nodes[i].set_parent_pointer(self.next_free_node)
            self.next_free_node = i

        if self.verbose:
            print("Allocated {} new nodes".format(INIT_NUM_OCTREE_NODES))
        
        # Print parent nodes
        if self.verbose:
            print("Parent Nodes for the Initially Allocated Nodes:")
        for i in range(self.num_nodes):
            parent_node_id = self.nodes[i].get_parent_pointer()
            if self.verbose: #and parent_node_id != -1:
                print(f"Node ID: {i}, Parent Node ID: {parent_node_id}")

        return True

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

        # print("next free node: ", self.next_free_node)
        next_node = self.next_free_node

        self.next_free_node = self.nodes[next_node].get_parent_pointer()

        # if self.verbose:
        #     print("Next node is", next_node)

        return next_node
    
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

        # if self.verbose:
            # print("Allocated {} nodes".format(new_num_nodes))

        return True
    
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

            minX = -5 # min(minX, self.atoms[j].getX())
            maxX = 20 # max(maxX, self.atoms[j].getX())

            minY = -5 # min(minY, self.atoms[j].getY())
            maxY = 20 # max(maxY, self.atoms[j].getY())

            minZ = -5 # min(minZ, self.atoms[j].getZ())
            maxZ = 20 # max(maxZ, self.atoms[j].getZ())

        cx, cy, cz = (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2

        dim = max(maxX - minX, maxY - minY, maxZ - minZ)
        dim *= slack_factor

        node.set_lx(cx - dim * 0.5)
        node.set_ly(cy - dim * 0.5)
        node.set_lz(cz - dim * 0.5)

        node.set_dim(dim)
        
        if self.verbose:
            print(f"Node {node_id}: Center=({cx}, {cy}, {cz}), Dimension={dim}")

    def expand_octree_node(self, node_id, indices, indices_temp, start_id, end_id):
        """
        TODO: Nodes with empty atom lists should be contracted if necessary, HOW TO DO THAT?
        
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
        # print("Start and End IDs: ", start_id, end_id)
        nAtoms = end_id - start_id + 1
        node.set_num_atoms(nAtoms)
        # print("nAtoms: ", nAtoms)
        # self.compute_leaf_attributes(node_id, indices, start_id, end_id)
        # dim = node.get_dim()
        
        # If the node is the root node, initialize its attributes
        if node_id == self.root_node_id:
            if self.verbose:
                print(f"Node {node_id} is the root node")
            node.leaf = False
            node.set_parent_pointer(-1)  # Root node has no parent
            # node.set_child_pointer(node_id, [-1] * 8)  # Initialize child pointers
            
        # Print the children of the current node
        # self.print_children(node_id)

        # Print the initial state of the node
        if self.verbose:
            print(f"Node {node_id}, parent={node.get_parent_pointer()}, atoms={node.atom_indices}")
        
        # The node is a leaf. If the atom is fixed, it is placed at the beginning of the list of atom indices.
        # print(self.needs_expansion(node))
        if not self.needs_expansion(node):
            if self.verbose:
                print(f"Node {node_id} is a leaf Node")
            node.set_leaf(True)
            # new_indices = [-1] * (2 * nAtoms)
            new_indices = [-1] * nAtoms # Initialize with -1 for safety
            
            if new_indices is None:
                if self.verbose:
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

            # Remove any -1 entries
            # node.atom_indices = [idx for idx in node.atom_indices if idx != -1]
                        
            # Update neighborhood lists for the atoms in this leaf node
            # for i in range(start_id, end_id + 1):
            #     atom_index = indices[i]
            #     self.update_nb_lists(atom_index, node)

        # If the node is not a leaf (i.e., it needs expansion), it proceeds to subdivide the node.
        
        #### elif nAtoms > self.K / self.alpha:    #### Internal node condition: Each internal node should have more than K/α points
        
        else:
            if self.verbose:
                print(f"Node {node_id} is not a leaf Node")
            node.leaf = False
            count = [0] * 8
            for i in range(start_id, end_id + 1):
                j = indices[i]
                k = self.get_child_id(node, self.atoms[j])
                count[k] += 1

            # print("count: ", count)
            start_index = [0] * 8
            cur_index = [0] * 8
            cur_index[0] = start_index[0] = start_id

            for i in range(1, 8):
                cur_index[i] = start_index[i] = start_index[i - 1] + count[i - 1]
                # start_index[i] = start_index[i - 1] + count[i - 1]
                # cur_index[i] = start_index[i]

            # Distribute indices into temporary list
            for i in range(start_id, end_id + 1):
                j = indices[i]
                k = self.get_child_id(node, self.atoms[j])
                indices_temp[cur_index[k]] = j
                cur_index[k] += 1
            
            # print("\n\n\n ===============Indices temp: ", indices_temp)
            # print(self.object_to_node_map)
            nfxd = 0
            
            # Create child nodes and recursively expand them
            for i in range(8):
                # print(f"---------------{i}-------------")
                if count[i] > 0:
                    j = self.get_next_free_node()
                    # print("J: ", j)
                    node.set_child_pointer(i, j)
                    self.nodes[j].set_parent_pointer(node_id)
                    self.compute_non_root_bounding_box(j, i)

                    # Update object_to_node_map for each atom in the child node
                    for k in range(start_index[i], start_index[i] + count[i]):
                        atom_index = indices_temp[k]
                        self.object_to_node_map[self.atoms[atom_index]] = j
                        # print("-=-=-=-=-=-=-=-=", atom_index)
                        # print(self.atoms[atom_index])
                        self.nodes[j].atom_indices.append(atom_index)
                        
                    # print("object to node map after expanding the octree: ", self.object_to_node_map)
                    
                    # Print the state after setting the parent pointer
                    if self.verbose:
                        print(f"Node {j} created with parent {node_id}")

                    if self.num_nodes < self.max_nodes:
                        if not self.expand_octree_node(j, indices_temp, indices, start_index[i], start_index[i] + count[i] - 1):
                            return False

                    nfxd += self.nodes[j].n_fixed
                    node.set_num_fixed(nfxd)
                    
                    node.atom_indices = []
                    
                    # Print the children of the current node
                    # self.print_children(j)

                else:
                    node.set_child_pointer(i, -1) 

        # Remove any -1 entries
        node.atom_indices = [idx for idx in node.atom_indices if idx != -1]
        
        # Collapse empty child nodes
        # for i in range(8):
        #     child_id = node.child_pointer[i]
        #     if child_id != -1 and self.nodes[child_id].num_atoms == 0:
        #         node.set_child_pointer(i, -1)
        #         self.nodes[child_id] = None  # Remove the empty node
                # self.num_nodes -= 1
                
        # Since we preallocate space in the self.nodes list for potential nodes, some of these slots might initially be set to None until they are actually filled with a DynamicOctreeNode instance.
        # Ensure no None nodes are left behind
        # self.nodes = [node for node in self.nodes if node is not None]

        # Update object_to_node_map for each atom in the expanded node
        # print("\nThe node_id considered while mapping: ", node_id)
        # print("\n")
        # for i in range(start_id, end_id + 1):
        #     j = indices[i]
        #     self.object_to_node_map[self.atoms[j]] = node_id
        
        # print("\nobject to node map after expanding the octree: ", self.object_to_node_map)
        # print("\n")
        
        return True
    
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

        # print("prepared list of atoms")

        self.nodes[node_id].compute_own_attribs(node_atoms)
        
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
        # print("In:",node.num_atoms <= self.construction_params.get_max_leaf_size())
        # print("in:", node.dim <= self.construction_params.get_max_leaf_dim())
        return not (node.num_atoms <= self.construction_params.get_max_leaf_size()) # or node.dim <= self.construction_params.get_max_leaf_dim())
    
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
            print(f"\nNode {node_id}: Center=({lx + dim * 0.5}, {ly + dim * 0.5}, {lz + dim * 0.5}), Dimension={dim}")

#---------------INSERTION-------------

    def insert_object(self, new_object):
        """
        Insert a new object into the octree.

        Args:
            new_object (Object): The object to be inserted.
        """
        if self.verbose:
            print(f"Inserting new object at position: {new_object.position}")

        # Step 1: Find the node where the object should be inserted
        target_node = self.get_node_containing_point(new_object)

        # Step 2: If the target node is None, find the nearest bounding ancestor
        if target_node is None:
            ancestor_node = self.nearest_bounding_ancestor(self.root_node_id, new_object)
            target_node = self.furthest_bounding_descendant(ancestor_node, new_object)

        # Step 3: Insert the object into the target node
        self.add_point_to_node(target_node, new_object)

        # Step 5: Update neighborhood lists
        self.update_nb_lists_local(self.atoms.index(new_object), target_node)

        if self.verbose:
            print(f"New object inserted into node: {self.nodes.index(target_node)}")

#---------------DELETION---------------

    def delete_atom(self, atom):
        """
        Deletes an atom from the octree.

        Args:
            atom (Object): The atom to delete.
        """
        if self.verbose:
            print("In DynamicOctree::deleteAtom")
        
        # Step 1: Locate the node containing the atom
        current_node = self.get_node_containing_point(atom)
        
        if current_node is None:
            if self.verbose:
                print("Atom not found in any node.")
            return False

        node_id = self.nodes.index(current_node)
        atom_id = self.atoms.index(atom)
        
        # Step 2: Remove the atom from the node
        if current_node.is_leaf():
            success = self.remove_atom_from_leaf(node_id, atom_id)
        else:
            success = self.remove_atom_from_non_leaf(node_id, atom_id)
        
        if not success:
            if self.verbose:
                print("Failed to remove atom.")
            return False
        
        # Step 3: Contract the octree if necessary
        contraction_node_id = self.find_furthest_proper_ancestor(current_node)
        if contraction_node_id and self.is_within_subtree(self.nodes[self.root_node_id], contraction_node_id):
            self.contract_octree_node(contraction_node_id)
        
        # Update the object to node map
        if atom in self.object_to_node_map:
            del self.object_to_node_map[atom]
        
        if self.verbose:
            print("Atom deleted successfully.")
        
        return True

#---------------UPDATION---------------

    def update_octree(self, atom, new_position):
        """
        Updates the octree to reflect the new position of the given atom.

        Args:
            atom (Object): The atom whose position has changed.
            new_position (tuple): The new coordinates of the atom.
        """
        if self.verbose:
            print("In DynamicOctree::updateOctree")
            print(f"\n===Atom to update: {self.atoms.index(atom)}===\n")
            
        # Step 1: Update the atom's position
        atom.set_position(new_position)
        
        # print(f"Updated Object {self.atoms.index(atom)} Position")
        
        # Step 2: Find the current node containing the atom
        current_node = self.get_node_containing_point(atom)
        if self.verbose:
            print(f"current node: {self.nodes.index(current_node)}, atoms in the current node: {current_node.atom_indices}")
            
        # Step 3: Remove the atom from the current node
        self.remove_point_from_node(current_node, atom)
        
        # Step 4: Find the nearest ancestor that can contain the new position: if it is in the same node, well and good else check its parent
        ancestor_node = self.nearest_bounding_ancestor(current_node, atom) 
        if self.verbose:
            print(f"Ancestor node: {self.nodes.index(ancestor_node)}")
            
        # Step 5: Find the furthest descendant of the ancestor that can contain the new position: check if any of the children of the ancestor has it or else, return the ancestor
        target_node = self.furthest_bounding_descendant(ancestor_node, atom)
        if self.verbose:
            print(f"Target node: {self.nodes.index(target_node)}")
            
        # Step 6: Add the atom to the target node
        self.add_point_to_node(target_node, atom)

        ###### a) remove the node if the node becomes empty after removing the atom
        ###### b) see if it gets better after using old_atom, new_atom

        # Step 7: Expand the node if necessary (Locally within the subtree T_i)
        # Only expand if the operation can be confined within the local subtree
        if self.is_within_subtree(ancestor_node, target_node):
            self.expand_node_if_needed(target_node)
        
       # Step 8: Contract nodes if necessary (locally within the subtree T_i)
        contraction_node = self.find_furthest_proper_ancestor(target_node)
        if contraction_node and self.is_within_subtree(ancestor_node, contraction_node):
            self.contract_octree_node(self.nodes.index(contraction_node))
            
        # self.print_all_atoms_in_nodes()

        # Step 9: Update neighborhood lists (locally within the subtree T_i)
        self.update_nb_lists_local(self.atoms.index(atom), target_node)
        
        if self.verbose:
            print("\nobject to node map after updating the octree: ", self.object_to_node_map)

    def print_all_atoms_in_nodes(self):
        """
        Prints all the atoms in all the nodes of the octree.
        """
        print("\n============Printing all atoms in all nodes of the octree============")

        # Iterate over each node in the octree
        for node_index, node in enumerate(self.nodes):
            if node is not None:  # Ensure the node exists
                print(f"\nNode {node_index} with parent as Node {self.nodes[node_index].parent_pointer} and is leaf:{self.nodes[node_index].is_leaf()}")
                if node.atom_indices:
                    print(f"  Atom indices: {node.atom_indices}")
                    for atom_index in node.atom_indices:
                        atom = self.atoms[atom_index]
                        print(f"    Atom {atom_index}: Position ({atom.getX()}, {atom.getY()}, {atom.getZ()})")
                else:
                    print("  No atoms in this node")
            # print()
        print()
        
    # def get_node_containing_point(self, p):
    #     """
    #     Gets the node containing the specified point using a depth-first search (DFS) approach.

    #     Args:
    #         p (Object): The point to find.

    #     Returns:
    #         DynamicOctreeNode: The node containing the point.
    #     """
    #     def dfs(node):
    #         if self.verbose:
    #             print(f"Visiting node ID: {self.nodes.index(node)}, is_leaf: {node.is_leaf()}")

    #         # Base case: if the current node is a leaf and contains the point, return it
    #         if node.is_leaf() and self.inside_node(node, p):
    #             return node
            
    #         # Explore all child nodes in a DFS manner
    #         for child_id in node.child_pointer:
    #             if child_id != -1 and self.nodes[child_id] is not None:
    #                 child_node = self.nodes[child_id]
    #                 if self.inside_node(child_node, p):
    #                     result = dfs(child_node)
    #                     if result:
    #                         return result

    #         # If the point isn't found in any children, but the current node contains it, return this node
    #         if self.inside_node(node, p):
    #             return node
            
    #         return None

    #     if self.verbose:
    #         print("In DynamicOctree::get_node_containing_point using DFS")

    #     # Start DFS from the root node
    #     root_node = self.nodes[0]
        
    #     if self.verbose:
    #         print(f"Starting DFS at root node, child pointers: {[idx for idx in root_node.child_pointer if idx != -1]}")
        
    #     result_node = dfs(root_node)

    #     if result_node is None:
    #         if self.verbose:
    #             print(f"Atom {self.atoms.index(p)} not found in any child of root node")
    #         return root_node  # Return the root node as a fallback if the point is not found

    #     return result_node

    def get_node_containing_point(self, p):
        """
        Gets the node containing the specified point.

        Args:
            p (Object): The point to find.

        Returns:
            DynamicOctreeNode: The node containing the point.
        """
        def find_node(node):
            if self.verbose:
                print(f"In DynamicOctree::find_node, current node ID: {self.nodes.index(node)}")
            
            if node.is_leaf():
                return node
            
            if self.verbose:
                print(f"Child pointers of the current node: {[idx for idx in node.child_pointer if idx != -1]}")
            
            for child_id in node.child_pointer:
                if child_id != -1:
                    # print(child_id)
                    child_node = self.nodes[child_id]
                    if self.inside_node(child_node, p):
                        return find_node(child_node)
            
            return None

        if self.verbose:
            print("In DynamicOctree::get_node_containing_point")
        
        root_node = self.nodes[0] # self.nodes[root_node_id]
        
        if self.verbose:
            print(f"Starting at root node, child pointers: {[idx for idx in root_node.child_pointer if idx != -1]}")
        
        result_node = find_node(root_node)

        if result_node is None:
            if self.verbose:
                print(f"Atom {self.atoms.index(p)} not found in any child of root node")
            return root_node

        return result_node
        
    def inside_node(self, node, obj):
        """
        Check if the object is inside the given node.

        Args:
            node (DynamicOctreeNode): The node to check.
            obj (Object): The object to check.

        Returns:
            bool: True if the object is inside the node, False otherwise.
        """
        if self.verbose:
            print(f"atom's X: {obj.getX()}, Y: {obj.getY()}, Z: {obj.getZ()}; Node's X: {node.get_lx()}, Y: {node.get_ly()}, Z: {node.get_lz()}, dim: {node.get_dim()}")
        return (obj.getX() - node.get_lx() >= 0 and obj.getX() - node.get_lx() <= node.get_dim() and
                obj.getY() - node.get_ly() >= 0 and obj.getY() - node.get_ly() <= node.get_dim() and
                obj.getZ() - node.get_lz() >= 0 and obj.getZ() - node.get_lz() <= node.get_dim())
        
    def remove_point_from_node(self, node, p):
        """
        Removes the point from the specified node.

        Args:
            node (DynamicOctreeNode): The node to remove the point from.
            p (Object): The point to remove.
        """
        try:
            print(f"Atom indices: {p}, {self.atoms.index(p.id)}, {p.id}")
            
            index_in_node = self.get_index_in_node(self.nodes.index(node), p.id)
            
            # Ensure that the object ID is valid and present in self.atoms
            atom_index = self.atoms.index(p.id)
            if node.is_leaf:
                self.remove_atom_from_leaf(self.nodes.index(node), atom_index)
            else:
                self.remove_atom_from_non_leaf(self.nodes.index(node), atom_index)
                
        except ValueError as e:
            if self.verbose:
                print(f"Error removing point: {e}")
                print(f"Object ID: {p.id}")
                # print(f"Atoms list: {self.atoms}")
                print(f"Node index: {self.nodes.index(node)}")
                print(f"Atoms in the current node: {node.atom_indices}")

            
    def get_index_in_node(self, node_id, atom_id):
        """
        Get the index of the atom within the node's atom indices list.

        Args:
            atom_id (int): The ID of the atom.

        Returns:
            int: The index of the atom within the node's atom indices list, or -1 if not found.
        """
        node = self.nodes[node_id]
        for index, atom_index in enumerate(node.atom_indices):
            if atom_index == atom_id:
                return index
        return -1

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
        node = self.nodes[node_id]
        
        try:
            j = node.atom_indices.index(atom_id)
        except ValueError:
            # Atom is not present in the leaf node
            if self.verbose:
                print("Atom is not present in the leaf node.")
            return False
        
        n = len(node.atom_indices)

        if atom.is_fixed():
            nf = node.num_fixed - 1
            node.num_fixed = nf
            
            if j < nf:
                # Swap the atom to be removed with the last fixed atom
                node.atom_indices[j], node.atom_indices[nf] = node.atom_indices[nf], node.atom_indices[j]
                self.atoms[node.atom_indices[j]].id = self.create_octree_ptr(node_id, j)
            
            # Swap the last fixed atom with the last atom
            node.atom_indices[nf], node.atom_indices[n - 1] = node.atom_indices[n - 1], node.atom_indices[nf]
            self.atoms[node.atom_indices[nf]].id = self.create_octree_ptr(node_id, nf)

        else:
            # Swap the atom to be removed with the last atom
            # print(f"len of atom indices: {len(node.atom_indices)}")
            # print(f"atom_indices j: {j}, atom_indices n-1: {n-1}")
            node.atom_indices[j], node.atom_indices[n - 1] = node.atom_indices[n - 1], node.atom_indices[j]
            self.atoms[node.atom_indices[j]].id = self.create_octree_ptr(node_id, j)

        # Remove the mapping from the object to node map
        if self.atoms[atom_id] in self.object_to_node_map:
            del self.object_to_node_map[self.atoms[atom_id]]
        
        # Update node_id attribute of the removed atom
        self.atoms[atom_id].node_id = None
        node.atom_indices.pop()
        node.num_atoms -= 1

        if n <= (node.id_cap >> 2) and node.id_cap > 1:
            new_indices = node.atom_indices[:node.id_cap >> 1]

            if new_indices is None:
                if self.verbose:
                    print("Failed to contract leaf storage for octree!")
                return False
            
            node.atom_indices = new_indices
            node.id_cap = node.id_cap >> 1

        node.update_attribs(atom, False)
        atom.id = -1
        
        if self.verbose:
            print("\n====================================")
            print("Number of atoms: ", self.num_atoms)
            print("Number of Nodes: ", self.num_nodes)

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
        
        try:
            j = node.atom_indices.index(atom_id)
        except ValueError:
            # Atom is not present in the node
            if self.verbose:
                print("Atom is not present in the non-leaf node.")
            return False

        node.num_atoms -= 1

        if atom.is_fixed():
            node.num_fixed -= 1

        # Remove the atom from the node's atom_indices list
        node.atom_indices[j] = node.atom_indices[-1]
        node.atom_indices.pop()

        # Update the node_id attribute of the removed atom to None
        atom.node_id = None

        # Remove the atom from the object_to_node_map dictionary
        del self.object_to_node_map[self.atoms[atom_id]]
        
        # Update the object_to_node_map
        if self.atoms[atom_id] in self.object_to_node_map:
            del self.object_to_node_map[self.atoms[atom_id]]

        # Check if the node needs dynamic contraction and contract if necessary
        if self.needs_dynamic_contraction(node):
            return self.contract_octree_node(node_id)

        return True
            
    def nearest_bounding_ancestor(self, u, p):
        """
        Finds the nearest ancestor of octree node u containing location p.  ### PREVENT THIS FROM GOING INTO INFINITE LOOP
  
        Args:
            u (DynamicOctreeNode): The octree node.
            p (Object): The point to check.

        Returns:
            DynamicOctreeNode: The nearest ancestor node containing point p.
        """
        if self.verbose:
            print("In DynamicOctree::nearest_bounding_ancestor")
        if self.inside_node(u, p):
            if self.verbose:
                print(f"Inside Node: {self.nodes.index(u)}, object_id: {self.atoms.index(p)}")
            return u
        else:
            if self.verbose:
                print(f"Node: {u.parent_pointer}, object_id: {self.atoms.index(p)}")
            # return
            return self.nearest_bounding_ancestor(self.nodes[u.parent_pointer], p)
        
    def furthest_bounding_descendant(self, u, p):
        """
        Finds the furthest descendant of octree node u containing location p.

        Args:
            u (DynamicOctreeNode): The octree node.
            p (Object): The point to check.

        Returns:
            DynamicOctreeNode: The furthest descendant node containing point p.
        """
        if self.verbose:
            print("In DynamicOctree::furthest_bounding_descendant")
        for child_id in u.child_pointer:
            if child_id != -1:
                v = self.nodes[child_id]
                if v is not None and self.inside_node(v, p):
                    return self.furthest_bounding_descendant(v, p)
        return u
    
    def add_point_to_node(self, node, p):
        """
        Adds the point to the specified node.

        Args:
            node (DynamicOctreeNode): The node to add the point to.
            p (Object): The point to add.
        """
        # node.atom_indices.append(p.id)
        # node.num_atoms += 1
        try:
            # index_in_node = self.get_index_in_node(self.nodes.index(node), p.id)
            
            # Ensure that the object ID is valid and present in self.atoms
            atom_index = self.atoms.index(p)

            if node.is_leaf:
                self.add_atom_to_leaf(self.nodes.index(node), atom_index)
            else:
                self.add_atom_to_non_leaf(self.nodes.index(node), atom_index)
                
        except ValueError as e:
            if self.verbose:
                print(f"Error adding point: {e}")
                print(f"Object ID: {p.id}")
                # print(f"Atoms list: {self.atoms}")
                print(f"Node index: {self.nodes.index(node)}")
                print(f"Atoms in the current node: {node.atom_indices}")
                
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
        node.atom_indices.append(atom_id)

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
        # if atom_id >= len(self.atoms):
        #     self.atoms += [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(atom_id - len(self.atoms) + 1)]

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
        node.atom_indices.append(atom_id)

        # Check if dynamic expansion is needed and expand if necessary
        if self.needs_expansion(node):
            temp = [None] * node.num_atoms

            if temp is None:
                print("Failed to allocate temporary storage for octree!")
                return False

            print("\n====================================")
            print("Number of atoms: ", self.num_atoms)
            print("Number of Nodes: ", self.num_nodes)
            # print("Node ID: ", node_id)
            done = self.expand_octree_node(node_id, node.atom_indices, temp, 0, node.num_atoms - 1)

            return done
        else:
            return True
        
    def expand_node_if_needed(self, node):
        # node = self.nodes[node_id]
        if self.needs_expansion(node):
            indices = node.atom_indices
            indices_temp = [-1] * len(indices)
            start_id = 0
            end_id = len(indices) - 1
            self.expand_octree_node(self.nodes.index(node), indices, indices_temp, start_id, end_id)
            
    def find_furthest_proper_ancestor(self, node):
        """
        Finds the furthest proper ancestor of the given node with N(u) < K/2.

        Args:
            node (DynamicOctreeNode): The node to find the ancestor for.

        Returns:
            DynamicOctreeNode: The furthest proper ancestor.
        """
        current_node = node
        while current_node.parent_pointer >= 0:
            parent_node = self.nodes[current_node.parent_pointer]
            if parent_node.num_atoms < (self.construction_params.max_leaf_size >> 1):
                return parent_node
            current_node = parent_node
        return None
    
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

        if not self.needs_contraction(node):
            return True
        
        self.compute_non_leaf_attributes(node_id)

        # Step 1: Traverse the subtree rooted at u, and copy all points from Tu to u
        collected_atoms = []
        self.collect_atoms_from_subtree(node_id, collected_atoms)
        
        # Separate fixed and non-fixed atoms
        fixed_atoms = [atom for atom in collected_atoms if self.atoms[atom].is_fixed()]
        non_fixed_atoms = [atom for atom in collected_atoms if not self.atoms[atom].is_fixed()]
        
        # Step 2: Delete the subtree Tu
        self.delete_subtree(node_id)
        
        # Step 3: Mark u as a leaf and add all collected points to it
        node.leaf = True
        node.atom_indices = fixed_atoms + non_fixed_atoms
        node.num_atoms = len(node.atom_indices)
        
        # Update atom IDs and node attributes
        for i, atom_index in enumerate(node.atom_indices):
            self.atoms[atom_index].set_id(self.create_octree_ptr(node_id, i))
            
        # Update neighborhood lists for atoms in the contracted node
        for atom_index in node.atom_indices:
            self.update_nb_lists(atom_index, node)

        return True
    
    def needs_contraction(self, node):
        """
        Checks whether a node needs contraction based on the maximum leaf size and dimension.

        Args:
            node (DynamicOctreeNode): The node to check.

        Returns:
            bool: True if the node needs contraction, False otherwise.
        """
        # Total number of atoms in the node and its children
        total_atoms = len(node.atom_indices)
        if not node.is_leaf():
            for child_pointer in node.child_pointers:
                if child_pointer != -1:
                    total_atoms += self.nodes[child_pointer].num_atoms
        
        # Check if the total number of atoms is less than or equal to the maximum leaf size
        if total_atoms <= self.construction_params.get_max_leaf_size():
            return True
        
        # Check if the dimension of the node and its children are within the maximum leaf dimension
        if node.dim <= self.construction_params.get_max_leaf_dim():
            if not node.is_leaf():
                for child_pointer in node.child_pointers:
                    if child_pointer != -1 and self.nodes[child_pointer].dim > self.construction_params.get_max_leaf_dim():
                        return False
            return True

        return False
    
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
        
    def collect_atoms_from_subtree(self, node_id, collected_atoms):
        """
        Recursively collects all atoms from the subtree rooted at node_id.

        Args:
            node_id (int): The ID of the node to start collection from.
            collected_atoms (list): List to collect atoms into.
        """
        node = self.nodes[node_id]
        if node.leaf:
            collected_atoms.extend(node.atom_indices)
        else:
            for child_id in node.children:
                self.collect_atoms_from_subtree(child_id, collected_atoms)
                
    def delete_subtree(self, node_id):
        """
        Recursively deletes the subtree rooted at node_id.

        Args:
            node_id (int): The ID of the node to delete.
        """
        node = self.nodes[node_id]
        if not node.leaf:
            for child_id in node.children:
                self.delete_subtree(child_id)
        # Finally delete this node
        self.nodes[node_id] = None  
        
    def is_within_subtree(self, ancestor_node, target_node):
        """
        Checks if a given target node is within the same subtree as the ancestor node.

        Args:
            ancestor_node (Node): The ancestor node.
            target_node (Node): The target node.

        Returns:
            bool: True if target_node is within the subtree rooted at ancestor_node.
        """
        # Check if the target node is within the subtree rooted at ancestor_node
        # print(f"Ancester node: {ancestor_node}, target_node: {target_node}")
        return self.is_ancestor_of(ancestor_node, target_node)
    
    def is_ancestor_of(self, ancestor_node, target_node):
        """
        Checks if the ancestor_node is an ancestor of the target_node.

        Args:
            ancestor_node (DynamicOctreeNode): The potential ancestor node.
            target_node (DynamicOctreeNode): The target node.

        Returns:
            bool: True if ancestor_node is an ancestor of target_node.
        """
        current_node = target_node
        while current_node.parent_pointer != -1:  # Traverse until the root node
            if current_node.parent_pointer == self.nodes.index(ancestor_node):
                return True
            current_node = self.nodes[current_node.parent_pointer]  # Move up to the parent node
        return False

    def update_nb_lists_local(self, atom_index, target_node):
        """
        Updates the neighborhood lists locally within the subtree T_i.

        Args:
            atom_index (int): The index of the atom to update.
            target_node (Node): The node where the atom is now located.
        """
        # Restrict neighborhood list updates to the local subtree
        # (This assumes `update_nb_lists` is modified to support local updates)
        self.update_nb_lists(atom_index, target_node, local_only=False)
        
    def update_nb_lists(self, atom_index, node, local_only=False):
        """
        Update neighborhood lists for a given atom index within the specified node.
        
        Args:
            atom_index: Index of the atom to update neighborhood lists for.
            node: The node in which the atom resides.
        """
        if self.verbose:
            print(f"\nIn update_nb_lists::With Atom {atom_index} and node {self.nodes.index(node)}")
            
        atom = self.atoms[atom_index]
        # stack = [self.nodes[self.root_node_id]]
        
        # Determine starting node for neighborhood search
        start_nodes = [node] if local_only else [self.nodes[self.root_node_id]]
        
        # Use sets to avoid duplicates
        neighbors_with_dist = set()
        neighbors = set()

        for start_node in start_nodes:
            stack = [start_node]
            while stack:
                current_node = stack.pop()
                if current_node is None:
                    continue  # Skip None nodes
                
                # print(f"Processing node: {self.nodes.index(current_node)}")
                if self.verbose:
                    print(f"Processing node: {self.nodes.index(current_node)}")
                    print(f"Node {self.nodes.index(current_node)} is away from node {self.nodes.index(node)} with a distance of {current_node.distance(node)}")
                
                if current_node.distance(node) > self.interaction_distance:
                    # print(f"Node {self.nodes.index(current_node)} is too far away from node {self.nodes.index(node)} with a distance of {current_node.distance(node)}")
                    if self.verbose:
                        print(f"Node {self.nodes.index(current_node)} is too far away from node {self.nodes.index(node)} with a distance of {current_node.distance(node)}")
                    continue

                if current_node.leaf:
                    for other_index in current_node.atom_indices:
                        if other_index is not None and other_index != atom_index:
                            other_atom = self.atoms[other_index]
                            # print(f"Update: Considering Atom {atom_index} and Atom {other_index}")
                            distance = atom.distance(other_atom)
                            if distance <= self.interaction_distance:
                                neighbors.add(other_index)
                                neighbors_with_dist.add((other_index, distance))
                                # Ensure symmetry
                                self.add_to_neighborhood(other_index, atom_index, distance)
                else:
                    for child_idx in current_node.child_pointer:
                        if child_idx != -1 and child_idx < len(self.nodes) and self.nodes[child_idx] is not None:
                            # Only add child nodes that are part of the local subtree or overlap with it
                            # print(f"Update: Considering Atom {atom_index} and Atom {child_idx}")
                            if not local_only or self.is_within_subtree(node, self.nodes[child_idx]):
                                stack.append(self.nodes[child_idx])

        # Convert sets back to lists
        self.nb_lists_with_dist[atom_index] = list(neighbors_with_dist)
        self.nb_lists[atom_index] = list(neighbors)

        if self.verbose:
            print(f"Updated neighborhood lists for atom {atom_index}")
            print(f"Neighbors with distance: {self.nb_lists_with_dist[atom_index]}")
            print(f"Neighbors: {self.nb_lists[atom_index]}")

    def add_to_neighborhood(self, atom_index_1, atom_index_2, distance):
        """
        Add atom_index_2 to the neighborhood list of atom_index_1 and ensure symmetry.
        
        Args:
            atom_index_1: The index of the first atom.
            atom_index_2: The index of the second atom to add to the neighborhood list.
            distance: The distance between the two atoms.
        """
        if atom_index_2 not in self.nb_lists[atom_index_1]:
            self.nb_lists[atom_index_1].append(atom_index_2)
            self.nb_lists_with_dist[atom_index_1].append((atom_index_2, distance))
            
        if atom_index_1 not in self.nb_lists[atom_index_2]:
            self.nb_lists[atom_index_2].append(atom_index_1)
            self.nb_lists_with_dist[atom_index_2].append((atom_index_1, distance))