# octree.pyx
cdef class Object:
    cdef public double x, y, z
    cdef public bint fixed
    cdef public int id
    cdef public int node_id
    cdef public object data

from objects cimport Object

import sys
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

cdef int INIT_NUM_OCTREE_NODES = 5
cdef int LOW_BITS = 14

cdef class OctreeScore:
    cdef public double score

    def __init__(self):
        self.score = 0.0

    cpdef void set_score(self, double d):
        self.score = d

    cpdef double get_score(self):
        return self.score

cdef class OctreeConstructionParams:
    cdef public int max_leaf_size
    cdef public int max_leaf_dim
    cdef public double slack_factor

    def __init__(self, int max_leaf_size=5, int max_leaf_dim=6, double slack_factor=1.0):
        self.max_leaf_size = max_leaf_size
        self.max_leaf_dim = max_leaf_dim
        self.slack_factor = slack_factor

    def set_max_leaf_size(self, int max_leaf_size):
        self.max_leaf_size = max_leaf_size

    def set_max_leaf_dim(self, int max_leaf_dim):
        self.max_leaf_dim = max_leaf_dim

    def set_slack_factor(self, double slack_factor):
        self.slack_factor = slack_factor

    cpdef int get_max_leaf_size(self):
        return self.max_leaf_size

    cpdef int get_max_leaf_dim(self):
        return self.max_leaf_dim

    cpdef double get_slack_factor(self):
        return self.slack_factor

    cpdef void print_params(self):
        print(f"Max leaf size: {self.max_leaf_size}, and max leaf dim: {self.max_leaf_dim}.")

cdef class DynamicOctreeNodeAttr:
    cdef public double sx, sy, sz

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self.sx = x
        self.sy = y
        self.sz = z

    cpdef void combine_s(self, list all_attribs):
        cdef DynamicOctreeNodeAttr attr
        for attr in all_attribs:
            self.sx += attr.sx
            self.sy += attr.sy
            self.sz += attr.sz

    cpdef void compute_s(self, list atoms):
        cdef Object atom
        for atom in atoms:
            self.sx += atom.x
            self.sy += atom.y
            self.sz += atom.z

    cpdef void update_s(self, Object atm, bint add):
        if add:
            self.sx += atm.x
            self.sy += atm.y
            self.sz += atm.z
        else:
            self.sx -= atm.x
            self.sy -= atm.y
            self.sz -= atm.z

cdef class DynamicOctreeNode:
    cdef public double lx, ly, lz
    cdef public double dim
    cdef public int num_atoms, n_fixed
    cdef public int id_cap, id_num
    cdef public list atom_indices
    cdef public int parent_pointer
    cdef public list child_pointer
    cdef public bint leaf
    cdef public DynamicOctreeNodeAttr attribs

    def __init__(self, int node_id=-1):
        self.lx = self.ly = self.lz = 0.0
        self.dim = 0.0
        self.num_atoms = 0
        self.n_fixed = 0
        self.id_cap = 5
        self.id_num = 0
        self.atom_indices = []
        self.parent_pointer = -1
        self.child_pointer = [-1] * 8
        self.leaf = True
        self.attribs = DynamicOctreeNodeAttr()

    cpdef void init_node(self):
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
        self.id = None
        self.attribs = DynamicOctreeNodeAttr()

    cpdef bint is_leaf(self):
        return self.leaf

    cpdef void set_child_pointer(self, int loc, int ptr):
        if 0 <= loc < 8:
            self.child_pointer[loc] = ptr
        else:
            print("Error: Invalid child pointer index")

    cpdef void set_parent_pointer(self, int i):
        self.parent_pointer = i

    cpdef int get_parent_pointer(self):
        return self.parent_pointer

    cpdef void set_id(self, int node_id):
        self.id = node_id

    cpdef void set_lx(self, double value):
        self.lx = value

    cpdef void set_ly(self, double value):
        self.ly = value

    cpdef void set_lz(self, double value):
        self.lz = value

    cpdef void set_dim(self, double value):
        self.dim = value

    cpdef void set_num_atoms(self, int i):
        self.num_atoms = i

    cpdef void set_atom_indices(self, list i):
        self.atom_indices = i

    cpdef void set_leaf(self, bint i):
        self.leaf = i

    cpdef void set_IdCap(self, int i):
        self.id_cap = i

    cpdef void set_num_fixed(self, int i):
        self.n_fixed = i

    cpdef void set_atom_index(self, int loc, int index):
        if loc < 0:
            raise ValueError("Location index must be non-negative")

        while loc >= len(self.atom_indices):
            self.atom_indices.append(None)

        self.atom_indices[loc] = index

    cpdef double get_dim(self):
        return self.dim

    cpdef void combine_and_set_attribs(self, list all_child_attribs):
        self.attribs.combine_s(all_child_attribs)

    cpdef void compute_own_attribs(self, list atoms):
        self.attribs.compute_s(atoms)

    cpdef void update_attribs(self, Object obj, bint add):
        if add:
            self.attribs.update_s(obj, True)
            self.num_atoms += 1
        else:
            self.attribs.update_s(obj, False)
            self.num_atoms -= 1

cdef class DynamicOctree:
    cdef public list nodes
    cdef public list atoms
    cdef public int num_atoms, num_nodes, next_free_node
    cdef public bint octree_built, verbose
    cdef public OctreeConstructionParams construction_params
    cdef public int max_nodes
    cdef public object scoring_params
    cdef public dict object_to_node_map

    def __init__(self, list atoms, int n_atoms, OctreeConstructionParams cons_par, bint verbose=True, int max_nodes=-1):
        self.nodes = []
        self.atoms = atoms
        self.num_atoms = n_atoms
        self.num_nodes = 0
        self.next_free_node = -1
        self.octree_built = False
        self.verbose = verbose
        self.construction_params = cons_par
        self.max_nodes = max_nodes
        self.scoring_params = None
        self.object_to_node_map = {}

    cpdef void set_node_id(self, Object obj, int node_id):
        self.object_to_node_map[obj] = node_id

    cpdef int get_node_id(self, Object obj):
        return self.object_to_node_map.get(obj)

    cpdef bint build_octree(self):
        if self.verbose:
            print("Starting DynamicOctree::buildOctree")

        cdef int i, j, k
        cdef list indices = [i for i in range(self.num_atoms)]
        cdef list indices_temp = [0] * self.num_atoms
        cdef int octree_root

        try:
            if not self.init_free_node_server():
                print("ERROR: Could not create free node server")
                return False

            octree_root = self.get_next_free_node()

            if octree_root == -1:
                print("ERROR: Could not get next node")
                return False

            self.compute_root_bounding_box(octree_root, self.construction_params.get_slack_factor(), indices, 0, self.num_atoms - 1)
            self.nodes[octree_root].set_parent_pointer(-1)

            self.octree_built = self.expand_octree_node(octree_root, indices, indices_temp, 0, self.num_atoms - 1)
        finally:
            del indices
            del indices_temp

        return self.octree_built

    cpdef int create_octree_ptr(self, int a, int b):
        return (a << LOW_BITS) + b

    cpdef int get_index_in_node(self, int c):
        return c & 0x3FFF

    cpdef bint needs_dynamic_expansion(self, DynamicOctreeNode node):
        return node.num_atoms > (self.construction_params.max_leaf_size << 1)

    cpdef bint needs_dynamic_contraction(self, DynamicOctreeNode node):
        return node.num_atoms < (self.construction_params.max_leaf_size >> 1)

    cpdef bint allocate_nodes(self, int new_num_nodes):
        if self.verbose:
            print(f"Inside DynamicOctree::allocateNodes({new_num_nodes})")

        cdef list nodes = []
        cdef int i
        for i in range(new_num_nodes):
            node = DynamicOctreeNode()
            nodes.append(node)

        self.nodes = nodes

        if self.verbose:
            print(f"Allocated {new_num_nodes} nodes")

        return True

    cpdef bint reallocate_nodes(self, int new_num_nodes):
        if self.verbose:
            print(f"Inside DynamicOctree::reallocateNodes({new_num_nodes})")

        self.nodes = self.nodes[:new_num_nodes] + [DynamicOctreeNode() for _ in range(new_num_nodes - len(self.nodes))]

        return True

    cpdef bint init_free_node_server(self):
        if self.verbose:
            print("In DynamicOctree::initFreeNodeServer")

        self.num_nodes = 0
        self.next_free_node = -1

        self.allocate_nodes(INIT_NUM_OCTREE_NODES)

        if self.nodes is None:
            return False

        self.num_nodes = INIT_NUM_OCTREE_NODES

        cdef int i
        for i in range(self.num_nodes):
            self.nodes[i].set_parent_pointer(self.next_free_node)
            self.next_free_node = i

        # Print parent nodes
        print("Parent Nodes:")
        for i in range(self.num_nodes):
            parent_node_id = self.nodes[i].get_parent_pointer()
            if parent_node_id != -1:
                print(f"Node ID: {i}, Parent Node ID: {parent_node_id}")

        return True

    cpdef int get_next_free_node(self):
        cdef int new_num_nodes, i, next_node
        cdef DynamicOctreeNode node

        if self.verbose:
            print("In DynamicOctree::getNextFreeNode")

        if self.next_free_node == -1:
            new_num_nodes = 2 * self.num_nodes

            if new_num_nodes <= 0:
                new_num_nodes = INIT_NUM_OCTREE_NODES

            self.reallocate_nodes(new_num_nodes)

            if self.nodes is None:
                return -1

            for i in range(new_num_nodes - 1, self.num_nodes - 1, -1):
                node = DynamicOctreeNode.__new__(DynamicOctreeNode)
                node.init_node()
                node.set_id(i)
                node.set_parent_pointer(self.next_free_node)
                self.nodes[i] = node
                self.next_free_node = i

            self.num_nodes = new_num_nodes

        next_node = self.next_free_node
        self.next_free_node = self.nodes[next_node].get_parent_pointer()

        return next_node

    cpdef void compute_root_bounding_box(self, int node_id, double slack_factor, list indices, int start_id, int end_id):
        if self.verbose:
            print("In DynamicOctree::computeRootBoundingBox")

        cdef DynamicOctreeNode node = self.nodes[node_id]
        cdef int s = indices[start_id]
        cdef double minX, minY, minZ, maxX, maxY, maxZ
        cdef int i, j

        minX = maxX = self.atoms[s].getX()
        minY = maxY = self.atoms[s].getY()
        minZ = maxZ = self.atoms[s].getZ()

        for i in range(start_id + 1, end_id + 1):
            j = indices[i]
            minX = min(minX, self.atoms[j].getX())
            maxX = max(maxX, self.atoms[j].getX())
            minY = min(minY, self.atoms[j].getY())
            maxY = max(maxY, self.atoms[j].getY())
            minZ = min(minZ, self.atoms[j].getZ())
            maxZ = max(maxZ, self.atoms[j].getZ())

        cdef double cx = (minX + maxX) / 2
        cdef double cy = (minY + maxY) / 2
        cdef double cz = (minZ + maxZ) / 2

        cdef double dim = max(maxX - minX, maxY - minY, maxZ - minZ)
        dim *= slack_factor

        node.set_lx(cx - dim * 0.5)
        node.set_ly(cy - dim * 0.5)
        node.set_lz(cz - dim * 0.5)
        node.set_dim(dim)

    cpdef bint needs_expansion(self, DynamicOctreeNode node):
        return not (node.num_atoms <= self.construction_params.get_max_leaf_size() or node.dim <= self.construction_params.get_max_leaf_dim())

    cpdef bint expand_octree_node(self, int node_id, list indices, list indices_temp, int start_id, int end_id):
        cdef DynamicOctreeNode node
        cdef int nAtoms, nfixed, i, j, k
        cdef double dim
        cdef list count, start_index, cur_index
        cdef int nfxd

        if self.verbose:
            print("In DynamicOctree::expandOctreeNode")

        node = self.nodes[node_id]
        nAtoms = end_id - start_id + 1
        node.set_num_atoms(nAtoms)

        self.compute_leaf_attributes(node_id, indices, start_id, end_id)
        dim = node.get_dim()

        if not self.needs_expansion(node):
            node.set_leaf(True)
            new_indices = [-1] * (2 * nAtoms)

            if new_indices is None:
                print("Failed to allocate leaf node memory for octree!")
                return False

            node.set_atom_indices(new_indices)
            node.set_IdCap(2 * nAtoms)

            nfixed = 0

            for i in range(start_id, end_id + 1):
                j = indices[i]

                if self.atoms[j].is_fixed():
                    node.set_atom_index(nfixed, j)
                    self.atoms[j].node_id = node_id
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
                        self.atoms[j].node_id = node_id
                        k += 1

        else:
            node.set_leaf(False)

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

                    for k in range(start_index[i], start_index[i] + count[i]):
                        atom_index = indices_temp[k]
                        self.object_to_node_map[self.atoms[atom_index]] = j
                        self.atoms[atom_index].setNodeID(j)

                    if not self.expand_octree_node(j, indices_temp, indices, start_index[i], start_index[i] + count[i] - 1):
                        return False

                    nfxd += self.nodes[j].n_fixed

                    node.set_num_fixed(nfxd)

                else:
                    node.set_child_pointer(i, -1)

        return True

    cpdef void print_octree(self):
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

    cdef void print_node_details(self, int node_id, str indent=""):
        """
        Print details of a node and its children recursively.

        Args:
            node_id (int): The ID of the node to print.
            indent (str): Indentation string for better visualization.
        """
        cdef DynamicOctreeNode node = self.nodes[node_id]
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

    cdef void traverse_and_print(self, int node_id, int depth):
        """
        Traverse the octree and print the contents of each node.

        Args:
            node_id: ID of the current node.
            depth: Depth of the current node in the octree.
        """
        cdef DynamicOctreeNode node = self.nodes[node_id]
        cdef str prefix = "  " * depth
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

        # if not node.leaf:
        #     for i, child_id in enumerate(node.child_pointer):
        #         if child_id != -1:
        #             print(prefix + f"Child {i}:")
        #             self.traverse_and_print(child_id, depth + 1)
        print("\n")

    cdef void compute_non_root_bounding_box(self, int node_id, int child_id=None):
        """
        Compute the bounding box for a non-root node.

        Args:
            node_id (int): The ID of the node.
            child_id (int): The ID of the child node.

        This method computes the bounding box for a non-root node based on its parent's bounding box.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeNonRootBoundingBox(node_id={node_id}, child_id={child_id})")

        cdef DynamicOctreeNode node = self.nodes[node_id]

        if node.parent_pointer < 0:
            return

        cdef DynamicOctreeNode pnode = self.nodes[node.parent_pointer]

        if child_id is None:
            for i in range(8):
                if pnode.child_pointer[i] == node_id:
                    child_id = i
                    break

            if child_id is None:
                return

        cdef double lx = pnode.lx
        cdef double ly = pnode.ly
        cdef double lz = pnode.lz
        cdef double dim = pnode.dim

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

        # if self.verbose:
        #     print(f"nonroot dim = {dim}")

    cdef void compute_non_leaf_attributes(self, int node_id):
        """
        Compute the attributes for a non-leaf node.

        Args:
            node_id (int): The ID of the node.

        This method computes the attributes for a non-leaf node by combining the attributes of its child nodes.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeNonLeafAttributes(node_id={node_id})")

        cdef double sumX = 0
        cdef double sumY = 0
        cdef double sumZ = 0
        cdef double sumQ = 0

        cdef list child_attribs = []

        for i in range(8):
            child_id = self.nodes[node_id].child_pointer[i]
            if child_id >= 0:
                child_attribs.append(self.nodes[child_id].attribs)

        self.nodes[node_id].combine_and_set_attribs(child_attribs)

    cdef void compute_leaf_attributes(self, int node_id, list indices, int start_id, int end_id):
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

        cdef list node_atoms = [self.atoms[j] for j in indices[start_id:end_id + 1]]

        # Update node_id attribute for each object
        for atom in node_atoms:
            atom.node_id = node_id

        # print("prepared list of atoms")

        self.nodes[node_id].compute_own_attribs(node_atoms)

        # print("computed attributes")

    cdef int get_child_id(self, DynamicOctreeNode node, Object atom):
        """
        Get the child ID of a node based on the position of an object.

        Args:
            node (DynamicOctreeNode): The node.
            atom (Object): The object.

        Returns:
            int: The child ID.

        This method returns the child ID of a node based on the position of the given object.
        """
        cdef double dim = 0.5 * node.dim
        cdef double cx = node.lx + dim
        cdef double cy = node.ly + dim
        cdef double cz = node.lz + dim

        cdef int k = ((atom.getZ() >= cz) << 2) + ((atom.getY() >= cy) << 1) + (atom.getX() >= cx)

        return k

    cdef void collect_atoms_from_leaves(self, int node_id, list indices, int start_id):
        """
        Recursively collects atoms from leaf nodes.

        Args:
            node_id (int): The ID of the current node.
            indices (List[int]): The list to store atom indices.
            start_id (int): The starting index in the indices list.

        This method recursively collects atoms from leaf nodes and stores their indices in the indices list.
        """
        if self.nodes[node_id].is_leaf:
            for i in range(self.nodes[node_id].num_atoms):
                indices[start_id + i] = self.nodes[node_id].atom_indices[i]
        else:
            for i in range(8):
                child_id = self.nodes[node_id].child_pointers[i]
                if child_id >= 0:
                    self.collect_atoms_from_leaves(child_id, indices, start_id)
                    start_id += self.nodes[child_id].num_atoms
                    self.free_node(child_id)

    cdef bint contract_octree_node(self, int node_id):
        """
        Contracts the octree node if necessary.

        Args:
            node_id (int): The ID of the node to contract.

        Returns:
            bool: True if successful, False otherwise.
        """
        if self.verbose:
            print("In DynamicOctree::contractOctreeNode")

        cdef DynamicOctreeNode node = self.nodes[node_id]

        cdef int n_atoms = node.num_atoms

        # if not self.needs_contraction(node):
        #     return True

        self.compute_non_leaf_attributes(node_id)

        cdef list new_indices = [0] * (2 * n_atoms)

        self.collect_atoms_from_leaves(node_id, new_indices, 0)

        cdef int k = 0
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

    cdef void traverse_octree(self, int node_id):
        """
        Traverses the octree recursively and prints information about each node.

        Args:
            node_id (int): The ID of the current node.
        """
        if self.verbose:
            print("In DynamicOctree::traverseOctree")

        cdef DynamicOctreeNode node = self.nodes[node_id]

        print("%d ( %d, %lf ): " % (node_id, node.num_atoms, node.dim), end="")  # REMOVED num_fixed from here, CHECK IF THAT IS USEFUL

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    print("%d " % node.child_pointers[i], end="")

        print()

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    self.traverse_octree(node.child_pointers[i])

    cdef int get_subtree_size(self, int node_id):
        """
        Calculates the size of the subtree rooted at the given node.

        Args:
            node_id (int): The ID of the root node of the subtree.

        Returns:
            int: The size of the subtree.
        """
        if self.verbose:
            print("In DynamicOctree::getSubtreeSize (Recursive)")

        cdef DynamicOctreeNode node = self.nodes[node_id]

        cdef int s = sys.getsizeof(DynamicOctreeNode)

        s += node.id_cap * sys.getsizeof(int)

        if not node.is_leaf:
            for i in range(8):
                if node.child_pointers[i] >= 0:
                    s += self.get_subtree_size(node.child_pointers[i])

        return s

    cdef int get_octree_size(self):
        """
        Calculates the size of the entire octree.

        Returns:
            int: The size of the octree.
        """
        if self.verbose:
            print("In DynamicOctree::getOctreeSize")

        return self.get_subtree_size(0)

cdef bint remove_atom_from_leaf(self, int node_id, int atom_id):
    """ Removes an atom from a leaf node.
    Args:
        node_id (int): The ID of the leaf node.
        atom_id (int): The ID of the atom to remove.
    Returns:
        bool: True if successful, False otherwise.
    """
cdef bint remove_atom_from_leaf(self, int node_id, int atom_id):
    """ Removes an atom from a leaf node.
    Args:
        node_id (int): The ID of the leaf node.
        atom_id (int): The ID of the atom to remove.
    Returns:
        bool: True if successful, False otherwise.
    """
    if self.verbose:
        print("In DynamicOctree::removeAtomFromLeaf")
    cdef Object atom = self.atoms[atom_id]
    cdef int j = self.get_index_in_node(atom.id)
    cdef DynamicOctreeNode node = self.nodes[node_id]
    cdef int n = node.num_atoms
    cdef int index_nf, index_j
    if atom.is_fixed():
        n_fixed = node.n_fixed
        node.atom_indices[j] = node.atom_indices[n_fixed - 1]
        node.atom_indices[n_fixed - 1] = node.atom_indices[n - 1]
        index_nf = node.atom_indices[n_fixed - 1]
        index_j = node.atom_indices[j]
        self.atoms[index_nf].id = self.atoms[index_j].id
        self.atoms[index_j].id = atom.id
        node.n_fixed = n_fixed - 1
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

    cdef bint remove_atom_from_non_leaf(self, int node_id, int atom_id):
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

        cdef DynamicOctreeNode node = self.nodes[node_id]
        cdef Object atom = self.atoms[atom_id]