# cython: language_level=3

import cython
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np
from objects cimport Object

cdef int INIT_NUM_OCTREE_NODES = 4
cdef int LOW_BITS = 14

cdef class OctreeConstructionParams:
    cdef public int max_leaf_size
    cdef public int max_leaf_dim
    cdef public float slack_factor

    def __init__(self, int max_leaf_size, int max_leaf_dim, float slack_factor=1.0):
        self.max_leaf_size = max_leaf_size
        self.max_leaf_dim = max_leaf_dim
        self.slack_factor = slack_factor

    cpdef set_max_leaf_size(self, int max_leaf_size):
        self.max_leaf_size = max_leaf_size

    cpdef set_max_leaf_dim(self, int max_leaf_dim):
        self.max_leaf_dim = max_leaf_dim

    cpdef set_slack_factor(self, float slack_factor):
        self.slack_factor = slack_factor

    cpdef int get_max_leaf_size(self):
        return self.max_leaf_size

    cpdef int get_max_leaf_dim(self):
        return self.max_leaf_dim

    cpdef float get_slack_factor(self):
        return self.slack_factor

    cpdef void print_params(self):
        print(f"Max leaf size: {self.max_leaf_size}, and max leaf dim: {self.max_leaf_dim}.")

cdef class DynamicOctreeNodeAttr:
    cdef public double sx, sy, sz

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        self.sx = x
        self.sy = y
        self.sz = z

    cdef combine_s(self, all_attribs):
        self.sx = sum(attr.sx for attr in all_attribs)
        self.sy = sum(attr.sy for attr in all_attribs)
        self.sz = sum(attr.sz for attr in all_attribs)

    cdef compute_s(self, atoms):
        self.sx = sum(atom.x for atom in atoms)
        self.sy = sum(atom.y for atom in atoms)
        self.sz = sum(atom.z for atom in atoms)

    cpdef update_s(self, atm, bint add):
        if add:
            self.sx += atm.x
            self.sy += atm.y
            self.sz += atm.z
        else:
            self.sx -= atm.x
            self.sy -= atm.y
            self.sz -= atm.z

cdef class DynamicOctreeNode:
    cdef public double lx, ly, lz, dim
    cdef public int num_atoms, n_fixed, id_cap, parent_pointer
    cdef public bint leaf
    cdef public list atom_indices
    cdef public DynamicOctreeNodeAttr attribs
    cdef public object id
    cdef public int child_pointer[8]

    def __init__(self, node_id=-1):
        self.init_node()
        self.id = node_id

    cpdef init_node(self):
        self.lx = 0.0
        self.ly = 0.0
        self.lz = 0.0
        self.dim = 0.0
        self.num_atoms = 0
        self.n_fixed = 0
        self.id_cap = 0
        self.atom_indices = []
        self.parent_pointer = -1
        self.child_pointer = [-1] * 8
        self.leaf = True
        self.attribs = DynamicOctreeNodeAttr()

    cpdef double distance(self, DynamicOctreeNode other):
        cdef double center_self_x = self.lx + self.dim / 2
        cdef double center_self_y = self.ly + self.dim / 2
        cdef double center_self_z = self.lz + self.dim / 2

        cdef double center_other_x = other.lx + other.dim / 2
        cdef double center_other_y = other.ly + other.dim / 2
        cdef double center_other_z = other.lz + other.dim / 2

        cdef double dx = center_self_x - center_other_x
        cdef double dy = center_self_y - center_other_y
        cdef double dz = center_self_z - center_other_z

        return sqrt(dx * dx + dy * dy + dz * dz)

    cdef bint is_leaf(self):
        return self.leaf

    cpdef set_child_pointer(self, int loc, int ptr):
        if 0 <= loc < 8:
            self.child_pointer[loc] = ptr
        else:
            print("Error: Invalid child pointer index")

    cpdef set_parent_pointer(self, int i):
        self.parent_pointer = i

    cpdef int get_parent_pointer(self):
        return self.parent_pointer

    cpdef set_id(self, node_id):
        self.id = node_id

    cpdef set_lx(self, double value):
        self.lx = value

    cpdef set_ly(self, double value):
        self.ly = value

    cpdef set_lz(self, double value):
        self.lz = value

    cpdef set_dim(self, double value):
        self.dim = value

    cpdef set_num_atoms(self, int i):
        self.num_atoms = i

    cpdef set_atom_indices(self, i):
        self.atom_indices = i

    cpdef set_leaf(self, bint i):
        self.leaf = i

    cpdef set_IdCap(self, int i):
        self.id_cap = i

    cpdef set_num_fixed(self, int i):
        self.n_fixed = i

    cpdef set_atom_index(self, int loc, int index):
        if loc < 0:
            raise ValueError("Location index must be non-negative")
        while loc >= len(self.atom_indices):
            self.atom_indices.append(None)
        self.atom_indices[loc] = index

    cpdef double get_lx(self):
        return self.lx

    cpdef double get_ly(self):
        return self.ly

    cpdef double get_lz(self):
        return self.lz

    cpdef double get_dim(self):
        return self.dim

    cpdef combine_and_set_attribs(self, all_child_attribs):
        self.attribs.combine_s(all_child_attribs)

    cpdef compute_own_attribs(self, atoms):
        self.attribs.compute_s(atoms)

    cpdef update_attribs(self, obj, bint add):
        if add:
            self.attribs.update_s(obj, add=True)
            self.num_atoms += 1
        else:
            self.attribs.update_s(obj, add=False)
            self.num_atoms -= 1

cdef class DynamicOctree:
    cdef list nodes  # List to store octree nodes
    cdef list atoms  # List of atoms
    cdef int num_atoms  # Number of atoms
    cdef int num_nodes  # Number of nodes in the octree
    cdef int next_free_node  # Index of the next free node
    cdef bint octree_built  # Flag indicating if the octree is built
    cdef bint verbose  # Verbosity flag
    cdef object construction_params  # Octree construction parameters
    cdef int max_nodes  # Maximum number of nodes allowed in the octree
    cdef object scoring_params  # Scoring parameters for the octree
    cdef dict object_to_node_map  # Initialize object to node mapping dictionary
    cdef list nb_lists  # Initialize neighborhood lists
    cdef list nb_lists_with_dist  # Initialize neighborhood lists which stores distances as well
    cdef int interaction_distance
    cdef int root_node_id

    def __init__(self, atoms, int n_atoms, cons_par, bint verbose=True, int max_nodes=4, int interaction_distance=100):
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
        self.nb_lists = [[] for _ in range(self.num_atoms)]
        self.nb_lists_with_dist = [[] for _ in range(self.num_atoms + 1)]
        self.interaction_distance = interaction_distance
        self.root_node_id = 0

    cpdef set_node_id(self, obj, int node_id):
        self.object_to_node_map[obj] = node_id

    cpdef int get_node_id(self, obj):
        return self.object_to_node_map.get(obj, -1)

    cpdef int create_octree_ptr(self, int a, int b):
        return (a << LOW_BITS) + b

    cpdef print_children(self, int node_id):
        children = [i for i, node in enumerate(self.nodes) if node.get_parent_pointer() == node_id]
        if self.verbose:
            print(f"Children of node {node_id}: {children}")

    cpdef build_nb_lists(self, double interaction_range):
        if self.verbose:
            print("\nBuilding neighbor lists...\n")
        self._accum_inter(self.nodes[self.root_node_id], self.nodes[self.root_node_id], interaction_range)
        if self.verbose:
            print("Neighbor lists construction completed.\n")

    cpdef _accum_inter(self, DynamicOctreeNode u, DynamicOctreeNode v, double d):
        if u.distance(v) > d:
            return
        elif u.leaf and v.leaf:
            for p_idx in u.atom_indices:
                p = self.atoms[p_idx]
                for q_idx in v.atom_indices:
                    q = self.atoms[q_idx]
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
                    v_child = self.nodes[v_child_idx]
                    self._accum_inter(u, v_child, d)
        elif v.leaf:
            for u_child_idx in u.child_pointer:
                if u_child_idx != -1:
                    u_child = self.nodes[u_child_idx]
                    self._accum_inter(u_child, v, d)
        else:
            for u_child_idx in u.child_pointer:
                if u_child_idx != -1:
                    u_child = self.nodes[u_child_idx]
                    for v_child_idx in v.child_pointer:
                        if v_child_idx != -1:
                            v_child = self.nodes[v_child_idx]
                            self._accum_inter(u_child, v_child, d)

    cpdef double _distance(self, p1, p2):
        cdef double distance = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
        return distance

    cpdef bint build_octree(self):
        if self.verbose:
            print("Starting DynamicOctree::buildOctree")

        indices = [i for i in range(self.num_atoms)]
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

            self.compute_root_bounding_box(octree_root, self.construction_params.get_slack_factor(), indices, 0, self.num_atoms - 1)
            self.nodes[octree_root].set_parent_pointer(-1)

            if self.verbose:
                print("Number of atoms considered while expanding octree: ", indices)

            if self.num_nodes < self.max_nodes:
                self.octree_built = self.expand_octree_node(octree_root, indices, indices_temp, 0, self.num_atoms - 1)
            
            if self.verbose:
                print("\nobject to node map after expanding the octree: ", self.object_to_node_map)
                print("\n")
                for i in range(self.num_nodes):
                    self.print_children(i)
            
            if self.verbose:
                print("-----------Atoms in the respective Nodes------------")
                for i in range(self.num_nodes):
                    print(f"The indices of the atoms in Node {i}: {self.nodes[i].atom_indices}")

            for atom_index in range(self.num_atoms):
                node = self.get_node_containing_point(self.atoms[atom_index])
                self.update_nb_lists(atom_index, node)
                                
        finally:
            del indices
            del indices_temp

        return self.octree_built

    cpdef bint init_free_node_server(self):
        if self.verbose:
            print("In DynamicOctree::initFreeNodeServer")

        self.num_nodes = 0
        self.next_free_node = -1

        self.allocate_nodes(INIT_NUM_OCTREE_NODES)

        if self.nodes is None:
            return False

        self.num_nodes = INIT_NUM_OCTREE_NODES

        for i in range(self.num_nodes - 1, -1, -1):
            self.nodes[i].set_parent_pointer(self.next_free_node)
            self.next_free_node = i

        if self.verbose:
            print("Allocated {} new nodes".format(INIT_NUM_OCTREE_NODES))
        
        if self.verbose:
            print("Parent Nodes for the Initially Allocated Nodes:")
        for i in range(self.num_nodes):
            parent_node_id = self.nodes[i].get_parent_pointer()
            if self.verbose:
                print(f"Node ID: {i}, Parent Node ID: {parent_node_id}")

        return True

    cpdef bint allocate_nodes(self, int new_num_nodes):
        if self.verbose:
            print("Inside DynamicOctree::allocateNodes({})".format(new_num_nodes))

        self.nodes = [DynamicOctreeNode() for _ in range(new_num_nodes)]

        if self.verbose:
            print("Allocated {} nodes".format(new_num_nodes))

        return True

    cdef int get_next_free_node(self):
        if self.verbose:
            print("In DynamicOctree::getNextFreeNode")

        cdef int new_num_nodes
        cdef int next_node

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

        return next_node

    cdef bint reallocate_nodes(self, int new_num_nodes):
        if self.verbose:
            print("Inside DynamicOctree::reallocateNodes({})".format(new_num_nodes))

        self.nodes = self.nodes[:new_num_nodes] + [DynamicOctreeNode() for _ in range(new_num_nodes - len(self.nodes))]

        return True

    cdef void compute_root_bounding_box(self, int node_id, float slack_factor, list indices, int start_id, int end_id):
        if self.verbose:
            print("In DynamicOctree::computeRootBoundingBox")

        node = self.nodes[node_id]

        s = indices[start_id]

        minX, minY, minZ = maxX, maxY, maxZ = self.atoms[s].getX(), self.atoms[s].getY(), self.atoms[s].getZ()

        for i in range(start_id + 1, end_id + 1):
            j = indices[i]

            minX = 0
            maxX = 50

            minY = 0
            maxY = 50

            minZ = 0
            maxZ = 50

        cx, cy, cz = (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2

        dim = max(maxX - minX, maxY - minY, maxZ - minZ)
        dim *= slack_factor

        node.set_lx(cx - dim * 0.5)
        node.set_ly(cy - dim * 0.5)
        node.set_lz(cz - dim * 0.5)
        node.set_dim(dim)

        if self.verbose:
            print(f"Node {node_id}: Center=({cx}, {cy}, {cz}), Dimension={dim}")

    cdef bint expand_octree_node(self, int node_id, list indices, list indices_temp_py, int start_id, int end_id):
        if self.verbose:
            print("In DynamicOctree::expandOctreeNode")

        node = self.nodes[node_id]
        nAtoms = end_id - start_id + 1
        node.set_num_atoms(nAtoms)
        self.compute_leaf_attributes(node_id, indices, start_id, end_id)
        dim = node.get_dim()
        
        if node_id == self.root_node_id:
            if self.verbose:
                print(f"Node {node_id} is the root node")
            node.set_leaf(False)
            node.set_parent_pointer(-1)

        if self.verbose:
            print(f"Node {node_id} initial state: leaf={node.is_leaf()}, parent={node.get_parent_pointer()}, atoms={node.atom_indices}")

        if not self.needs_expansion(node):
            if self.verbose:
                print(f"Node {node_id} is a leaf Node")
            node.set_leaf(True)
            new_indices = [-1] * nAtoms

            if new_indices is None:
                if self.verbose:
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
                indices_temp_py[cur_index[k]] = j
                cur_index[k] += 1
            
            nfxd = 0
            
            for i in range(8):
                if count[i] > 0:
                    j = self.get_next_free_node()
                    node.set_child_pointer(i, j)
                    self.nodes[j].set_parent_pointer(node_id)
                    self.compute_non_root_bounding_box(j, i)

                    for k in range(start_index[i], start_index[i] + count[i]):
                        atom_index = indices_temp_py[k]
                        self.object_to_node_map[self.atoms[atom_index]] = j
                        self.nodes[j].atom_indices.append(atom_index)
                    
                    if self.verbose:
                        print(f"Node {j} created with parent {node_id}")

                    if self.num_nodes < self.max_nodes:
                        if not self.expand_octree_node(j, indices_temp_py, indices, start_index[i], start_index[i] + count[i] - 1):
                            return False

                    nfxd += self.nodes[j].n_fixed
                    node.set_num_fixed(nfxd)
                    node.atom_indices = []

                else:
                    node.set_child_pointer(i, -1)
        
        node.atom_indices = [idx for idx in node.atom_indices if idx != -1]

        for i in range(8):
            child_id = node.child_pointer[i]
            if child_id != -1 and self.nodes[child_id].num_atoms == 0:
                node.set_child_pointer(i, -1)
                self.nodes[child_id] = None

        return True

    cpdef compute_leaf_attributes(self, int node_id, list indices, int start_id, int end_id):
        """
        Compute the attributes for a leaf node.

        Args:
            node_id (int): The ID of the node.
            indices (list): List of atom indices.
            start_id (int): Start index of atoms.
            end_id (int): End index of atoms.
        """
        if self.verbose:
            print(f"In DynamicOctree::computeLeafAttributes(node_id={node_id}, indices[start_id]={indices[start_id]}, indices[end_id]={indices[end_id]})")

        cdef list node_atoms = [self.atoms[j] for j in indices[start_id:end_id + 1]]

        # Update node_id attribute for each object
        for atom in node_atoms:
            atom.node_id = node_id

        # print("prepared list of atoms")

        self.nodes[node_id].compute_own_attribs(node_atoms)

    cpdef bint needs_expansion(self, node):
        """
        Checks whether a node needs expansion based on the maximum leaf size and dimension.

        Args:
            node (DynamicOctreeNode): The node to check.

        Returns:
            bool: True if the node needs expansion, False otherwise.
        """
        # Check if the number of atoms in the node exceeds the maximum leaf size
        return not (node.num_atoms <= self.construction_params.get_max_leaf_size())

    cpdef int get_child_id(self, node, atom):
        """
        Get the child ID of a node based on the position of an object.

        Args:
            node (DynamicOctreeNode): The node.
            atom (Object): The object.

        Returns:
            int: The child ID.
        """
        cdef double dim = 0.5 * node.dim
        cdef double cx = node.lx + dim
        cdef double cy = node.ly + dim
        cdef double cz = node.lz + dim

        cdef int k = ((atom.getZ() >= cz) << 2) + ((atom.getY() >= cy) << 1) + (atom.getX() >= cx)
        return k

    cpdef compute_non_root_bounding_box(self, int node_id, int child_id=-1):
        """
        Compute the bounding box for a non-root node.

        Args:
            node_id (int): The ID of the node.
            child_id (int): The ID of the child node.
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

        if self.verbose:
            print(f"\nNode {node_id}: Center=({lx + dim * 0.5}, {ly + dim * 0.5}, {lz + dim * 0.5}), Dimension={dim}")

    cpdef update_octree(self, atom, tuple new_position):
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

        print(f"Updated Object {self.atoms.index(atom)} Position")

        # Step 2: Find the current node containing the atom
        current_node = self.get_node_containing_point(atom)
        if self.verbose:
            print(f"current node: {self.nodes.index(current_node)}, atoms in the current node: {current_node.atom_indices}")

        # Step 3: Remove the atom from the current node
        self.remove_point_from_node(current_node, atom)

        # Step 4: Find the nearest ancestor that can contain the new position
        ancestor_node = self.nearest_bounding_ancestor(current_node, atom)
        if self.verbose:
            print(f"Ancestor node: {self.nodes.index(ancestor_node)}")

        # Step 5: Find the furthest descendant of the ancestor that can contain the new position
        target_node = self.furthest_bounding_descendant(ancestor_node, atom)
        if self.verbose:
            print(f"Target node: {self.nodes.index(target_node)}")

        # Step 6: Add the atom to the target node
        self.add_point_to_node(target_node, atom)

        # Step 7: Expand the node if necessary
        # self.expand_node_if_needed(target_node)

        # Step 8: Contract nodes if necessary
        contraction_node = self.find_furthest_proper_ancestor(target_node)
        if contraction_node:
            self.contract_octree_node(self.nodes.index(contraction_node))

        # Step 9: Update neighborhood lists
        self.update_nb_lists(self.atoms.index(atom), target_node)

        if self.verbose:
            print("\nobject to node map after updating the octree: ", self.object_to_node_map)

    cdef get_node_containing_point(self, p):
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
                    child_node = self.nodes[child_id]
                    if self.inside_node(child_node, p):
                        return find_node(child_node)

            return None

        if self.verbose:
            print("In DynamicOctree::get_node_containing_point")

        cdef DynamicOctreeNode root_node = self.nodes[0]

        if self.verbose:
            print(f"Starting at root node, child pointers: {[idx for idx in root_node.child_pointer if idx != -1]}")

        cdef DynamicOctreeNode result_node = find_node(root_node)

        if result_node is None:
            if self.verbose:
                print(f"Atom {self.atoms.index(p)} not found in any child of root node")
            return root_node

        return result_node

    cpdef bint inside_node(self, DynamicOctreeNode node, Object obj):
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

    cpdef remove_point_from_node(self, DynamicOctreeNode node, Object p):
        """
        Removes the point from the specified node.

        Args:
            node (DynamicOctreeNode): The node to remove the point from.
            p (Object): The point to remove.
        """
        cdef int index_in_node
        cdef int atom_index
        cdef bint is_leaf

        try:
            index_in_node = self.get_index_in_node(self.nodes.index(node), p.id)

            # Ensure that the object ID is valid and present in self.atoms
            atom_index = self.atoms.index(p.id)

            # Convert Python boolean to cdef bint
            is_leaf = node.leaf

            if is_leaf:
                self.remove_atom_from_leaf(self.nodes.index(node), atom_index)
            else:
                self.remove_atom_from_non_leaf(self.nodes.index(node), atom_index)

        except ValueError as e:
            if self.verbose:
                print(f"Error removing point: {e}")
                print(f"Object ID: {p.id}")
                print(f"Node index: {self.nodes.index(node)}")
                print(f"Atoms in the current node: {node.atom_indices}")

    cpdef int get_index_in_node(self, int node_id, int atom_id):
        """
        Get the index of the atom within the node's atom indices list.

        Args:
            atom_id (int): The ID of the atom.

        Returns:
            int: The index of the atom within the node's atom indices list, or -1 if not found.
        """
        cdef DynamicOctreeNode node = self.nodes[node_id]
        for index, atom_index in enumerate(node.atom_indices):
            if atom_index == atom_id:
                return index
        return -1

    cpdef bint remove_atom_from_leaf(self, int node_id, int atom_id):
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

        cdef Object atom = self.atoms[atom_id]
        cdef DynamicOctreeNode node = self.nodes[node_id]
        cdef int j
        cdef int n
        cdef int nf

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
            # Use Python list for slicing
            new_indices = node.atom_indices[:node.id_cap >> 1]

            if new_indices is None:
                if self.verbose:
                    print("Failed to contract leaf storage for octree!")
                return False

            # Assign the sliced list back to the node
            node.atom_indices = new_indices
            node.id_cap = node.id_cap >> 1

        node.update_attribs(atom, False)
        atom.id = -1

        if self.verbose:
            print("\n====================================")
            print("Number of atoms: ", self.num_atoms)
            print("Number of Nodes: ", self.num_nodes)

        return True

    cpdef bint remove_atom_from_non_leaf(self, int node_id, int atom_id):
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
        cdef int j
        cdef int n = len(node.atom_indices)  # Get the length once

        try:
            j = node.atom_indices.index(atom_id)
        except ValueError:
            # Atom is not present in the node
            if self.verbose:
                print("Atom is not present in the non-leaf node.")
            return False

        # Decrease atom count
        node.num_atoms -= 1

        if atom.is_fixed():
            node.num_fixed -= 1

        # Remove the atom from the node's atom_indices list
        if j < n - 1:  # Only swap if the index is not the last
            node.atom_indices[j] = node.atom_indices[-1]
        node.atom_indices.pop()  # Remove the last element

        # Update the node_id attribute of the removed atom to None
        atom.node_id = -1

        # Remove the atom from the object_to_node_map dictionary
        if self.atoms[atom_id] in self.object_to_node_map:
            del self.object_to_node_map[self.atoms[atom_id]]

        # Check if the node needs dynamic contraction and contract if necessary
        if self.needs_dynamic_contraction(node):
            return self.contract_octree_node(node_id)

        return True

    cpdef DynamicOctreeNode nearest_bounding_ancestor(self, DynamicOctreeNode u, Object p):
        if self.verbose:
            print("In DynamicOctree::nearest_bounding_ancestor")
        if self.inside_node(u, p):
            if self.verbose:
                print(f"Inside Node: {self.nodes.index(u)}, object_id: {self.atoms.index(p)}")
            return u
        else:
            if self.verbose:
                print(f"Node: {u.parent_pointer}, object_id: {self.atoms.index(p)}")
            if u.parent_pointer >= 0:
                return self.nearest_bounding_ancestor(self.nodes[u.parent_pointer], p)
            return None

    cpdef DynamicOctreeNode furthest_bounding_descendant(self, DynamicOctreeNode u, Object p):
        if self.verbose:
            print("In DynamicOctree::furthest_bounding_descendant")

        cdef int child_id
        cdef DynamicOctreeNode v  # Declare cdef variable outside the loop

        # Iterate over the child pointers
        for child_id in u.child_pointer:
            if child_id != -1:
                # Access node dynamically
                v = self.nodes[child_id]

                if self.inside_node(v, p):
                    # Recursively find the furthest bounding descendant
                    return self.furthest_bounding_descendant(v, p)
        
        return u

    cpdef void add_point_to_node(self, DynamicOctreeNode node, Object p):
        try:
            index_in_node = self.get_index_in_node(self.nodes.index(node), p.id)
            atom_index = self.atoms.index(p.id)

            if node.leaf:
                self.add_atom_to_leaf(self.nodes.index(node), atom_index)
            else:
                self.add_atom_to_non_leaf(self.nodes.index(node), atom_index)
                
        except ValueError as e:
            if self.verbose:
                print(f"Error adding point: {e}")
                print(f"Object ID: {p.id}")
                print(f"Node index: {self.nodes.index(node)}")
                print(f"Atoms in the current node: {node.atom_indices}")

    cpdef void add_atom_to_non_leaf(self, int node_id, int atom_id):
        if self.verbose:
            print("In DynamicOctree::addAtomToNonLeaf")

        cdef DynamicOctreeNode node
        if node_id >= len(self.nodes):
            self.nodes += [DynamicOctreeNode() for _ in range(node_id - len(self.nodes) + 1)]
        
        node = self.nodes[node_id]
        cdef Object atom = self.atoms[atom_id]

        node.num_atoms += 1

        node.update_attribs(atom, True)

        if atom.is_fixed():
            node.num_fixed += 1

        atom.node_id = node_id
        self.object_to_node_map[atom] = node_id
        node.atom_indices.append(atom_id)

    cpdef add_atom_to_leaf(self, int node_id, int atom_id):
        if self.verbose:
            print("In DynamicOctree::addAtomToLeaf")

        cdef DynamicOctreeNode node
        cdef Object atom
        cdef int n
        cdef int nf
        cdef list temp

        if node_id >= len(self.nodes):
            self.nodes += [DynamicOctreeNode() for _ in range(node_id - len(self.nodes) + 1)]
        
        node = self.nodes[node_id]
        atom = self.atoms[atom_id]

        n = node.num_atoms

        if n == node.id_cap:
            if node.id_cap == 0:
                node.id_cap = 1
                node.atom_indices = [None] * (node.id_cap << 1)
            else:
                node.atom_indices += [None] * (node.id_cap << 1)
            if node.atom_indices is None:
                print("Failed to expand leaf storage for octree!")
                return

            node.id_cap = node.id_cap << 1

        if atom.is_fixed():
            nf = node.num_fixed

            if n > 0:
                node.atom_indices[n] = node.atom_indices[nf]
                self.atoms[node.atom_indices[n]].id = self.create_octree_ptr(node_id, n)

            node.atom_indices[nf] = atom_id
            atom.id = self.create_octree_ptr(node_id, nf)
            atom.node_id = node_id

            node.num_fixed = nf + 1
        else:
            if n >= len(node.atom_indices):
                node.atom_indices += [-1] * (n - len(node.atom_indices) + 1)

            node.atom_indices[n] = atom_id
            atom.id = self.create_octree_ptr(node_id, n)
            atom.node_id = node_id

        node.update_attribs(atom, True)
        self.object_to_node_map[atom] = node_id
        node.atom_indices.append(atom_id)

        if self.needs_dynamic_expansion(node):
            # Use Python list for temporary storage
            temp = [None] * node.num_atoms

            if temp is None:
                print("Failed to allocate temporary storage for octree!")
                return

            print("\n====================================")
            print("Number of atoms: ", self.num_atoms)
            print("Number of Nodes: ", self.num_nodes)
            done = self.expand_octree_node(node_id, node.atom_indices, temp, 0, node.num_atoms - 1)

            return done
        else:
            return True

    cpdef void expand_node_if_needed(self, DynamicOctreeNode node):
        cdef int start_id, end_id
        cdef list indices_temp_py
        cdef int i

        if self.needs_expansion(node):
            start_id = 0
            end_id = len(node.atom_indices) - 1
            
            # Initialize indices_temp as a Python list
            indices_temp_py = [-1] * len(node.atom_indices)
            
            try:
                # Call the function with the Python list
                self.expand_octree_node(self.nodes.index(node), node.atom_indices, indices_temp_py, start_id, end_id)
            
            finally:
                # No need to free indices_temp_py as it's managed by Python's garbage collector
                pass

    cpdef DynamicOctreeNode find_furthest_proper_ancestor(self, DynamicOctreeNode node):
        cdef DynamicOctreeNode current_node = node
        cdef DynamicOctreeNode parent_node

        while current_node.parent_pointer >= 0:
            parent_node = self.nodes[current_node.parent_pointer]
            if parent_node.num_atoms < (self.construction_params.max_leaf_size >> 1):
                return parent_node
            current_node = parent_node

        return None

    cpdef bint contract_octree_node(self, int node_id):
        if self.verbose:
            print("In DynamicOctree::contractOctreeNode")

        cdef DynamicOctreeNode node = self.nodes[node_id]

        if not self.needs_contraction(node):
            return True
        
        self.compute_non_leaf_attributes(node_id)

        cdef list collected_atoms = []
        self.collect_atoms_from_subtree(node_id, collected_atoms)
        
        cdef list fixed_atoms = [atom for atom in collected_atoms if self.atoms[atom].is_fixed()]
        cdef list non_fixed_atoms = [atom for atom in collected_atoms if not self.atoms[atom].is_fixed()]
        
        self.delete_subtree(node_id)
        
        node.leaf = True
        node.atom_indices = fixed_atoms + non_fixed_atoms
        node.num_atoms = len(node.atom_indices)
        
        for i, atom_index in enumerate(node.atom_indices):
            self.atoms[atom_index].set_id(self.create_octree_ptr(node_id, i))
            
        for atom_index in node.atom_indices:
            self.update_nb_lists(atom_index, node)

        return True

    cpdef bint needs_contraction(self, DynamicOctreeNode node):
        cdef int total_atoms = len(node.atom_indices)
        if not node.leaf:
            for child_pointer in node.child_pointer:
                if child_pointer != -1:
                    total_atoms += self.nodes[child_pointer].num_atoms
        
        if total_atoms <= self.construction_params.get_max_leaf_size():
            return True
        
        if node.dim <= self.construction_params.get_max_leaf_dim():
            if not node.leaf:
                for child_pointer in node.child_pointer:
                    if child_pointer != -1 and self.nodes[child_pointer].dim > self.construction_params.get_max_leaf_dim():
                        return False
            return True

        return False

    cpdef void compute_non_leaf_attributes(self, int node_id):
        cdef double sumX, sumY, sumZ
        cdef double sumQ
        cdef int child_id
        cdef DynamicOctreeNode node
        cdef list child_attribs = []
        cdef DynamicOctreeNode child_node
        cdef int i

        if self.verbose:
            print(f"In DynamicOctree::computeNonLeafAttributes(node_id={node_id})")

        # Initialize sums
        sumX = 0
        sumY = 0
        sumZ = 0
        sumQ = 0

        node = self.nodes[node_id]

        for i in range(8):
            child_id = node.child_pointer[i]
            if child_id >= 0:
                child_node = self.nodes[child_id]
                child_attribs.append(child_node.attribs)

        self.nodes[node_id].combine_and_set_attribs(child_attribs)

    cpdef void collect_atoms_from_subtree(self, int node_id, list collected_atoms):
        cdef DynamicOctreeNode node = self.nodes[node_id]
        if node.leaf:
            collected_atoms.extend(node.atom_indices)
        else:
            for child_id in node.child_pointer:
                if child_id >= 0:
                    self.collect_atoms_from_subtree(child_id, collected_atoms)

    cpdef void delete_subtree(self, int node_id):
        cdef DynamicOctreeNode node = self.nodes[node_id]
        if not node.leaf:
            for child_id in node.child_pointer:
                if child_id >= 0:
                    self.delete_subtree(child_id)
        self.nodes[node_id] = None

cpdef void update_nb_lists(self, int atom_index, DynamicOctreeNode node):
    if self.verbose:
        print(f"\nIn update_nb_lists::With Atom {atom_index} and node {self.nodes.index(node)}")
    
    cdef Object atom = self.atoms[atom_index]
    cdef list stack = [self.nodes[self.root_node_id]]
    
    cdef set neighbors_with_dist = set()
    cdef set neighbors = set()
    cdef DynamicOctreeNode current_node
    cdef Object other_atom
    cdef double distance
    cdef int child_id

    while stack:
        current_node = stack.pop()
        if current_node is None:
            continue

        if self.verbose:
            print(f"Processing node: {self.nodes.index(current_node)}")
            print(f"Node {self.nodes.index(current_node)} is away from node {self.nodes.index(node)} with a distance of {current_node.distance(node)}")

        if current_node.distance(node) > self.interaction_distance:
            if self.verbose:
                print(f"Node {self.nodes.index(current_node)} is too far away from node {self.nodes.index(node)} with a distance of {current_node.distance(node)}")
            continue

        if current_node.leaf:
            for other_index in current_node.atom_indices:
                if other_index is not None and other_index != atom_index:
                    other_atom = self.atoms[other_index]
                    distance = atom.distance(other_atom)
                    neighbors_with_dist.add((other_index, distance))
                    if distance <= self.interaction_distance:
                        neighbors.add(other_index)
        else:
            for child_id in current_node.child_pointer:
                if child_id != -1 and child_id < len(self.nodes) and self.nodes[child_id] is not None:
                    stack.append(self.nodes[child_id])

    self.nb_lists_with_dist[atom_index] = list(neighbors_with_dist)
    self.nb_lists[atom_index] = list(neighbors)

    if self.verbose:
        print(f"Updated neighborhood lists for atom {atom_index}")
        print(f"Neighbors with distance: {self.nb_lists_with_dist[atom_index]}")
        print(f"Neighbors: {self.nb_lists[atom_index]}")

