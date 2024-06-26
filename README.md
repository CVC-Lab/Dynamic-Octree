# Dynamic Octree Implementation in Python

>Overview of the Dynamic Octree implementation in C++: 
>https://hackmd.io/@-SXqtY-rT3qDfaj3SFi27g/SyojQsVo3

I have implemented the python version of the Dynamic Octree because of 2 main reasons:
- We need to make it adaptable, the original implementation was tailored to application on molecular structure. 
    - Here, I have written the code independant of the objects being considered, ofcourse the objects are supposed to be spheres and it is possible to convert any object into it's respective union of spheres representation which was discussed and considered to be arguably the most flexible representation for 4D planning and tracking of arbitrary objects.
    - Check these two repositories to get an idea of what the union of spheres representation means and how to implement: - https://colab.research.google.com/drive/1kS191jwUItQrp-3NZDp1BE1wKIx8nlM_?usp=sharing https://colab.research.google.com/drive/16eknfafU0WkxBSGqV5COmtO9LLiFKsL9?usp=sharing
- The code is written modular, so that it can directly be use while working with GVD which is the ultimate GOAL.

## Python Implementation
In the Python implementation of the DynamicOctree, the core structure and functionality remain faithful to the original C++ version, with necessary adaptations to suit the Python language and its conventions. Here are the key points of the Python implementation:
![DYNAMIC OCTREE](https://hackmd.io/_uploads/Syf0jqGxC.png)

### Class Structure
- The **`OctreeConstructionParams` class** encapsulates parameters used during the construction of the octree, such as maximum leaf size, maximum leaf dimension, and slack factor. These parameters control the granularity and structure of the octree, allowing for customization based on specific application requirements.
- The **`DynamicOctree` class** encapsulates the octree data structure, providing methods for building, traversing, and manipulating the octree. It contains attributes such as nodes, atoms, construction parameters, and verbosity settings.
- The **`DynamicOctreeNode` class** represents individual nodes in the octree. It encapsulates attributes such as position, dimensions, number of objects, and child pointers. This class also provides methods for updating node attributes and computing node properties.
- The **`Object` class** represents objects that are inserted into the octree. It encapsulates attributes such as position and fixed status, along with methods for accessing and modifying object properties. This class allows for easy integration of arbitrary objects into the octree structure.

### Changes from C++ Version
Several adjustments were made to the Python implementation to accommodate the language differences and idioms. These include changes in syntax, handling of dynamic memory, and data structures. For example, lists are used extensively in Python instead of arrays (which can be replaced by tensors if deemed necessary in the future), and memory management is handled automatically by Python's garbage collector.

### Key Methods
- Initialization
- Building the octree
- Adding and removing objects
- Traversing and printing the octree

Overall, this Python implementation of the DynamicOctree provides a flexible and efficient data structure for spatial indexing and collision detection in Python applications. It combines the power of the octree data structure with the ease of use and expressiveness of the Python language.

The structuring of the repository is as follows:
- `octree.py`: which contains the key classes of the data structure
- `main.py`: contains several test cases to showcase the working of the data structure in action
- `objects.py`: a template class to show how any given object should be defined to work with the data structure.
