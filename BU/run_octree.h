#ifndef _RUN_OCTREE_H_
#define _RUN_OCTREE_H_

/**
 * Added by Conrad Li from the Computational Visualization Labt at the 
 * University of Texas at Austin.
 * 
 * These functions were created to combine multiple useful functions from 
 * other files and port them to Python with Cython.
 **/
#include <math.h>
//#include <gmp.h>
#include <limits.h>


// Copied over from the BU file test.c
// Defines the parameters for the minimization
struct my_par
{
   int nbupdate;
   struct atomgrp *ag;
   struct agsetup *ags;
   double efacv;
   double eface;
   int useVdw, useElec, useHbond;
   double dcVdw, dcElec, dcHbond;
   double aprxdcVdw, aprxdcElec;
   OCTREE_PARAMS *octpar;
};


/**
 * Creates an atomgrp struct given user parameters 
 * 
 * Params:
 * pdbFile: path to the pdb file of the molecule
 * pdbFixedFile: path to the fixed pdb file
 * psfFile: path to the psf file
 * mol2File: path to the mol2 file of the molecule
 * prmFile: PRM file path
 * rtfFile: rtf file path
 * aprmFile: atomic prm file
 * int useHbond: flag to calculate hydrogen bonding energy
 * int dcHbond: distnace cutoff for hydrogen bonding
 * */
struct atomgrp *create_atomgrp (char *pdbFile, char *pdbFixedFile, char *psfFile,
                                char *mol2File, char *prmFile, char *rtfFile,
                                char *arpmFile, int useHbond);
/**
 * Runs a minimization of the hydrogen bonding network
 * 
 * */
void hydro_min (struct atomgrp * ag, char* pdbFile, char *outnFile, char *outoFile, 
    int maxIter, int useHbond, double *dcHbond, double *dcVdw, double *dcElec);
/**
 * Updates the octree with new atom positions and calculates the energy
 * Gradients can also be calculated
 * 
 * Params:
 * n: total dimensions of atom positions (3 * num atoms)
 * inp: the new array of atom positions
 * prms: parm struct containing energy parms
 * en: double pointer where the calculated energy is stored
 * grad: double pointer where the gradients are stored 
 * */
void my_en_grad( int n, double *inp, void *prms, double *en, double *grad );

/**
 * Creates a my_par struct from the parameters that stores what energetics
 * to calculate, the atomgroup struct, and the octree
 * The octree is also built for the first in the step
 * */
struct my_par *create_par(struct atomgrp *ag, int useVdw, int useElec, 
                         int useHbond ,double *dcVdw, double *dcElec, 
                         double *dcHbond, double aprxVdw, double aprxElec);
            
void free_parm (struct my_par *parm);
void find_neighbors (OCTREE_PARAMS *octpar, double *energy);
void read_fix( char *file, int *nfix, int **fix );
void add_neighbor(struct atom *a, int neighbor_index);
int *get_bondls (struct atomgrp *ag, struct atom *a);
void only_find_neighbors(int n, double *inp, void *prms);

#endif