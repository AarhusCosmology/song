/** @file output.c Documented output module
 *
 * Julien Lesgourgues, 26.08.2010    
 *
 * This module writes the output in files.
 *
 * The following function can be called from other modules or from the main:
 *
 * -# output_init() (must be called after spectra_init())
 * -# output_total_cl_at_l() (can be called even before output_init())
 *
 * No memory needs to be deallocated after that, 
 * hence there is no output_free() routine like in other modules.
 */

#include "output.h"

int output_total_cl_at_l(
			 struct spectra * psp,
			 struct lensing * ple,
			 struct output * pop,
			 int l,
			 double * cl
			 ){

  double ** cl_md_ic; /* array with argument 
			 cl_md_ic[index_mode][index_ic1_ic2*psp->ct_size+index_ct] */

  double ** cl_md;    /* array with argument 
			 cl_md[index_mode][index_ct] */

  int index_mode;

  if (ple->has_lensed_cls == _TRUE_) {
    class_call(lensing_cl_at_l(ple,
			       l,
			       cl),
	       ple->error_message,
	       pop->error_message);
  }
  else {

    class_alloc(cl_md_ic,
		psp->md_size*sizeof(double *),
		pop->error_message);
    
    class_alloc(cl_md,
		psp->md_size*sizeof(double *),
		pop->error_message);
    
    for (index_mode = 0; index_mode < psp->md_size; index_mode++) {

      if (psp->md_size > 1)
	
	class_alloc(cl_md[index_mode],  
		    psp->ct_size*sizeof(double),
		    pop->error_message);	
      
      if (psp->ic_size[index_mode] > 1)
	
	class_alloc(cl_md_ic[index_mode],
		    psp->ic_ic_size[index_mode]*psp->ct_size*sizeof(double),
		    pop->error_message);
    }
    
    class_call(spectra_cl_at_l(psp,
			       (double)l,
			       cl,
			       cl_md,
			       cl_md_ic),
	       psp->error_message,
	       pop->error_message);

    for (index_mode = 0; index_mode < psp->md_size; index_mode++) {
      
      if (psp->md_size > 1) 
	free(cl_md[index_mode]);  
      
      if (psp->ic_size[index_mode] > 1)
	free(cl_md_ic[index_mode]);
      
    }
    
    free(cl_md_ic);
    free(cl_md);
    
  }
  
  return _SUCCESS_;
  
}

/** 
 * This routine writes the output in files.
 *
 * 
 * @param pba Input: pointer to background structure (needed for calling spectra_pk_at_z())
 * @param ppt Input : pointer perturbation structure
 * @param psp Input : pointer to spectra structure
 * @param pop Input : pointer to output structure
 */

int output_init(
		struct background * pba,
		struct perturbs * ppt,
		struct spectra * psp,
		struct nonlinear * pnl,
		struct lensing * ple,
    /* MY MODIFICATIONS */
    struct bispectra * pbi,
    struct fisher * pfi,
    /* END OF MY MODIFICATIONS */
		struct output * pop
		) {

  /** Summary: */

  /** - check that we really want to output at least one spectrum */

  /* MY MODIFICATIONS */
  /* Include Fisher */
  if ((ppt->has_cls == _FALSE_) && (ppt->has_pk_matter == _FALSE_) && (pfi->has_fisher == _FALSE_)) {
    if (pop->output_verbose > 0)
      printf("No spectra nor fisher matrices requested. Output module skipped.\n");
    return _SUCCESS_;
  /* END OF MY MODIFICATIONS */
  }
  else {
    if (pop->output_verbose > 0)
      printf("Writing output files in %s... \n",pop->root);
  }

  /** - deal with all anisotropy power spectra C_l's */

  if (ppt->has_cls == _TRUE_) {

    class_call(output_cl(pba,ppt,psp,ple,pop),
	       pop->error_message,
	       pop->error_message);
  }

  /** - deal with all Fourier matter power spectra P(k)'s */

  if (ppt->has_pk_matter == _TRUE_) {

    class_call(output_pk(pba,ppt,psp,pop),
	       pop->error_message,
	       pop->error_message);
  }

  if (pnl->method > nl_none) {

    class_call(output_pk_nl(pba,pnl,pop),
	       pop->error_message,
	       pop->error_message);
  }

  if (ppt->has_matter_transfers == _TRUE_) {

    class_call(output_tk(pba,ppt,psp,pop),
	       pop->error_message,
	       pop->error_message);
  }

  /* MY MODIFICATIONS */
  if (pfi->has_fisher == _TRUE_) {

    class_call(output_fisher(pbi,pfi,pop),
	       pop->error_message,
	       pop->error_message);
  
  }
  /* END OF MY MODIFICATIONS */
  

  return _SUCCESS_;

}


int output_fisher(
        struct bispectra * pbi,
        struct fisher * pfi,
        struct output * pop
        )
{
  
  FileName filename, filename_lmin, filename_lmax;
  char label[256];
  
  // ============================================================================================
  // =                                   Full fisher matrix                                     =
  // ============================================================================================
  
  FILE * fisher_file;
  
  /* Open the file for the Fisher matrix */
  if ((pfi->l_min_estimator > pfi->l_min) || (pfi->l_max_estimator < pfi->l_max))
    sprintf (filename, "%s%s_%d_%d.dat", pop->root, "fisher",
      MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
  else
    sprintf (filename, "%s%s.dat", pop->root, "fisher");

  class_open (fisher_file, filename, "w", pop->error_message);
  
  /* Print the Fisher matrix to file */
  fprintf (fisher_file, "%s", pfi->info);

  /* Close file */
  fclose (fisher_file);

  // ============================================================================================
  // =                               Same, with lensing variance                                =
  // ============================================================================================

  if (pfi->include_lensing_effects == _TRUE_) {
    
    if ((pfi->l_min_estimator > pfi->l_min) || (pfi->l_max_estimator < pfi->l_max))
      sprintf (filename, "%s%s_%d_%d.dat", pop->root, "fisher_lensvar",
        MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
    else
      sprintf (filename, "%s%s.dat", pop->root, "fisher_lensvar");

    class_open (fisher_file, filename, "w", pop->error_message);
  
    /* Print the Fisher matrix to file */
    fprintf (fisher_file, "%s", pfi->info_lensvar);

    /* Close file */
    fclose (fisher_file);
    
  }

  // ============================================================================================
  // =                            Full Fisher matrix for lmin/lmax                              =
  // ============================================================================================

  FILE * fisher_file_lmin;
  FILE * fisher_file_lmax;

  /* Open the file for the signal to noise as a function of the maximum resolution of the experiment */
  if ((pfi->l_min_estimator > pfi->l_min) || (pfi->l_max_estimator < pfi->l_max)) {
    sprintf (filename_lmax, "%s%s_%d_%d.dat", pop->root, "fisher_lmax",
      MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
    sprintf (filename_lmin, "%s%s_%d_%d.dat", pop->root, "fisher_lmin",
      MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
  }
  else { 
    sprintf (filename_lmax, "%s%s.dat", pop->root, "fisher_lmax");
    sprintf (filename_lmin, "%s%s.dat", pop->root, "fisher_lmin");
  }

  class_open (fisher_file_lmax, filename_lmax, "w", pop->error_message);
  class_open (fisher_file_lmin, filename_lmin, "w", pop->error_message);
  
  // --------------------------------------------------------------------------
  // -                                Print labels                            -
  // --------------------------------------------------------------------------
  
  fprintf (fisher_file_lmax, "%20s ", "l_max");
  for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
    sprintf (label, "F_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
    fprintf (fisher_file_lmax, "%20s ", label);
  }
  for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
    sprintf (label, "F_cum_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
    fprintf (fisher_file_lmax, "%20s ", label);
  }
  fprintf (fisher_file_lmax, "\n");
  
  fprintf (fisher_file_lmin, "%20s ", "l_min");
  for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
    sprintf (label, "F_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
    fprintf (fisher_file_lmin, "%20s ", label);
  }
  for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
    sprintf (label, "F_cum_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
    fprintf (fisher_file_lmin, "%20s ", label);
  }
  fprintf (fisher_file_lmin, "\n");
  
  // ---------------------------------------------------------------------
  // -                           Print values                            -
  // ---------------------------------------------------------------------
  
  /* Fisher as a function of the l_max of the experiment */
  for (int index_l3=0; index_l3 < pfi->l3_size; ++index_l3) {
    
    fprintf (fisher_file_lmax, "%20d ", pfi->l3[index_l3]);

    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
      fprintf (fisher_file_lmax, "%20.7g ", (pfi->fisher_matrix_largest[index_l3][index_ft][index_ft]));

    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
      fprintf (fisher_file_lmax, "%20.7g ", (pfi->fisher_matrix_lmax[index_l3][index_ft][index_ft]));

    fprintf (fisher_file_lmax, "\n");

  } // end of for (index_l3)
  
  for (int index_l1=0; index_l1 < pfi->l1_size; ++index_l1) {
  
    fprintf (fisher_file_lmin, "%20d ", pfi->l1[index_l1]);

    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
      fprintf (fisher_file_lmin, "%20.7g ", (pfi->fisher_matrix_smallest[index_l1][index_ft][index_ft]));

    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
      fprintf (fisher_file_lmin, "%20.7g ", (pfi->fisher_matrix_lmin[index_l1][index_ft][index_ft]));

    fprintf (fisher_file_lmin, "\n");

  } // end of for (index_l1)    
  
  // ---------------------------------------------------------------------------
  // -                            Close files                                  -
  // ---------------------------------------------------------------------------
  
  fclose (fisher_file_lmax);
  fclose (fisher_file_lmin);
  
  
  // ============================================================================================
  // =                               Same, with lensing variance                                =
  // ============================================================================================

  if (pfi->include_lensing_effects == _TRUE_) {

    /* No lmax file for lensing variance */

    if ((pfi->l_min_estimator > pfi->l_min) || (pfi->l_max_estimator < pfi->l_max)) {
      sprintf (filename_lmin, "%s%s_%d_%d.dat", pop->root, "fisher_lensvar_lmin",
        MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
    }
    else { 
      sprintf (filename_lmin, "%s%s.dat", pop->root, "fisher_lensvar_lmin");
    }

    class_open (fisher_file_lmin, filename_lmin, "w", pop->error_message);
  
    // --------------------------------------------------------------------------
    // -                                Print labels                            -
    // --------------------------------------------------------------------------
  
    fprintf (fisher_file_lmin, "%20s ", "l_min");
    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
      sprintf (label, "F_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
      fprintf (fisher_file_lmin, "%20s ", label);
    }
    for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
      sprintf (label, "F_cum_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
      fprintf (fisher_file_lmin, "%20s ", label);
    }
    fprintf (fisher_file_lmin, "\n");
  
    // ---------------------------------------------------------------------
    // -                           Print values                            -
    // ---------------------------------------------------------------------
  
    for (int index_l1=0; index_l1 < pfi->l1_size; ++index_l1) {
  
      fprintf (fisher_file_lmin, "%20d ", pfi->l1[index_l1]);

      for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
        fprintf (fisher_file_lmin, "%20.7g ", (pfi->fisher_matrix_lensvar_smallest[index_l1][index_ft][index_ft]));

      for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
        fprintf (fisher_file_lmin, "%20.7g ", (pfi->fisher_matrix_lensvar_lmin[index_l1][index_ft][index_ft]));

      fprintf (fisher_file_lmin, "\n");

    } // end of for (index_l1)    
  
    // ---------------------------------------------------------------------------
    // -                            Close files                                  -
    // ---------------------------------------------------------------------------
  
    fclose (fisher_file_lmin);

  } // end of if(include_lensing_effects)
  



  // ============================================================================================
  // =                           Fisher matrix for XYZ and lmin/lmax                            =
  // ============================================================================================

  if (pfi->fisher_verbose > 2) {

    /* One file per XYZ bispectrum, so that if you ask for temperature and polarisation, 8 files
    (TTT, EEE, TEE, ...) will be produced */
    FILE **** fisher_file_XYZ_lmin, **** fisher_file_XYZ_lmax;

    class_alloc (fisher_file_XYZ_lmin, pbi->bf_size*sizeof(FILE ***), pop->error_message);
    class_alloc (fisher_file_XYZ_lmax, pbi->bf_size*sizeof(FILE ***), pop->error_message);

    for (int X=0; X < pbi->bf_size; ++X) {

      class_alloc (fisher_file_XYZ_lmin[X], pbi->bf_size*sizeof(FILE **), pop->error_message);
      class_alloc (fisher_file_XYZ_lmax[X], pbi->bf_size*sizeof(FILE **), pop->error_message);

      for (int Y=0; Y < pbi->bf_size; ++Y) {

        class_alloc (fisher_file_XYZ_lmin[X][Y], pbi->bf_size*sizeof(FILE *), pop->error_message);
        class_alloc (fisher_file_XYZ_lmax[X][Y], pbi->bf_size*sizeof(FILE *), pop->error_message);
      }
    }
  
    /* Copy the code above for l_min and l_max, but now for each bispectrum XYZ */
    for (int X=0; X < pbi->bf_size; ++X) {
      for (int Y=0; Y < pbi->bf_size; ++Y) {
        for (int Z=0; Z < pbi->bf_size; ++Z) {
  
          /* Open the file for the signal to noise as a function of the maximum resolution of the experiment */
          if ((pfi->l_min_estimator > pfi->l_min) || (pfi->l_max_estimator < pfi->l_max)) {
            sprintf (filename_lmax, "%s%s_%s_%d_%d.dat",
              pop->root, "fisher_lmax", pbi->bfff_labels[X][Y][Z],
                MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
            sprintf (filename_lmin, "%s%s_%s_%d_%d.dat",
              pop->root, "fisher_lmin", pbi->bfff_labels[X][Y][Z],
                MAX(pfi->l_min, pfi->l_min_estimator), MIN(pfi->l_max, pfi->l_max_estimator));
          }
          else {
            sprintf (filename_lmax, "%s%s_%s.dat", pop->root, "fisher_lmax", pbi->bfff_labels[X][Y][Z]);
            sprintf (filename_lmin, "%s%s_%s.dat", pop->root, "fisher_lmin", pbi->bfff_labels[X][Y][Z]);
          }

          class_open (fisher_file_XYZ_lmax[X][Y][Z], filename_lmax, "w", pop->error_message);
          class_open (fisher_file_XYZ_lmin[X][Y][Z], filename_lmin, "w", pop->error_message);

          
          // --------------------------------------------------------------------------
          // -                                Print labels                            -
          // --------------------------------------------------------------------------
          
          fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20s ", "l_max");
          for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
            sprintf (label, "F_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
            fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20s ", label);
          }
          for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
            sprintf (label, "F_cum_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
            fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20s ", label);
          }
          fprintf (fisher_file_XYZ_lmax[X][Y][Z], "\n");

            
          fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20s ", "l_min");
          for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
            sprintf (label, "F_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
            fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20s ", label);
          }
          for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft) {
            sprintf (label, "F_cum_%s", pbi->bt_labels[pfi->index_bt_of_ft[index_ft]]);
            fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20s ", label);
          }
          fprintf (fisher_file_XYZ_lmin[X][Y][Z], "\n");
          
          // ---------------------------------------------------------------------
          // -                           Print values                            -
          // ---------------------------------------------------------------------
          
          /* Fisher as a function of the l_max of the experiment */
          for (int index_l3=0; index_l3 < pfi->l3_size; ++index_l3) {
            
            fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20d ", pfi->l3[index_l3]);
          
            for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
              fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20.7g ",
                (pfi->fisher_matrix_XYZ_largest[X][Y][Z][index_l3][index_ft][index_ft]));
          
            for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
              fprintf (fisher_file_XYZ_lmax[X][Y][Z], "%20.7g ",
                (pfi->fisher_matrix_XYZ_lmax[X][Y][Z][index_l3][index_ft][index_ft]));
          
            fprintf (fisher_file_XYZ_lmax[X][Y][Z], "\n");
          
          } // end of for (index_l3)
            
          for (int index_l1=0; index_l1 < pfi->l1_size; ++index_l1) {
          
            fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20d ", pfi->l1[index_l1]);
        
            for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
              fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20.7g ",
                (pfi->fisher_matrix_XYZ_smallest[X][Y][Z][index_l1][index_ft][index_ft]));
        
            for (int index_ft=0; index_ft < pfi->fisher_size; ++index_ft)
              fprintf (fisher_file_XYZ_lmin[X][Y][Z], "%20.7g ",
                (pfi->fisher_matrix_XYZ_lmin[X][Y][Z][index_l1][index_ft][index_ft]));
        
            fprintf (fisher_file_XYZ_lmin[X][Y][Z], "\n");
          }
  
          // ---------------------------------------------------------------------------
          // -                            Close files                                  -
          // ---------------------------------------------------------------------------
  
          fclose (fisher_file_XYZ_lmax[X][Y][Z]);
          fclose (fisher_file_XYZ_lmin[X][Y][Z]);
  
        } // end of for Z
      } // end of for Y
    } // end of for X
  } // end of if(fisher_verbose>1)

  
  return _SUCCESS_;
  
}


/** 
 * This routines writes the output in files for anisotropy power spectra C_l's.
 *
 * @param pba Input: pointer to background structure (needed for T_cmb)
 * @param ppt Input : pointer perturbation structure
 * @param psp Input : pointer to spectra structure
 * @param ple Input : pointer to lensing structure
 * @param pop Input : pointer to output structure
 */

int output_cl(
	      struct background * pba,
	      struct perturbs * ppt,
	      struct spectra * psp,
	      struct lensing * ple,
	      struct output * pop
	      ) {

  /** Summary: */

  /** - define local variables */

  FILE *** out_md_ic; /* array of pointers to files with argument 
			 out_md_ic[index_mode][index_ic1_ic2] 
			 (will contain cl's for each mode and pairs of initial conditions) */

  FILE ** out_md;     /* array of pointers to files with argument 
			 out_md[index_mode] 
			 (will contain cl's for each mode, summed eventually over ic's) */

  FILE * out;         /* (will contain total cl's, summed eventually over modes and ic's) */

  FILE * out_lensed;         /* (will contain total lensed cl's) */

  double ** cl_md_ic; /* array with argument 
			 cl_md_ic[index_mode][index_ic1_ic2*psp->ct_size+index_ct] */

  double ** cl_md;    /* array with argument 
			 cl_md[index_mode][index_ct] */

  double * cl_tot;    /* array with argument 
			 cl_tot[index_ct] */

  int index_mode;
  int index_ic1,index_ic2,index_ic1_ic2;
  int l;

  FileName file_name;
  char first_line[_LINE_LENGTH_MAX_];

  /** - first, allocate all arrays of files and cls */

  class_alloc(out_md_ic,
	      psp->md_size*sizeof(FILE * *),
	      pop->error_message);
        
  class_alloc(cl_md_ic,
	      psp->md_size*sizeof(double *),
	      pop->error_message);

  class_alloc(out_md,
	      psp->md_size*sizeof(FILE *),
	      pop->error_message);
    
  class_alloc(cl_md,
	      psp->md_size*sizeof(double *),
	      pop->error_message);

  // *** MY MODIFICATIONS ***

  /* Allocate array for derivatives of C_l's */
  double ** dcl_md;
  double ** dcl_md_ic;
  
  if (psp->compute_cl_derivative == _TRUE_) {

    class_alloc(dcl_md_ic,
  	      psp->md_size*sizeof(double *),
  	      pop->error_message);

    class_alloc(dcl_md,
  	      psp->md_size*sizeof(double *),
  	      pop->error_message);
  }

  // *** END MY MODIFICATIONS ***    


  for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {

    class_alloc(out_md_ic[index_mode],
		psp->ic_ic_size[index_mode]*sizeof(FILE *),
		pop->error_message);
      
  }

  /** - second, open only the relevant files, and write a heading in each of them */

  sprintf(file_name,"%s%s",pop->root,"cl.dat");

  class_call(output_open_cl_file(psp,
				 pop,
				 &out,
				 file_name,
				 "total [l(l+1)/2pi] C_l's",
				 psp->l_max_tot
				 ),
	     pop->error_message,
	     pop->error_message);
   
  class_alloc(cl_tot,
	      psp->ct_size*sizeof(double),
	      pop->error_message);

  // *** MY MODIFICATIONS ***

  /* Allocate array for derivatives of C_l's */
  double * dcl_tot;
  
  if (psp->compute_cl_derivative == _TRUE_) {

    class_alloc(dcl_tot,
  	      psp->ct_size*sizeof(double),
  	      pop->error_message);
          
  }

  // *** END MY MODIFICATIONS ***    

  if (ple->has_lensed_cls == _TRUE_) {
    
    sprintf(file_name,"%s%s",pop->root,"cl_lensed.dat");
    
    class_call(output_open_cl_file(psp,
				   pop,
				   &out_lensed,
				   file_name,
				   "total lensed [l(l+1)/2pi] C_l's",
				   ple->l_lensed_max
				   ),
	       pop->error_message,
	       pop->error_message);
  }


  // *** MY MODIFICATIONS ***

  /* Create file for derivatives of C_l's */
  FILE * out_dcl;
  
  if (psp->compute_cl_derivative == _TRUE_) {

    sprintf(file_name,"%s%s",pop->root,"dcl.dat");
    
    class_call(output_open_cl_file(psp,
				   pop,
				   &out_dcl,
				   file_name,
           "total dln(l*l*C_l)/dln(l*l)",
				   ple->l_lensed_max
				   ),
	       pop->error_message,
	       pop->error_message);          
  }
  // *** END MY MODIFICATIONS ***    


  if (ppt->md_size > 1) {

    for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
	
      if ((ppt->has_scalars) && (index_mode == ppt->index_md_scalars)) {
	  
	sprintf(file_name,"%s%s",pop->root,"cls.dat");
	strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar mode");

      }

      if ((ppt->has_tensors) && (index_mode == ppt->index_md_tensors)) {
	  	  
	sprintf(file_name,"%s%s",pop->root,"clt.dat");
	strcpy(first_line,"[l(l+1)/2pi] C_l's for tensor mode");

      }
	
      class_call(output_open_cl_file(psp,
				     pop,
				     &(out_md[index_mode]),
				     file_name,
				     first_line,
				     psp->l_max[index_mode]
				     ),
		 pop->error_message,
		 pop->error_message);
	
      class_alloc(cl_md[index_mode],
		  psp->ct_size*sizeof(double),
		  pop->error_message);

    // *** MY MODIFICATIONS ***

    /* Allocate array for derivatives of C_l's */
    if (psp->compute_cl_derivative == _TRUE_) {

      class_alloc(dcl_md[index_mode],
		  psp->ct_size*sizeof(double),
		  pop->error_message);
  
    }

    // *** END MY MODIFICATIONS ***    


    }
  }

  for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {

    if (ppt->ic_size[index_mode] > 1) {

      for (index_ic1 = 0; index_ic1 < ppt->ic_size[index_mode]; index_ic1++) {
	  
	for (index_ic2 = index_ic1; index_ic2 < ppt->ic_size[index_mode]; index_ic2++) {
	    
	  if ((ppt->has_scalars) && (index_mode == ppt->index_md_scalars)) {

	    if ((ppt->has_ad) && 
		(index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_ad)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_ad.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar adiabatic (AD) mode");
	    }

	    if ((ppt->has_bi) && 
		(index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_bi)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_bi.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar baryon isocurvature (BI) mode");
	    }
	      
	    if ((ppt->has_cdi) && 
		(index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_cdi)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_cdi.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar CDM isocurvature (CDI) mode");
	    }

	    if ((ppt->has_nid) && 
		(index_ic1 == ppt->index_ic_nid) && (index_ic2 == ppt->index_ic_nid)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_nid.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar neutrino density isocurvature (NID) mode");
	    }

	    if ((ppt->has_niv) && 
		(index_ic1 == ppt->index_ic_niv) && (index_ic2 == ppt->index_ic_niv)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_niv.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar neutrino velocity isocurvature (NIV) mode");
	    }

	    if ((ppt->has_ad) && 
		(ppt->has_bi) && (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_bi)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_ad_bi.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross ADxBI mode");
	    }
	      
	    if ((ppt->has_ad) && (ppt->has_cdi) && 
		(index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_cdi)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_ad_cdi.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross ADxCDI mode");
	    }

	    if ((ppt->has_ad) && (ppt->has_nid) && 
		(index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_nid)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_ad_nid.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross ADxNID mode");
	    }

	    if ((ppt->has_ad) && (ppt->has_niv) && 
		(index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_niv)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_ad_niv.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross ADxNIV mode");
	    }

	    if ((ppt->has_bi) && (ppt->has_cdi) && 
		(index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_cdi)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_bi_cdi.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross BIxCDI mode");
	    }

	    if ((ppt->has_bi) && (ppt->has_nid) && 
		(index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_nid)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_bi_nid.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross BIxNID mode");
	    }

	    if ((ppt->has_bi) && (ppt->has_niv) && 
		(index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_niv)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_bi_niv.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross BIxNIV mode");
	    }

	    if ((ppt->has_cdi) && (ppt->has_nid) && 
		(index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_nid)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_cdi_nid.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross CDIxNID mode");
	    }

	    if ((ppt->has_cdi) && (ppt->has_niv) && 
		(index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_niv)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_cdi_niv.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross CDIxNIV mode");
	    }

	    if ((ppt->has_nid) && (ppt->has_niv) && 
		(index_ic1 == ppt->index_ic_nid) && (index_ic2 == ppt->index_ic_niv)) {

	      sprintf(file_name,"%s%s",pop->root,"cls_nid_niv.dat");
	      strcpy(first_line,"[l(l+1)/2pi] C_l's for scalar cross NIDxNIV mode");
	    }

	  }

	  if ((ppt->has_tensors) && (index_mode == ppt->index_md_tensors)) {

	    class_test(0==0,
		       pop->error_message,
		       "Seems that we have mixed initial conditions for tensors? Should not happen!\n");
	      
	  }

	  index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,psp->ic_size[index_mode]);

	  if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {

	    class_call(output_open_cl_file(psp,
					   pop,
					   &(out_md_ic[index_mode][index_ic1_ic2]),
					   file_name,
					   first_line,
					   psp->l_max[index_mode]
					   ),
		       pop->error_message,
		       pop->error_message);

	  }
	}
      }

      class_alloc(cl_md_ic[index_mode],
		  psp->ic_ic_size[index_mode]*psp->ct_size*sizeof(double),
		  pop->error_message);
      
      /* Allocate array for derivatives of C_l's */
      if (psp->compute_cl_derivative == _TRUE_) {

        class_alloc(dcl_md_ic[index_mode],
  		  psp->ic_ic_size[index_mode]*psp->ct_size*sizeof(double),
  		  pop->error_message);
  
      }

      
    }
  }

  /** - third, perform loop over l. For each multipole, get all C_l's
      by calling spectra_cl_at_l() and distribute the results to 
      relevant files */

  for (l = 2; l <= psp->l_max_tot; l++) {  

    /* Factor multiplying all outputted C_l's */
    double factor = l*(l+1)/2./_PI_;

    class_call(spectra_cl_at_l(psp,(double)l,cl_tot,cl_md,cl_md_ic),
	       psp->error_message,
	       pop->error_message);


    class_call(output_one_line_of_cl(pba,psp,pop,out,(double)l,cl_tot,NULL,factor,psp->ct_size),
	       pop->error_message,
	       pop->error_message);


    // *** MY MODIFICATIONS ***

    /* Interpolate array with the derivatives of C_l's and output to file */
  
    if (psp->compute_cl_derivative == _TRUE_) { 

      /* The logarithmic derivative should never be multiplied by any factor,
      as it is naturally of order one */
      double factor = 1;
      
      class_call(spectra_dcl_at_l(psp,(double)l,dcl_tot,dcl_md,dcl_md_ic),
  	       psp->error_message,
  	       pop->error_message);

      class_call(output_one_line_of_cl(pba,psp,pop,out_dcl,(double)l,cl_tot,dcl_tot,factor,psp->ct_size),
  	       pop->error_message,
  	       pop->error_message);      
    }

    // *** END OF MY MODIFICATIONS ***


    if ((ple->has_lensed_cls == _TRUE_) && (l<=ple->l_lensed_max)) {

      class_call(lensing_cl_at_l(ple,
				 (double)l,
				 cl_tot),
		 ple->error_message,
		 pop->error_message);

      class_call(output_one_line_of_cl(pba,psp,pop,out_lensed,l,cl_tot,NULL,factor,psp->ct_size),
		 pop->error_message,
		 pop->error_message);
    }

    if (ppt->md_size > 1) {
      for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
	if (l <= psp->l_max[index_mode]) {

	  class_call(output_one_line_of_cl(pba,psp,pop,out_md[index_mode],l,cl_md[index_mode],NULL,factor,psp->ct_size),
		     pop->error_message,
		     pop->error_message);
	}
      }
    }
	  
    for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
      if ((ppt->ic_size[index_mode] > 1) && (l <= psp->l_max[index_mode])) {
	for (index_ic1_ic2 = 0; index_ic1_ic2 < psp->ic_ic_size[index_mode]; index_ic1_ic2++) {
	  if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {

	    class_call(output_one_line_of_cl(pba,psp,pop,out_md_ic[index_mode][index_ic1_ic2],l,&(cl_md_ic[index_mode][index_ic1_ic2*psp->ct_size]),NULL,factor,psp->ct_size),
		       pop->error_message,
		       pop->error_message);
	  }
	}
      }
    }
  }

  /** - finally, close files and free arrays of files and cls */

  for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
    if (ppt->ic_size[index_mode] > 1) {
      for (index_ic1_ic2 = 0; index_ic1_ic2 < psp->ic_ic_size[index_mode]; index_ic1_ic2++) {
	if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {
	  fclose(out_md_ic[index_mode][index_ic1_ic2]);
	}
      }
      free(cl_md_ic[index_mode]);
    }
  }
  if (ppt->md_size > 1) {
    for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
      fclose(out_md[index_mode]);
      free(cl_md[index_mode]);
    }
  }
  fclose(out);
  if (ple->has_lensed_cls == _TRUE_) {
    fclose(out_lensed);
  }
  // *** MY MODIFICATIONS ***
  if (psp->compute_cl_derivative == _TRUE_) {
    fclose(out_dcl);
    free(dcl_tot);
    free(dcl_md_ic);
    free(dcl_md);
  }
  // *** END OF MY MODIFICATIONS ***
  free(cl_tot);
  for (index_mode = 0; index_mode < ppt->md_size; index_mode++) {
    free(out_md_ic[index_mode]);
  }
  free(out_md_ic);
  free(cl_md_ic);
  free(out_md);
  free(cl_md);
      
  return _SUCCESS_;

} 














/** 
 * This routines writes the output in files for Fourier matter power spectra P(k)'s.
 * 
 * @param pba Input: pointer to background structure (needed for calling spectra_pk_at_z())
 * @param ppt Input : pointer perturbation structure
 * @param psp Input : pointer to spectra structure
 * @param pop Input : pointer to output structure
 */

int output_pk(
	      struct background * pba,
	      struct perturbs * ppt,
	      struct spectra * psp,
	      struct output * pop
	      ) {

  /** Summary: */

  /** - define local variables */

  FILE ** out_ic=NULL; /* array of pointers to files with argument 
		     out_ic[index_ic1_ic2] 
		     (will contain P(k)'s for each pair of initial conditions) */
  
  FILE * out;     /* (will contain total P(k) summed eventually over initial conditions) */
  
  double * pk_ic=NULL;  /* array with argument 
		      pk_ic[index_k * psp->ic_ic_size[index_mode] + index_ic1_ic2] */

  double * pk_tot; /* array with argument 
		      pk_tot[index_k] */

  int index_mode;
  int index_ic1,index_ic2;
  int index_ic1_ic2=0;
  int index_k;
  int index_z;

  FileName file_name;
  FileName redshift_suffix;
  char first_line[_LINE_LENGTH_MAX_];
    
  index_mode=ppt->index_md_scalars;

  for (index_z = 0; index_z < pop->z_pk_num; index_z++) {

    /** - first, check that requested redshift z_pk is consistent */

    class_test((pop->z_pk[index_z] > psp->z_max_pk),
	       pop->error_message,
	       "P(k,z) computed up to z=%f but requested at z=%f. Must increase z_max_pk in precision file.",psp->z_max_pk,pop->z_pk[index_z]);

    if (pop->z_pk_num == 1) 
      redshift_suffix[0]='\0';
    else
      sprintf(redshift_suffix,"z%d_",index_z+1);

    /** - second, open only the relevant files, and write a heading in each of them */
    
    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk.dat");

    class_call(output_open_pk_file(pba,
				   psp,
				   pop,
				   &out,
				   file_name,
				   "",
				   pop->z_pk[index_z]
				   ),
	       pop->error_message,
	       pop->error_message);
   
    class_alloc(pk_tot,
		psp->ln_k_size*sizeof(double),
		pop->error_message);

    if (psp->ic_size[index_mode] > 1) {

      class_alloc(out_ic,
		  psp->ic_ic_size[index_mode]*sizeof(FILE *),
		  pop->error_message);

      class_alloc(pk_ic,
		  psp->ln_k_size*psp->ic_ic_size[index_mode]*sizeof(double),
		  pop->error_message);

      for (index_ic1 = 0; index_ic1 < ppt->ic_size[index_mode]; index_ic1++) {
	  
	for (index_ic2 = index_ic1; index_ic2 < ppt->ic_size[index_mode]; index_ic2++) {
	  
	  if ((ppt->has_ad) && 
	      (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_ad)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_ad.dat");
	    strcpy(first_line,"for adiabatic (AD) mode ");
	  }

	  if ((ppt->has_bi) && 
	      (index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_bi)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_bi.dat");
	    strcpy(first_line,"for baryon isocurvature (BI) mode ");
	  }
	  
	  if ((ppt->has_cdi) && 
	      (index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_cdi)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_cdi.dat");
	    strcpy(first_line,"for CDM isocurvature (CDI) mode ");
	  }
	  
	  if ((ppt->has_nid) && 
	      (index_ic1 == ppt->index_ic_nid) && (index_ic2 == ppt->index_ic_nid)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nid.dat");
	    strcpy(first_line,"for neutrino density isocurvature (NID) mode ");
	  }
	  
	  if ((ppt->has_niv) && 
	      (index_ic1 == ppt->index_ic_niv) && (index_ic2 == ppt->index_ic_niv)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_niv.dat");
	    strcpy(first_line,"for neutrino velocity isocurvature (NIV) mode ");
	  }
	  
	  if ((ppt->has_ad) && 
	      (ppt->has_bi) && (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_bi)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_ad_bi.dat");
	    strcpy(first_line,"for cross ADxBI mode ");
	  }
	  
	  if ((ppt->has_ad) && (ppt->has_cdi) && 
	      (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_cdi)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_ad_cdi.dat");
	    strcpy(first_line,"for cross ADxCDI mode ");
	  }
	  
	  if ((ppt->has_ad) && (ppt->has_nid) && 
	      (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_nid)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_ad_nid.dat");
	    strcpy(first_line,"for scalar cross ADxNID mode ");
	  }
	  
	  if ((ppt->has_ad) && (ppt->has_niv) && 
	      (index_ic1 == ppt->index_ic_ad) && (index_ic2 == ppt->index_ic_niv)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_ad_niv.dat");
	    strcpy(first_line,"for cross ADxNIV mode ");
	  }
	  
	  if ((ppt->has_bi) && (ppt->has_cdi) && 
	      (index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_cdi)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_bi_cdi.dat");
	    strcpy(first_line,"for cross BIxCDI mode ");
	  }
	  
	  if ((ppt->has_bi) && (ppt->has_nid) && 
	      (index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_nid)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_bi_nid.dat");
	    strcpy(first_line,"for cross BIxNID mode ");
	  }
	  
	  if ((ppt->has_bi) && (ppt->has_niv) && 
	      (index_ic1 == ppt->index_ic_bi) && (index_ic2 == ppt->index_ic_niv)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_bi_niv.dat");
	    strcpy(first_line,"for cross BIxNIV mode ");
	  }
	  
	  if ((ppt->has_cdi) && (ppt->has_nid) && 
	      (index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_nid)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_cdi_nid.dat");
	    strcpy(first_line,"for cross CDIxNID mode ");
	  }
	  
	  if ((ppt->has_cdi) && (ppt->has_niv) && 
	      (index_ic1 == ppt->index_ic_cdi) && (index_ic2 == ppt->index_ic_niv)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_cdi_niv.dat");
	    strcpy(first_line,"for cross CDIxNIV mode ");
	  }
	  
	  if ((ppt->has_nid) && (ppt->has_niv) && 
	      (index_ic1 == ppt->index_ic_nid) && (index_ic2 == ppt->index_ic_niv)) {
	    
	    sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nid_niv.dat");
	    strcpy(first_line,"for cross NIDxNIV mode ");
	  }

	  index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic2,psp->ic_size[index_mode]);

	  if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {

	    class_call(output_open_pk_file(pba,
					   psp,
					   pop,
					   &(out_ic[index_ic1_ic2]),
					   file_name,
					   first_line,
					   pop->z_pk[index_z]
					   ),
		       pop->error_message,
		       pop->error_message);
	  }
	} 
      }
    }
  
    /** - third, compute P(k) for each k (if several ic's, compute it for each ic and compute also the total); if z_pk = 0, this is done by directly reading inside the pre-computed table; if not, this is done by interpolating the table at the correct value of tau. */
  
    /* if z_pk = 0, no interpolation needed */

    if (pop->z_pk[index_z] == 0.) {

      for (index_k=0; index_k<psp->ln_k_size; index_k++) {

	if (psp->ic_size[index_mode] == 1) {
	  pk_tot[index_k] = exp(psp->ln_pk[(psp->ln_tau_size-1) * psp->ln_k_size + index_k]);
	}
	else {
	  pk_tot[index_k] = 0.;
	  for (index_ic1=0; index_ic1 < psp->ic_size[index_mode]; index_ic1++) {
	    index_ic1_ic2 = index_symmetric_matrix(index_ic1,index_ic1,psp->ic_size[index_mode]);
	    pk_ic[index_k * psp->ic_ic_size[index_mode] + index_ic1_ic2] = exp(psp->ln_pk[((psp->ln_tau_size-1) * psp->ln_k_size + index_k) * psp->ic_ic_size[index_mode] + index_ic1_ic2]);
	    pk_tot[index_k] += pk_ic[index_k * psp->ic_ic_size[index_mode] + index_ic1_ic2];
	  }
	  for (index_ic1=0; index_ic1 < psp->ic_size[index_mode]; index_ic1++) {
	    for (index_ic2 = index_ic1+1; index_ic2 < psp->ic_size[index_mode]; index_ic2++) {
	      pk_ic[index_k * psp->ic_ic_size[index_mode] + index_symmetric_matrix(index_ic1,index_ic2,psp->ic_size[index_mode])] = 
		psp->ln_pk[index_k * psp->ic_ic_size[index_mode] + index_symmetric_matrix(index_ic1,index_ic2,psp->ic_size[index_mode])]
		*sqrt(pk_ic[index_k * psp->ic_ic_size[index_mode] + index_symmetric_matrix(index_ic1,index_ic1,psp->ic_size[index_mode])] *
		      pk_ic[index_k * psp->ic_ic_size[index_mode] + index_symmetric_matrix(index_ic2,index_ic2,psp->ic_size[index_mode])]);
	      pk_tot[index_k] += 2.*pk_ic[index_k * psp->ic_ic_size[index_mode] + index_ic1_ic2];
	    }
	  }
	}
      }
    }
      
    /* if 0 <= z_pk <= z_max_pk, interpolation needed, */
    else {

      class_call(spectra_pk_at_z(pba,
				 psp,
				 linear,
				 pop->z_pk[index_z],
				 pk_tot,
				 pk_ic),
		 psp->error_message,
		 pop->error_message);
    }

    /** - fourth, write in files */

    for (index_k=0; index_k<psp->ln_k_size; index_k++) {
	
      class_call(output_one_line_of_pk(out,
				       exp(psp->ln_k[index_k])/pba->h,
				       pk_tot[index_k]*pow(pba->h,3)),
		 pop->error_message,
		 pop->error_message);

      if (psp->ic_size[index_mode] > 1) {
	  
	for (index_ic1_ic2 = 0; index_ic1_ic2 < psp->ic_ic_size[index_mode]; index_ic1_ic2++) {

	  if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {

	    class_call(output_one_line_of_pk(out_ic[index_ic1_ic2],
					     exp(psp->ln_k[index_k])/pba->h,
					     pk_ic[index_k * psp->ic_ic_size[index_mode] + index_ic1_ic2]*pow(pba->h,3)),
		       pop->error_message,
		       pop->error_message);
	  }
	}
      }
    }

    /** - fifth, free memory and close files */

    free(pk_tot);
    fclose(out);

    if (psp->ic_size[index_mode] > 1) {
      for (index_ic1_ic2 = 0; index_ic1_ic2 < psp->ic_ic_size[index_mode]; index_ic1_ic2++) {
	if (psp->is_non_zero[index_mode][index_ic1_ic2] == _TRUE_) {
	  fclose(out_ic[index_ic1_ic2]);
	}
      }
      free(out_ic);
      free(pk_ic);
    }
  
  }
  
  return _SUCCESS_;

}

int output_pk_nl(
		 struct background * pba,
		 struct nonlinear * pnl,
		 struct output * pop
		 ) {

  int index_z;
  int index_k;
  double * pz_density;
  double * pz_velocity;
  double * pz_cross;
  FILE * out_density;
  FILE * out_velocity;
  FILE * out_cross;
  FileName file_name;
  FileName redshift_suffix;
  int k_size_at_z;

  class_alloc(pz_density,pnl->k_size[0]*sizeof(double),pnl->error_message);

  if ((pnl->method >= nl_trg_linear) && (pnl->method <= nl_trg)) {
    class_alloc(pz_velocity,pnl->k_size[0]*sizeof(double),pnl->error_message);
    class_alloc(pz_cross,pnl->k_size[0]*sizeof(double),pnl->error_message);
  }

  for (index_z = 0; index_z < pop->z_pk_num; index_z++) {

    class_test((pop->z_pk[index_z] < pnl->z[pnl->z_size-1]) || (pop->z_pk[index_z] > pnl->z[0]),
	       pop->error_message,
	       "P_nl(k,z) computed in range %f<=z<=%f but requested at z=%f. You should probably increase z_ini in precision file.",pnl->z[pnl->z_size-1],pnl->z[0],pop->z_pk[index_z]);

    if (pop->z_pk_num == 1) 
      redshift_suffix[0]='\0';
    else
      sprintf(redshift_suffix,"z%d_",index_z+1);

    /** - second, open only the relevant files, and write a heading in each of them */
    
    class_call(nonlinear_pk_at_z(pnl,pop->z_pk[index_z],pz_density,pz_velocity,pz_cross,&k_size_at_z),
	       pop->error_message,
	       pop->error_message);

    if (pnl->method == nl_halofit) {

      sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nl.dat");

      class_call(output_open_pk_nl_file(pba,
					pnl,
					pop,
					&out_density,
					file_name,
					"(using HALOFIT) ",
					pop->z_pk[index_z],
					k_size_at_z
					),
		 pop->error_message,
		 pop->error_message);
    }

    if ((pnl->method >= nl_trg_linear) && (pnl->method <= nl_trg)) {

      sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nl_density.dat");

      class_call(output_open_pk_nl_file(pba,
					pnl,
					pop,
					&out_density,
					file_name,
					"(density auto-correlation) ",
					pop->z_pk[index_z],
					k_size_at_z
					),
		 pop->error_message,
		 pop->error_message);

      sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nl_velocity.dat");

      class_call(output_open_pk_nl_file(pba,
					pnl,
					pop,
					&out_velocity,
					file_name,
					"(velocity auto-correlation) ",
					pop->z_pk[index_z],
					k_size_at_z
					),
		 pop->error_message,
		 pop->error_message);
      
      sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"pk_nl_cross.dat");
      
      class_call(output_open_pk_nl_file(pba,
					pnl,
					pop,
					&out_cross,
					file_name,
					"(density-velocity cross-correlation) ",
					pop->z_pk[index_z],
					k_size_at_z
					),
		 pop->error_message,
		 pop->error_message);
    }

    for (index_k=0; index_k<k_size_at_z;index_k++) {

      class_call(output_one_line_of_pk(out_density,
				       pnl->k[index_k]/pba->h,
				       pz_density[index_k]*pow(pba->h,3)),
		 pop->error_message,
		 pop->error_message);

      if ((pnl->method >= nl_trg_linear) && (pnl->method <= nl_trg)) {

	class_call(output_one_line_of_pk(out_velocity,
					 pnl->k[index_k]/pba->h,
					 pz_velocity[index_k]*pow(pba->h,3)),
		   pop->error_message,
		   pop->error_message);
	
	class_call(output_one_line_of_pk(out_cross,
					 pnl->k[index_k]/pba->h,
					 pz_cross[index_k]*pow(pba->h,3)),
		   pop->error_message,
		   pop->error_message);
      }
    }

    fclose(out_density);

    if ((pnl->method >= nl_trg_linear) && (pnl->method <= nl_trg)) {          
      fclose(out_velocity);
      fclose(out_cross);
    }
 }

  free(pz_density);
  if ((pnl->method >= nl_trg_linear) && (pnl->method <= nl_trg)) {
    free(pz_velocity);
    free(pz_cross);
  }

  return _SUCCESS_;
}

/** 
 * This routines writes the output in files for matter transfer functions T_i(k)'s.
 * 
 * @param pba Input: pointer to background structure (needed for calling spectra_pk_at_z())
 * @param ppt Input : pointer perturbation structure
 * @param psp Input : pointer to spectra structure
 * @param pop Input : pointer to output structure
 */

int output_tk(
	      struct background * pba,
	      struct perturbs * ppt,
	      struct spectra * psp,
	      struct output * pop
	      ) {

  /** Summary: */

  /** - define local variables */

  FILE ** out_ic; /* array of pointers to files with argument 
		     out_ic[index_ic] 
		     (will contain T_i(k)'s for each initial conditions) */
    
  double * tk;  /* array with argument 
		      pk_ic[(index_k * psp->ic_size[index_mode] + index_ic)*psp->tr_size+index_tr] */

  double * tk_cmbfast = NULL; /* array with argument tk_cmbfast[index_tr] */
			 

  int index_mode;
  int index_ic;
  int index_k;
  int index_z;
  int index_tr;
  int n_ncdm;

  FileName file_name;
  FileName redshift_suffix;
  char first_line[_LINE_LENGTH_MAX_];
    
  index_mode=ppt->index_md_scalars;

  for (index_z = 0; index_z < pop->z_pk_num; index_z++) {

    /** - first, check that requested redshift z_pk is consistent */

    class_test((pop->z_pk[index_z] > psp->z_max_pk),
	       pop->error_message,
	       "T_i(k,z) computed up to z=%f but requested at z=%f. Must increase z_max_pk in precision file.",psp->z_max_pk,pop->z_pk[index_z]);

    if (pop->z_pk_num == 1) 
      redshift_suffix[0]='\0';
    else
      sprintf(redshift_suffix,"z%d_",index_z+1);

    /** - second, open only the relevant files, and write a heading in each of them */
    
    class_alloc(out_ic,
		psp->ic_size[index_mode]*sizeof(FILE *),
		pop->error_message);
    
    class_alloc(tk,
		psp->ln_k_size*psp->ic_size[index_mode]*psp->tr_size*sizeof(double),
		pop->error_message);

    for (index_ic = 0; index_ic < ppt->ic_size[index_mode]; index_ic++) {
	  
      if ((ppt->has_ad) && (index_ic == ppt->index_ic_ad)) {
	
	if (ppt->ic_size[index_mode] == 1)
	  sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk.dat");
	else
	  sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk_ad.dat");
	strcpy(first_line,"for adiabatic (AD) mode (normalized to initial curvature=1) ");
      }

      if ((ppt->has_bi) && (index_ic == ppt->index_ic_bi)) {

	sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk_bi.dat");
	strcpy(first_line,"for baryon isocurvature (BI) mode (normalized to initial entropy=1)");
      }
	  
      if ((ppt->has_cdi) && (index_ic == ppt->index_ic_cdi)) {
	
	sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk_cdi.dat");
	strcpy(first_line,"for CDM isocurvature (CDI) mode (normalized to initial entropy=1)");
      }
	  
      if ((ppt->has_nid) && (index_ic == ppt->index_ic_nid)) {
	    
	sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk_nid.dat");
	strcpy(first_line,"for neutrino density isocurvature (NID) mode (normalized to initial entropy=1)");
      }
	  
      if ((ppt->has_niv) && (index_ic == ppt->index_ic_niv)) {
	    
	sprintf(file_name,"%s%s%s",pop->root,redshift_suffix,"tk_niv.dat");
	strcpy(first_line,"for neutrino velocity isocurvature (NIV) mode (normalized to initial entropy=1)");
      }

      class_call(output_open_tk_file(pba,
				     psp,
				     pop,
				     &(out_ic[index_ic]),
				     file_name,
				     first_line,
				     pop->z_pk[index_z]
				     ),
		 pop->error_message,
		 pop->error_message);
      
    }
  
    /** - third, compute T_i(k) for each k (if several ic's, compute it for each ic; if z_pk = 0, this is done by directly reading inside the pre-computed table; if not, this is done by interpolating the table at the correct value of tau. */
  
    /* if z_pk = 0, no interpolation needed */

    if (pop->z_pk[index_z] == 0.) {

      for (index_k=0; index_k<psp->ln_k_size; index_k++) {
	for (index_tr=0; index_tr<psp->tr_size; index_tr++) {
	  for (index_ic=0; index_ic<psp->ic_size[index_mode]; index_ic++) {
	    tk[(index_k * psp->ic_size[index_mode] + index_ic) * psp->tr_size + index_tr] = psp->matter_transfer[(((psp->ln_tau_size-1)*psp->ln_k_size + index_k) * psp->ic_size[index_mode] + index_ic) * psp->tr_size + index_tr];
	  }
	}
      }
    }
      
    /* if 0 <= z_pk <= z_max_pk, interpolation needed, */
    else {

      class_call(spectra_tk_at_z(pba,
				 psp,
				 pop->z_pk[index_z],
				 tk),
		 psp->error_message,
		 pop->error_message);
    }

    /** - fourth, write in files */

    if (pop->output_format == camb_format)
      class_alloc(tk_cmbfast,
		  6*sizeof(double),
		  pop->error_message);

    for (index_k=0; index_k<psp->ln_k_size; index_k++) {
      for (index_ic = 0; index_ic < psp->ic_size[index_mode]; index_ic++) {

	if (pop->output_format == class_format) {
	  
	  class_call(output_one_line_of_tk(out_ic[index_ic],
					   exp(psp->ln_k[index_k])/pba->h,
					   &(tk[(index_k * psp->ic_size[index_mode] + index_ic) * psp->tr_size]),
					   psp->tr_size),
		     pop->error_message,
		     pop->error_message);

	}
	else if (pop->output_format == camb_format) {

	  /* rescale and reorder the matter transfer functions following the CMBFAST/CAMB convention */

	  if (pba->has_cdm == _TRUE_)
	    tk_cmbfast[0]=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_cdm]/exp(2.*psp->ln_k[index_k]);
	  else
	    tk_cmbfast[0]= 0.;
	  tk_cmbfast[1]=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_b]/exp(2.*psp->ln_k[index_k]);
	  tk_cmbfast[2]=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_g]/exp(2.*psp->ln_k[index_k]);
	  if (pba->has_ur == _TRUE_)
	    tk_cmbfast[3]=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_ur]/exp(2.*psp->ln_k[index_k]);
	  else
	    tk_cmbfast[3]=0.;
	  if (pba->has_ncdm == _TRUE_) {
	    tk_cmbfast[4]=0.;
	    for (n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++) {
	      tk_cmbfast[4]+=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_ncdm1+n_ncdm]/exp(2.*psp->ln_k[index_k]);
	    }
	  }
	  else
	    tk_cmbfast[4]=0.;
	  tk_cmbfast[5]=-tk[(index_k*psp->ic_size[index_mode]+index_ic)*psp->tr_size+psp->index_tr_tot]/exp(2.*psp->ln_k[index_k]);

	  class_call(output_one_line_of_tk(out_ic[index_ic],
					   exp(psp->ln_k[index_k])/pba->h,
					   tk_cmbfast,
					   6),
		     pop->error_message,
		     pop->error_message);

	}
      }
    }

    /** - fifth, free memory and close files */

    if (pop->output_format == camb_format)
      free(tk_cmbfast);

    free(tk);

    for (index_ic = 0; index_ic < psp->ic_size[index_mode]; index_ic++) {
      fclose(out_ic[index_ic]);
    }
    free(out_ic);
 
  }
  
  return _SUCCESS_;
  
}

/**
 * This routine opens one file where some C_l's will be written, and writes 
 * a heading with some general information concerning its content.
 *
 * @param psp        Input : pointer to spectra structure
 * @param pop        Input : pointer to output structure
 * @param clfile     Output: returned pointer to file pointer
 * @param filename   Input : name of the file
 * @param first_line Input : text describing the content (mode, initial condition..)
 * @param lmax       Input : last multipole in the file (the first one is assmued to be 2)
 * @return the error status
 */

int output_open_cl_file(
			struct spectra * psp,
			struct output * pop,
			FILE * * clfile,
			FileName filename,
			char * first_line,
			int lmax
			) {

  int index_d1,index_d2;

  class_open(*clfile,filename,"w",pop->error_message);

  if (pop->write_header == _TRUE_) {

    if (pop->output_format == class_format) {
      fprintf(*clfile,"# dimensionless %s\n",first_line);
    }
    if (pop->output_format == camb_format) {
      fprintf(*clfile,"# %s (units: [microK]^2)\n",first_line);
    }
 
    fprintf(*clfile,"# for l=2 to %d, i.e. number of multipoles equal to %d\n",lmax,lmax-1);
    fprintf(*clfile,"#\n");

    if (pop->output_format == class_format) {
    fprintf(*clfile,"# -> if you prefer output in CAMB/HealPix/LensPix units/order, set 'format' to 'camb' in input file\n");
    }

    fprintf(*clfile,"# -> if you don't want to see such a header, set 'headers' to 'no' in input file\n");
    fprintf(*clfile,"#\n");
    fprintf(*clfile,"#  l ");
   
    if (pop->output_format == class_format) {
      if (psp->has_tt == _TRUE_)
  fprintf(*clfile,"TT               ");
      // *** MY MODIFICATIONS ***
      if (psp->has_rr == _TRUE_)
  fprintf(*clfile,"RR               ");
      if (psp->has_tr == _TRUE_)
	fprintf(*clfile,"TR                "); 
      // *** END OF MY MODIFICATIONS ***
      if (psp->has_ee == _TRUE_)
	fprintf(*clfile,"EE               ");
      if (psp->has_te == _TRUE_)
	fprintf(*clfile,"TE                "); 
      // *** MY MODIFICATIONS ***
      if (psp->has_tz == _TRUE_)
  fprintf(*clfile,"TZ               ");
      if (psp->has_ez == _TRUE_)
  fprintf(*clfile,"EZ               ");
      // *** END OF MY MODIFICATIONS ***
      if (psp->has_bb == _TRUE_)
	fprintf(*clfile,"BB               ");
      if (psp->has_pp == _TRUE_)
	fprintf(*clfile,"phiphi           ");
      if (psp->has_tp == _TRUE_)
	fprintf(*clfile,"Tphi             ");
      if (psp->has_ep == _TRUE_)
	fprintf(*clfile,"Ephi             ");
      if (psp->has_dd == _TRUE_)
	for (index_d1=0; index_d1<psp->d_size; index_d1++)
	  for (index_d2=index_d1; index_d2<psp->d_size; index_d2++)
	    fprintf(*clfile,"dens[%d]-dens[%d]  ",index_d1+1,index_d2+1);
      if (psp->has_td == _TRUE_)
	for (index_d1=0; index_d1<psp->d_size; index_d1++)
	  fprintf(*clfile,"T-dens[%d]        ",index_d1+1);
      fprintf(*clfile,"\n");
    }

    if (pop->output_format == camb_format) {
      if (psp->has_tt == _TRUE_)
	fprintf(*clfile,"TT               ");
      // *** MY MODIFICATIONS ***
      if (psp->has_rr == _TRUE_)
  fprintf(*clfile,"RR               ");
      if (psp->has_tr == _TRUE_)
	fprintf(*clfile,"TR                "); 
      // *** END OF MY MODIFICATIONS ***
      if (psp->has_ee == _TRUE_)
	fprintf(*clfile,"EE               ");
      if (psp->has_bb == _TRUE_)
	fprintf(*clfile,"BB               ");
      if (psp->has_te == _TRUE_)
	fprintf(*clfile,"TE                "); 
  // *** MY MODIFICATIONS ***
      if (psp->has_tz == _TRUE_)
  fprintf(*clfile,"TZ               ");
      if (psp->has_ez == _TRUE_)
  fprintf(*clfile,"EZ               ");
  // *** END OF MY MODIFICATIONS ***
      if (psp->has_pp == _TRUE_)
	fprintf(*clfile,"dd               ");
      if (psp->has_tp == _TRUE_)
	fprintf(*clfile,"dT               ");
      if (psp->has_ep == _TRUE_)
	fprintf(*clfile,"dE               ");
      if (psp->has_dd == _TRUE_)
	for (index_d1=0; index_d1<psp->d_size; index_d1++)
	  for (index_d2=index_d1; index_d2<psp->d_size; index_d2++)
	    fprintf(*clfile,"dens[%d]-dens[%d]  ",index_d1+1,index_d2+1);
      if (psp->has_td == _TRUE_)
	for (index_d1=0; index_d1<psp->d_size; index_d1++)
	  fprintf(*clfile,"T-dens[%d]        ",index_d1+1);
      fprintf(*clfile,"\n");

    }
  }    

  return _SUCCESS_;

}

/**
 * This routine write one line with l and all C_l's for all types (TT, TE...)
 *
 * @param pba        Input: pointer to background structure (needed for T_cmb)
 * @param psp        Input : pointer to spectra structure
 * @param pop        Input : pointer to output structure
 * @param clfile  Input : file pointer
 * @param l       Input : multipole
 * @param cl      Input : C_l's for all types
 * @param ct_size Input : number of types
 * @return the error status
 */

int output_one_line_of_cl(
			  struct background * pba,
			  struct spectra * psp,
			  struct output * pop,
			  FILE * clfile,
			  double l,
			  double * cl, /* array with argument cl[index_ct] */
        // *** MY MODIFICATIONS ***
        double * dcl,
        double factor,
        // *** END OF MY MODIFICATIONS ***
			  int ct_size
			  ) {

  int index_ct;

  fprintf(clfile,"%4d",(int)l);

  // *** MY MODIFICATIONS ***

  /* Moved the following variable to the argument list. */
  // factor = l*(l+1)/2./_PI_;

  /* Added the logarithmic derivative of the C_l's, which is the same regardless of the required output
  format (no dimension involved). Note that the result will blow for mixed C_l's such as TE, because
  they cross the zero line. */
  if ((psp->compute_cl_derivative == _TRUE_) && (dcl != NULL)) {

    for (index_ct=0; index_ct < ct_size; index_ct++) {
      fprintf(clfile," %16.10e",factor*dcl[index_ct]/(l*cl[index_ct]));
    }
    fprintf(clfile,"\n");	

    return _SUCCESS_;    
  }


  // *** END OF MY MODIFICATIONS ***





  if (pop->output_format == class_format) {

    for (index_ct=0; index_ct < ct_size; index_ct++) {
      fprintf(clfile," %16.10e",factor*cl[index_ct]);
    }
    fprintf(clfile,"\n");	
  }    

  if (pop->output_format == camb_format) {

    if (psp->has_tt == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_tt]);
    // *** MY MODIFICATIONS ***
    if (psp->has_rr == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_rr]);
    if (psp->has_tr == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_tr]);
    // *** END OF MY MODIFICATIONS ***
    if (psp->has_ee == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_ee]);
    if (psp->has_bb == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_bb]);
    if (psp->has_te == _TRUE_)
      fprintf(clfile," %16.10e",factor*pow(pba->T_cmb*1.e6,2)*cl[psp->index_ct_te]);
    // *** MY MODIFICATIONS ***
    if (psp->has_tz == _TRUE_)
      fprintf(clfile," %16.10e",factor*pba->T_cmb*1.e6*cl[psp->index_ct_tz]);
    if (psp->has_ez == _TRUE_)
      fprintf(clfile," %16.10e",factor*pba->T_cmb*1.e6*cl[psp->index_ct_ez]);
    // *** END OF MY MODIFICATIONS ***
    if (psp->has_pp == _TRUE_)
      fprintf(clfile," %16.10e",l*(l+1)*factor*cl[psp->index_ct_pp]);
    if (psp->has_tp == _TRUE_)
      fprintf(clfile," %16.10e",sqrt(l*(l+1))*factor*pba->T_cmb*1.e6*cl[psp->index_ct_tp]);
    if (psp->has_ep == _TRUE_)
      fprintf(clfile," %16.10e",sqrt(l*(l+1))*factor*pba->T_cmb*1.e6*cl[psp->index_ct_ep]);
    if (psp->has_dd == _TRUE_)
      for (index_ct=0; index_ct<psp->d_size*psp->d_size; index_ct++) 
	fprintf(clfile," %16.10e",factor*cl[psp->index_ct_dd+index_ct]);
    if (psp->has_td == _TRUE_)
      for (index_ct=0; index_ct<psp->d_size; index_ct++) 
	fprintf(clfile," %16.10e",factor*cl[psp->index_ct_td+index_ct]);
    fprintf(clfile,"\n");
  }

  return _SUCCESS_;
    
}

/**
 * This routine opens one file where some P(k)'s will be written, and writes 
 * a heading with some general information concerning its content.
 *
 * @param pba        Input: pointer to background structure (needed for h)
 * @param psp        Input : pointer to spectra structure
 * @param pop        Input : pointer to output structure
 * @param tkfile     Output: returned pointer to file pointer
 * @param filename   Input : name of the file
 * @param first_line Input : text describing the content (initial conditions, ...)
 * @param z          Input : redshift of the output
 * @return the error status
 */

int output_open_pk_file(
			struct background * pba,
			struct spectra * psp,
			struct output * pop,
			FILE * * pkfile,
			FileName filename,
			char * first_line,
			double z
			) {

  class_open(*pkfile,filename,"w",pop->error_message);

  if (pop->write_header == _TRUE_) {
    fprintf(*pkfile,"# Matter power spectrum P(k) %sat redshift z=%g\n",first_line,z); 
    fprintf(*pkfile,"# for k=%g to %g h/Mpc,\n",
	    exp(psp->ln_k[0])/pba->h,
	    exp(psp->ln_k[psp->ln_k_size-1])/pba->h);
    fprintf(*pkfile,"# number of wavenumbers equal to %d\n",psp->ln_k_size);
    fprintf(*pkfile,"# k (h/Mpc)  P (Mpc/h)^3:\n");
  }

  return _SUCCESS_;
}

/**
 * This routine writes one line with k and P(k)
 *
 * @param tkfile  Input : file pointer
 * @param one_k   Input : wavenumber
 * @param one_pk  Input : matter power sectrum
 * @return the error status
 */

int output_one_line_of_pk(
			  FILE * pkfile,
			  double one_k,
			  double one_pk
			  ) {

  fprintf(pkfile,"%e %16.10e\n",one_k,one_pk);
    
  return _SUCCESS_;
    
}

/**
 * This routine opens one file where some P_nl(k)'s will be written, and writes 
 * a heading with some general information concerning its content.
 *
 * @param pba        Input: pointer to background structure (needed for h)
 * @param pnl        Input : pointer to nonlinear structure
 * @param pop        Input : pointer to output structure
 * @param tkfile     Output: returned pointer to file pointer
 * @param filename   Input : name of the file
 * @param first_line Input : text describing the content (initial conditions, ...)
 * @param z          Input : redshift of the output
 * @param k_size     Input : number of k values
 * @return the error status
 */

int output_open_pk_nl_file(
			   struct background * pba,
			   struct nonlinear * pnl,
			   struct output * pop,
			   FILE * * pkfile,
			   FileName filename,
			   char * first_line,
			   double z,
			   int k_size
			   ) {

  class_open(*pkfile,filename,"w",pop->error_message);

  if (pop->write_header == _TRUE_) {
    fprintf(*pkfile,"# Non-linear matter power spectrum P_nl(k) %sat redshift z=%g\n",first_line,z); 
    fprintf(*pkfile,"# for k=%g to %g h/Mpc,\n",pnl->k[0]/pba->h,pnl->k[k_size-1]/pba->h);
    fprintf(*pkfile,"# number of wavenumbers equal to %d\n",k_size);
    fprintf(*pkfile,"# k (h/Mpc)  P_nl (Mpc/h)^3:\n");
  }

  return _SUCCESS_;
}

/**
 * This routine opens one file where some T_i(k)'s will be written, and writes 
 * a heading with some general information concerning its content.
 *
 * @param psp        Input : pointer to spectra structure
 * @param pop        Input : pointer to output structure
 * @param tkfile     Output: returned pointer to file pointer
 * @param filename   Input : name of the file
 * @param first_line Input : text describing the content (initial conditions, ...)
 * @param z          Input : redshift of the output
 * @return the error status
 */

int output_open_tk_file(
			struct background * pba,
			struct spectra * psp,
			struct output * pop,
			FILE * * tkfile,
			FileName filename,
			char * first_line,
			double z
			) {

  int n_ncdm;

  class_open(*tkfile,filename,"w",pop->error_message);

  if (pop->write_header == _TRUE_) {

    if (pop->output_format == class_format) {

      fprintf(*tkfile,"# Matter transfer functions T_i(k) %sat redshift z=%g\n",first_line,z); 
      fprintf(*tkfile,"# for k=%g to %g h/Mpc,\n",exp(psp->ln_k[0])/pba->h,exp(psp->ln_k[psp->ln_k_size-1])/pba->h);
      fprintf(*tkfile,"# number of wavenumbers equal to %d\n",psp->ln_k_size);
      fprintf(*tkfile,"# T_i   stands for (delta rho_i/rho_i)(k,z) with above normalization \n");
      fprintf(*tkfile,"# T_tot stands for (delta rho_tot/rho_tot)(k,z) with rho_Lambda NOT included in rho_tot\n");
      fprintf(*tkfile,"# (note that this differs from the transfer function output from CAMB/CMBFAST, which gives the same\n");
      fprintf(*tkfile,"#  quantities divided by -k^2 with k in Mpc^-1; use format=camb to match CAMB)\n");
      fprintf(*tkfile,"#\n");
      fprintf(*tkfile,"# k (h/Mpc)       ");
      fprintf(*tkfile,"T_g                ");
      fprintf(*tkfile,"T_b                ");
      if (pba->has_cdm == _TRUE_)
	fprintf(*tkfile,"T_cdm              ");
      if (pba->has_fld == _TRUE_)
	fprintf(*tkfile,"T_de               ");
      if (pba->has_ur == _TRUE_)
	fprintf(*tkfile,"T_ur               ");
      if (pba->has_ncdm == _TRUE_) {
	for (n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++) {
	  fprintf(*tkfile,"T_ncdm[%d]          ",n_ncdm);
	}
      }
      fprintf(*tkfile,"T_tot\n");
  
    }

    else if (pop->output_format == camb_format) {

      fprintf(*tkfile,"# Rescaled matter transfer functions [-T_i(k)/k^2] %sat redshift z=%g\n",first_line,z);
      fprintf(*tkfile,"# for k=%g to %g h/Mpc,\n",exp(psp->ln_k[0])/pba->h,exp(psp->ln_k[psp->ln_k_size-1])/pba->h);
      fprintf(*tkfile,"# number of wavenumbers equal to %d\n",psp->ln_k_size);
      fprintf(*tkfile,"# T_i   stands for (delta rho_i/rho_i)(k,z) with above normalization \n");
      fprintf(*tkfile,"# The rescaling factor [-1/k^2] with k in 1/Mpc is here to match the CMBFAST/CAMB output convention\n");
      fprintf(*tkfile,"#\n");
      fprintf(*tkfile,"# k (h/Mpc)       ");
      fprintf(*tkfile,"-T_cdm/k2         ");
      fprintf(*tkfile,"-T_b/k2           ");
      fprintf(*tkfile,"-T_g/k2           ");
      fprintf(*tkfile,"-T_ur/k2          ");
      fprintf(*tkfile,"-T_ncdm_tot/k2    ");
      fprintf(*tkfile,"-T_tot/k2         ");
      fprintf(*tkfile,"\n");
	
    }

  }

  return _SUCCESS_;
}

/**
 * This routine writes one line with k and T_i(k)'s
 *
 * @param tkfile  Input : file pointer
 * @param one_k   Input : wavenumber
 * @param tk      Input : vector of transfer functions tk[index_tr] 
 * @param tr_size Input : number of transfer functions
 * @return the error status
 */

int output_one_line_of_tk(
			  FILE * tkfile,
			  double one_k,
			  double * tk,
			  int tr_size
			  ) {

  int index_tr;

  fprintf(tkfile,"%16.10e",one_k);

  for (index_tr=0; index_tr<tr_size; index_tr++)
    fprintf(tkfile,"  %16.10e",tk[index_tr]);
  
  fprintf(tkfile,"\n");
  
  return _SUCCESS_;
  
}
