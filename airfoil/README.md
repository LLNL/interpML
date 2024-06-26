# Case Study with UIUC Airfoil Data

This directory contains scripts, notebooks, and data files for reproducing
our experiments with the UIUC airfoil dataset.

Before using, make sure that you have followed the set-up instructions in
``../interpml`` and pip installed the parent directory.

The directory structure is as follows:
 - ``./data`` contains the UIUC airfoil dataset and our embeddings thereof.
   We have use a train/valid/test split, where convolutional autoencoders
   were trained on the training data to produce 2, 4, 6, and 8-dimensional
   latent feature spaces. In the ``data`` directory:
    - ``{Train|Val|Test}_labels.npy`` contains the labels (response values)
      for the training, validation, and test splits.
    - The directories ``Dim_{2|4|6|8}`` contain the 2, 4, 6, and 8-dimensional
      latent space embedding of the training and validation sets.
    - In Section 4.3, we perform attitional interpretation with the
      4-dimensional latent space. The following additional files are contained
      in the ``Dim_4`` subdirectory:
       - The pooled and averaged training labels are contained in
         ``Train_labels_reduced.npy``.
       - ``decoder_model.h5`` contains the decoder model for the 4-dimensional
         latent space.
       - ``Dim_4`` also contains the 4-dimensional latent space embedding of
         the test data since this is the embedding upon which we performed
         further testing.
 - ``run_scripts`` contains the Python scripts used to generate results data
   for the paper.
 - ``final_results`` contains the results generated by the scripts in
   ``run_scripts``.
 - ``notebooks`` contains the jupyter notebooks used to analyze and interpret
   results.
