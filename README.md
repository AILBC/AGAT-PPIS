# AGAT-PPIS
## 1 Description
    (a) AGAT-PPIS is a protein-protein interaction site predictor based on deep graph neural networks, which treats protein-protein interaction site prediction as a classification task of graph nodes.
    (b) Two kinds of node features and two kinds of edge features are introduced to improve the performance of the model.
    (c) The base model of AGAT-PPIS is AGAT, which is a variant of GAT, with adding edge features into the process of GAT calculating the attention scores and updating the node embeddings.
    (d) The combination of the initial residual and identity mapping and AGAT makes the model to learn the high-order embedding representations of resides more sufficiently, and so the model performance surpasses others significantly.
## 2 Installation
### 2.1 system requirements
  For fast prediction and training process, we recommend using a GPU. To use AGAT-PPIS with GPUs, you will need: cuda = 10.1
### 2.2 virtual environment requirements
    (1) python 3.6
    (2) torch-1.7.0+cu101
    (3) torchaudio-0.7.0
    (4) torchvision-0.8.0
    (5) dgl_cu101.0.7.0
    (6) cudatoolkit-10.1.168
    (7) pandas
    (8) sklearn
## 3 Datasets
  The files in "./Dataset" include the datasets used in this experiment(Test_315-28.pkl, Test_60.pkl, Train_335.pkl, UBtest_31-6.pkl, bound_unbound_mapping31-6.txt), and the rest of the datasets are the original datasets from GraphPPIS.<br>
  All the processed pdb files of the protein chains used in this experiment are put in the directory "./Dataset/pdb/".
## 4 Features
    The extracted features are in the directory "./Feature". The specific meanings are listed as follows.
        (1) distance_map_C: using the centroid of residues as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (2) distance_map_CA: using the position of the alpha-C atom of residues as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (3) distance_map_SC: using the centroid of the residue side chain as the pseudo position of resiudes, and using them to calculate the distance matrix of the protein chain.
        (4) dssp: the DSSP matrix of the protein chains used in this experiment.
        (5) hmm: the HMM matrix of the protein chains used in this experiment.
        (6) psepos: the resiude pseudo positions of the protein chains in those datasets, with SC, CA, C standing for centriod of side chain, alpha-C atom and centroid of the residue, respectively.
        (7) pssm: the PSSM matrix of the protein chains used in this experiment.
        (8) resAF: the atom features of the residues for each protein used in the experiment.
## 5 The trained model
  The models with trained parameters are put in the directory "./Model/2022-09-06-11-49-17/model/" and the predicted results of the test datasets are put in the directory "./Model/2022-09-06-11-49-17/result_metrics".
## 6 Usage
  The construction of the model is in the "AGATPPIS_model.py".<br>
  You can run "train.py" to train the deep model from stratch and use the "test.py" to test the test datasets with the trained model.
## 7 Access for the paper of AGAT-PPIS
  Paper title: "AGAT-PPIS: A novel Protein-Protein Interaction Site predictor based on Augmented Graph Attention Network with initial residual and identity mapping". <br>
  Paper link: https://doi.org/10.1093/bib/bbad122
  
