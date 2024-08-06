## SpeCollate

SpeCollate is the first Deep Learning-based peptide-spectrum similarity network. It allows searching a peptide database by generating embeddings for both mass spectra and database peptides. K-nearest neighbor search is performed on a GPU in the embedding space to find the k (usually k=5) nearest peptide for each spectrum.


### Network Architecture

SpeCollate network consists of two branch, i.e., Spectrum Sub-Network (SSN) and Peptide Sub-Network (PSN). SSN processes spectra and generates spectral embeddings while PSN processes peptide sequences and generates peptides embeddings. Both types of embeddings are generated in real space of dimension 256. The network architecture is shown in Fig 1 below.

![SpeCollate Architecture](https://user-images.githubusercontent.com/6886675/132553654-ccfd96b1-29b4-4506-b3e1-3560d7ef7b2e.png)    
*Fig 1: SpeCollate network architecture. Spectra are encodded in dense arrays of length 80,000 each where each index represents a m/z bin width of 0.1 Da. Hence, spectra with maximum m/z of 8,000 can be encoded using this technique. Encoded spectra are passed through SSN which consists of two fully connected layers of dimessions 80,000 x 1,024 and 1,024 x 256. Output from the second layer is normalized to have unit length. Similarly, peptides sequences are integer encoded where each amino acid and modification character is assigned a unique integer value. These encoded peptide vectors are passed through the embedding layer which learns 256 dimension embedding for each amino acid. The output from the embedding layer is then passed throug PSN which consists of two BiLSTMs and two fully connected layers of length 2,048 x 1,024 and 1,024 x 256. Output from the last layer is normalzied to unit length.*

### SNAP-Loss Function

To train SpeCollate, we design a custom loss function called SNAP-Loss which is inspired from Triplet Loss function. In SNAP-Loss, loss is calcualted over sextuplets of datapoints where each sextuplet consists of an anchor spectrum, a positive peptide, two negative spectra and two negative peptides.

We design SNAP-loss which extends Triplet-Loss to multi-modal data, in our case numerical spectra and sequence peptides. For this purpose, we consider all possible negatives (*q<sub>j</sub>, p<sub>k</sub>, q<sub>l</sub>, p<sub>m</sub>*) for a given positive pair (*q<sub>i</sub>, p<sub>i</sub>*) and average the total loss. The four possible negatives are explained below:
- *q<sub>j</sub>*: The negative spectrum for *q<sub>i</sub>*.
- *p<sub>k</sub>*: The negative peptide for *q<sub>i</sub>*.
- *q<sub>l</sub>*: The negative spectrum for *p<sub>i</sub>*.
- *p<sub>m</sub>*: The negative peptide for *p<sub>i</sub>*.

To calculate the loss value, we first define a few variables that are precomputed in distances matrices above as follows:

![eqs](https://user-images.githubusercontent.com/6886675/132554014-80a4e77a-427d-4bed-94c6-c8633b1433fb.png)

Then the SNAP-loss is calculated for a batch of size b as follows:

![SNAP-Loss](https://user-images.githubusercontent.com/6886675/132554095-5fd14826-da1f-4fde-80db-50ef9e17f337.png)

The training process is visualized in the figure below:

![training](https://user-images.githubusercontent.com/6886675/132570020-ff4ab8b4-7572-4244-8b6f-79dd58b2eec5.png)

Once, the sextuplets are genrated, the loss is calculated using the SNAP-Loss function and the network paramenters are updated by back propagation.

Tuned hyperparameters are given in table 1 below and the ranges for which their value was tuned for:

| Hyperparameter | Value  | Values Tested                                        |
| -------------- | ------ | ---------------------------------------------------- |
| Learning Rate  | 0.0001 | 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01 |
| Weight Decay   | 0.0001 | 1xe^-6, 1xe^-5, 1xe^-4, 1xe^-3                       |
| Margin         | 0.2    | 0.1, 0.2, 0.3, 0.4                                   |
| Embedding Dim  | 256    | 32, 64, 128, 256, 512, 1028, 2048                    |
| FC Layers      | 2      | 1, 2, 3                                              |
| BiLSTM Layers  | 2      | 1, 2, 3, 4                                           |

SpeCollate is available as a standalone executable that can be downloaded and run on a Linux server with a Cuda-enabled GPU.

Two different executables are included in the downloadable specollate.tar.gz file; 1) specollate_train for retraining a model and 2) specollate_search for performing database search using a trained model. A pre-trained model is provided within the download file.

The below sections explain the setup for retraining the model.

### Prerequisites

- A Computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- At least 120GBs of system memory and 10 CPU cores.
- Cuda enabled GPU with at least 12 GBs of memory. Cuda Toolkit 10.0 (or later).
- OpenMS tool for creating custom peptide database. (Optional)
- Crux for FDR analysis using its percolator option.

### Retrain the Model

1. Download the [specollate.tar.gz](https://drive.google.com/file/d/1GkJw8xLl-U-1cTSnOMcKdZhsHd2LWfnb/view?usp=sharing) file and extract the contents using the following command:  
`tar -xzf specollate.tar.gz`  
The extracted directory contains multiple files, including:
    - `specollate-train`: This is the executable for training SpeCollate.
    - `specollate-search`: This is the executable for database search.
    - `config.ini`: Parameter file for training and searching.
    - `models (dir)`: Contains the pre-trained model. New models will also be stored here.
    - `percolator (dir)`: Percolator input (.pin) files be placed here after the search is complete.

2. Download the preprocessed data for training ([here](https://drive.google.com/uc?export=download&id=10bZbMdc2eN_l4ToJd6ruzNX7t6wIUfHw)) and extract the contents using:  
`tar -xzf specollate-training-data.tar.gz`

3. Open the config.ini file from step 1 in your favorite text editor and set the following parameters:
    - `in_tensor_dir` in [preprocess] section: Absolute path of the decompressed file from step 2.
    - `model_name` in [ml] section: The name by which to wish to save the trained model file.
    - other parameters in the [ml] section: You can adjust different hyperparameters in the [ml] section, e.g., learning_rate, dropout, etc.

4. Execute the specollate_train file.  
`./specollate_train`

### Database Search

1. Same as step 1 in the **Previous** section.
2. Download one of the [mgf files](https://drive.google.com/drive/folders/1dvvbYjtz9PrFcMzB-VvtGbrWNX-hl6Io?usp=sharing). Or you can use your own spectra files in mgf format.
3. Download the [human peptide database](https://drive.google.com/uc?export=download&id=1pOBYkCFl66Yk1DjSIw6l9RRi7f6iSXSf). You can provide your own peptide database file created using the Digestor tool provided by [OpenMS](https://www.openms.de/download/openms-binaries/).
4. Set the following parameters in the [search] section of the `config.ini` file:
    - `model_name`: Name of the model to be used. The model should be in the `/models` directory.
    - `mgf_dir`: Absolute path to the directory containing mgf files to be searched.
    - `prep_dir`: Absolute path to the directory where preprocessed mgf files will be saved.
    - `pep_dir`: Absolute path to the directory containing peptide database.
    - `out_pin_dir`: Absolute path to a directory where percolator pin files will be saved. The directory must exist; otherwise, the process will exit with an error.
    - Set database search parameters e.g. `precursor_mass_tolerance` etc.

5. Execute the specollate_search file:  
`./specollate_search`  
If you want to use the preprocessed spectra from a previous run, use the `-p False` flag:  
`./specollate_search -p False`

6. Once the search is complete; you can analyze the percolator files using the crux percolator tool:
```shell
cd <out_pin_dir>
crux percolator target.pin decoy.pin --list-of-files T --overwrite T
```

<p>
 If you use our tool, please cite our work:<br>  
 
 [1]. Tariq, Muhammad Usman, and Fahad Saeed. "SpeCollate: Deep cross-modal similarity network for mass spectrometry data based peptide deductions." PloS one 16.10 (2021): e0259349.
    
 <br>
 For questions, suggestions, or technical problems, contact:<br>
 <a href = "mailto: mtari008@fiu.edu">mtari008@fiu.edu</a>
</p>
