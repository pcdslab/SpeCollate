## Welcome to the Official Site of SpeCollate

SpeCollate is the first Deep Learning based peptide database search tool. It allows you to search a peptide database by generating embeddings for both mass spectra and peptides and then performs the search process on GPU.

SpeCollate is available as a standalone executable that can be downloaded and run on a linux server with a Cuda enabled GPU.

Two different executables are included in the downloadable specollate.tar.gz file; 1) specollate_train for retraining a model and 2) specollate_search for perform database search using a trained model. A pretrained model is provided within the download file.

Below sections separately explain setup for training and search operation. You can skip the training section if you only intend to use SpeCollate for database search.

## Prerequisites

- Linux server with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- At least 120GBs of system memory and 10 CPU cores.
- Cuda enabled GPU with at least 12 GBs of memory. Cuda Toolkit 10.0 (or later).

## Training

1. Download the [specollate.tar.gz](link) file and extract the contents using the following command:  
`tar -xzf specollate.tar.gz`  
The extracted directory contains multiple files including:
    - `specollate-train`: This is the executable for training SpeCollate.
    - `specollate-search`: This is the executable for database.
    - `config.ini`: Parameter file for training and searching.
    - `models (dir)`: Contains the pretrained model. New models will also be stored here.
    - `percolator (dir)`: Percolator input (.pin) files be placed here after the search is complete.

2. Download the preprocessed data for training ([specollate-training-data.tar.gz](link)) and extract the contents using:  
`tar -xzf specollate-training-data.tar.gz`

3. Open the config.ini file from step 1 in your favorite text editor and set the following parameters:
    - `in_tensor_dir` in [preprocess] section: Absolute path of the decompressed file from step 2.
    - `model_name` in [ml] section: The name by which to wish to save the trained model file.
    - other parameters in [ml] section: You can adjust different hyperparameters in the [ml] section e.g. learning-rate, dropout etc.

4. Execute the specollate_train file.  
`./specollate_search`

## Search



You can use the [editor on GitHub](https://github.com/Usman095/SpeCollate/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Usman095/SpeCollate/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
