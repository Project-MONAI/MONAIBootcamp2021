# MONAIBootcamp2021
This repository hosts the notebooks for the 2021 MONAI Bootcamp event. The data required for the notebooks is available through the download mechanisms given in each notebook or through the organizers. All bootcamp participants can access the bootcamp Slack channel to ask for help with any issues.

Most of the notebooks in this repository would benefit considerably from having GPU support enabled. Therefore, it is recommended to run notebooks on Google Colab. Instructions to replicate the Python environment on your local machines are also provided (See Install Local Environment)

## Run on Google Colab (Recommended)

Notebooks can be accessed in Colab by using the links below:

**Day 1 Notebooks:**

* <a href="https://colab.research.google.com/github/Project-MONAI/MONAIBootcamp2021/blob/master/day1/1. Getting Started with MONAI.ipynb">1. Getting Started with MONAI.ipynb</a>
* <a href="https://colab.research.google.com/github/Project-MONAI/MONAIBootcamp2021/blob/master/day1/2. MONAI Datasets and Caching.ipynb">2. MONAI Datasets and Caching.ipynb</a>
* <a href="https://colab.research.google.com/github/Project-MONAI/MONAIBootcamp2021/blob/master/day1/3. End-To-End Workflow with MONAI.ipynb">3. End-To-End Workflow with MONAI.ipynb</a>

**Day 3 Notebooks:**

### Further notes

#### 1. Required Packages for Colab Execution

The Day 1 notebooks have the `pip` command for install MONAI, however this will have to be added to any subsequent notebook.
Place this at the top of the first cell to install MONAI the first time a colab notebook is run:

```bash
%pip install -qU "monai[nibabel,ignite,torchvision]==0.6.0"
```

#### 2. Enabling GPU Support

To use GPU resources through Colab remember to change the runtime to GPU:

1. From the "Runtime" menu select "Change Runtime Type"
2. Choose "GPU" from the drop-down menu
3. Click "SAVE"

This will reset the notebook and probably ask you if you are a robot (these instructions assume you are not).
Running

```shell
!nvidia-smi
```

in a cell will verify this has worked and show you what kind of hardware you have access to.

## Instal Local Environment

Instructions to setup the (local) Python development environment are reported below, either using [`venv`](#venv) or [`conda`](#conda) (for Anaconda Python distribution):

<a name="conda"></a>

### Set up environment using ``conda`` 

If you are using Anaconda Python distribution, it is possible to re-create the entire virtual (conda) environment using the `monai_bootcamp.yml` (`YAML`) file:

```shell
conda env create -f monai_bootcamp.yml
```

This will create a new environment named `monai-bootcamp`, with all the required packages.
To **activate** the environment:

```shell
conda activate monai-bootcamp
```

<a name="venv"></a>

### Set up environment using `venv` 

>The `venv` module provides support for creating lightweight “virtual environments” with their own site directories,  optionally isolated from system site directories. Each virtual environment has its own Python binary (which matches the version of the binary that was used to create this environment) and can have its own independent set of installed Python packages in
>its site directories.

**Note**: The `venv` module is part of the **Python Standard Library**, so no further installation is required. **Python 3.7+ is assumed**.

The following **`3`** steps are required to setup a new virtual environment 
using `venv`:

1. Create the environment:

    ```shell
    python -m venv <PATH-TO-VENV-FOLDER>/monai-bootcamp
    ```

    

2. Activate the environment:

  ```shell
  source <PATH-TO-VENV-FOLDER>/monai-bootcamp/bin/activate
  ```

  

3. Install the Required Package (using the `requirements.txt` file):

    ```shell
    pip install -r requirements.txt
    ```
    

#### Notes on Jupyter Notebook Kernel

**Notes:** The following instructions **only** applies to virtual environment created using `venv`

In order to enable the new `venv` environment within your default Jupyter server, a new **Jupyter Kernel** should be added.

In order to do so, the following command should be executed:

```shell
python -m ipykernel install --user --prefix <PATH-TO-VENV-FOLDER>/monai-bootcamp --display-name "Python 3 (MONAI Bootcamp 2020)"
```

This will add a new _Python 3 (MONAI Bootcamp 2020)_ to the list of available Jupyter kernel. Please make sure to **select** or **change** this kernel to run the notebooks in this repository.

Further information [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)

### Notes on GPU support on Local machine

If your local machine has GPU support, please follow the instructions on the official [PyTorch documentation](https://pytorch.org/get-started/locally/) on how to install PyTorch with GPU support in your local environment, depending on your system configuration.
