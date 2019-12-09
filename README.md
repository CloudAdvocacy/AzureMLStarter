# AzureMLStarter

## Materials for talks / blog posts on Azure ML Service

Here is the recommended path to success with Azure ML Service:

### Stage 1

1. Install [Visual Studio Code](http://code.visualstudio.com) and [Azure ML Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai#overview)
2. [Create Azure ML Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-1st-experiment-sdk-setup#create-a-workspace/?WT.mc_id=ca-github-dmitryso
) through [Azure Portal](http://portal.azure.com/?WT.mc_id=ca-github-dmitryso)
3. Open this directory in VS Code: `code .`
4. Run `train_local.py` (either completely, or in Python interactive console line by line) -- it is a simple script that trains MNIST model using Scikit Learn locally
5. Make sure your Azure ML Extension is connected to your cloud account and you can see your workspace.
6. Observe `train_universal.py` script --- it is a training script that can be run both locally and submitted to Azure ML for training.
7. Submit `train_universal.py` to Azure ML using VS Code interface and observe results in [Azure ML Portal](http://ml.azure.com/?WT.mc_id=ca-github-dmitryso).

You now know that submitting runs to Azure ML is not complicated, and you get some goodies (like storing all statistics from your runs, models, etc.) for free.

### Stage 2

Now let's learn how to submit scripts programmatically:

1. Create small MNIST dataset for our experiments by running `create_dataset.ipynb`. It will create `dataset` subdirectory.
2. Open `submit.ipynb` file in Jupyter Notebook. You can either:
    - Start a local jupyter notebook in the current directory: `jupyter notebook`
    - Upload `submit.ipynb` and `datasets` folder to [Azure Notebooks](http://aka.ms/aznb)
    - Create a notebook in your Azure ML Workspace (in this case you would also have to create a VM to run it on).
3. Download `config.json` file from your Azure Portal, which contains all credentials for accessing the Workspace, and place it in the current directory, or where your Jupyter notebook is.
4. Execute `submit.py` to submit the simple experiment
5. Go on to perform Hyperparameter optimization

Have fun!

-- [Dmitry Soshnikov](http://soshnikov.com)