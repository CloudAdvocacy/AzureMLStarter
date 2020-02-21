# Submit an experiment programmatically to Azure ML
# This file is intended to be run in interactive mode
# It is essentially equivalent to submit.ipynb notebook,
# for those who prefer plain python files.


from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.estimator import Estimator

# Step 1: Create the workspace (or load from config.json)

try:
    ws = Workspace.from_config()
    print(ws.name, ws.location, ws.resource_group, ws.location, sep='\t')
    print('Library configuration succeeded')
except:
    print('Workspace not found')


# Step 2: Create / Get Reference to Compute Resource

# Choose a name for your CPU cluster
cluster_name = "AzMLCompute"

# Verify that cluster does not exist already
try:
    cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2',
                                                           vm_priority='lowpriority',
                                                           min_nodes=1,
                                                           max_nodes=4)
    cluster = ComputeTarget.create(ws, cluster_name, compute_config)

# Step 3: Upload the dataset to the cluster

ds = ws.get_default_datastore()
ds.upload('./dataset', target_path='mnist_data', overwrite=True, show_progress=True)

# Step 4: Create/Run Experiment

experiment_name = 'Keras-MNIST'
exp = Experiment(workspace=ws, name=experiment_name)
script_params = {
    '--data_folder': ws.get_default_datastore(),
}

est = Estimator(source_directory='.',
                script_params=script_params,
                compute_target=cluster,
                entry_script='mytrain.py',
                pip_packages=['keras','tensorflow']
)

run = exp.submit(est)

# Hyperdrive

from azureml.train.hyperdrive import *

param_sampling = RandomParameterSampling({
         '--hidden': choice([50,100,200,300]),
         '--batch_size': choice([64,128]), 
         '--epochs': choice([5,10,50]),
         '--dropout': choice([0.5,0.8,1])
    })

early_termination_policy = MedianStoppingPolicy(evaluation_interval=1, delay_evaluation=0)
hd_config = HyperDriveConfig(estimator=est,
                            hyperparameter_sampling=param_sampling,
                            policy=early_termination_policy,
                            primary_metric_name='Accuracy',
                            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                            max_total_runs=16,
                            max_concurrent_ru
                            ns=4)
experiment = Experiment(workspace=ws, name='keras-hyperdrive')
hyperdrive_run = experiment.submit(hd_config)

# Register the best model

best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(best_run)
print('Best accuracy: {}'.format(best_run_metrics['Accuracy']))

best_run.register_model(model_name='mnist_keras', model_path='outputs/mnist_model.hdf5')
