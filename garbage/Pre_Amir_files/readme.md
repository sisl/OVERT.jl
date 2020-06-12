# Overview
This respository contains all the code ever written for the darpa aahaa project. It's messy. 

## Running Overt.jl alone

The newest stuff is in OverApprox/OverApprox/src/
Use the project.toml file in OverApprox/OverApprox/src/ to set up your julia environment. Examples of using overt.jl to overapproximate functions can be found in unittest_overapprox_nd_relational.jl. The running overapprox_nd(...) produces an object of type OverApproximation which is defined in OA_relational_util.jl

## Running the Model Checker

The model checking code can be found in MarabouMC. To run this code first create a conda environment from the ova_env.yml file. 
See: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
if you're not familiar with conda. (Chelsea recommends using miniconda3)

You'll also want to clone Marabou from https://github.com/NeuralNetworkVerification/Marabou . Follow their instructions regarding cmake and make to compile. 

There are several constantly evolving test scripts that test the model checker. The first is test_TFc_pwld.py which tests a tensorflow controller and simple piecewise linear dynamics (not from OVERT). 

The scripts we are currently debugging which call both OVERT and tensorflow/keras are test_Kerasc_pwld.py and test_Keras_pwld2.py