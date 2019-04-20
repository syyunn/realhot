# realhot

## Goal of the Project
The project goal is about the way to determine the `optimal number of latent 
dimension`. 

First, the project introduces the linearity and non-linearity and postulates 
the assumption that linearity corresponds to `one` dimension. Then, this 
linearity could be split into `two` non-overlapping dimension by one ReLU based non-linearity. 

Therefore, this project shows that the determination of optimal number of latent dimension
 preliminarily `not depend on the data distribution itself`, but depends on `the network structure`, 
 more specifically, depends on the `total number of dimension that the model 
about to express`. The paper will call this total number of dimension that the 
model about to express as **model dimension**.

After the model dimension being set, one can train the network and check whether 
it's possible to over-fit the network with the data given. If the data points 
over-fit in some point of train epochs, this network can be thought as "enough to 
express the data distribution". However, if not over-fit, one can consider to 
enlarge the **model dimension** and re-try the over-fit process.

## To-do
Define the over-fit.
The classification threshold of over-fit depends on the experiment. 
- In which epoch of training process one should determine over-fit? 

## Caution
It's better to use whole data when to determine the "model dimension" since 
it's about how much non-linearity is required for the collected or targeted 
data domain.


## Experiment Workflow

##### Exp_1 : 1 ReLU applied to 256 dimension. (Then Linear Transformation to LatentDim)

By the assumption, the **model dimension** is 512(256*2). Thus, we verify the assumption by

1) check the sequential decrease of Loss at certain train epoch while sequentially increase the LatentDim

with `1 * (MLP + ReLU) + LatentDim 1` 

    Epoch 09/10 Batch 0937/937, Loss  165.5437
 
with `1 * (MLP + ReLU) + LatentDim 2` 

    Epoch 09/10 Batch 0937/937, Loss  150.2990

with `1 * (MLP + ReLU) + LatentDim 3` 

    Epoch 09/10 Batch 0937/937, Loss  133.2206
    

> ... must keep decreasing. write the code to automatically does this job 
    
with `1 * (MLP + ReLU) + LatentDim 512` 

    Epoch 09/10 Batch 0937/937, Loss   53.2412
    
> ... Check whether at any LatentDim > 512, no decrease of Loss at fixed train epoch. 

    
with `1 * (MLP + ReLU) + LatentDim 1024` 

    Epoch 09/10 Batch 0937/937, Loss   54.3255

> As you see, with the expansion of LatentDim `doubled`, still the LossAtFixedStep is not decreased, 
which means model dimension already being saturated. 
#### Exp_2: Now Introduce the Twice more model dimension by ReLU 

with `2 * (MLP + ReLU) + LatentDim 1024`

    
