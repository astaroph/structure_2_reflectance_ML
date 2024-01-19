# structure_2_reflectance_ML
A custom ML regression for making multispectral reflectance prediction algorithms from paired SEM image and multispectral reflectance inputs

I have implemented a ResNet-based Convolutional Neural Network framework adapted for spectral reflectance prediction from structural SEM images in Pytorch

Thus far I have tested this code on a single NVIDIA GPU node with 26-28GB memory in the Harvard RC high performance computing environment (Rocky8 OS). Workflow here assumes the python code is submitted by a job submitted to a high performance compute environment via slurm.

## Submitting jobs
submit using sbatch:
```
sbatch struct2refl_model_running_hyperparam_tuning_saving.sh \
'/PATH/TO/DIR/climate_change_solution_structural_image_reflectancevalues_dataset_updatedstructural_prunedmagnification.csv' \
{BATCHSIZE} {LEARNINGRATE} {EPOCHS} {WEIGHTDECAY} \
'/PATH/TO/SAVEDIR/'
```
