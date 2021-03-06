## Parameters
`num_gpus` The number of GPUs used for training. 

`test` Test mode. \
`train` Train mode. 

`plotting` Plot one image after each training epoch. \
`debug` Debug mode. Should be disabled when training with `num_gpus > 1`. \
`logging` Log loss for each epoch.

`img_size` Size of each image in the dataset and number of channels. \
`training_samples` If more than the training dataset, the entire training dataset is used. \
`testing_samples` Number of testing samples, each image is plotted. \
`max_iters` Number of training epochs. \
`global_batch_size` The total batch size.\
`replica_batch_size` The batch size per GPU, has to correspond with the global batch size.

`dataset_dir` Dataset directory. \
`log_dir` Loss logs directory.\
`checkpoint_dir` Checkpoint directory. \
`checkpoint_prefix` Checkpoint prefix, should be ckpt.

`checkpoint_nsave` Frequencey of checkpoint save.\
`checkpoint_save` Whether to save or not. \
`checkpoint_load` Whether to load or not.

`disc_loss` Discriminiator loss enabled/disabled.\
`l1_loss` L1 loss enabled/disabled. \
`l1_lambda` L1 loss variable.

## Requirements
`pip install -r requirements.txt`
