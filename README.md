# DCGAN with PyTorch
A toy experiment of DCGAN given MNIST Datasets.
## Usage
```bash
git clone 
cd GAN
pip install -r requirements.txt
```
## Model Architecture
```
Discriminator
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 14, 14]             160
              ReLU-2           [-1, 16, 14, 14]               0
            Conv2d-3             [-1, 32, 7, 7]           4,640
              ReLU-4             [-1, 32, 7, 7]               0
            Conv2d-5             [-1, 64, 4, 4]          18,496
              ReLU-6             [-1, 64, 4, 4]               0
            Conv2d-7              [-1, 1, 1, 1]           1,025
           Sigmoid-8              [-1, 1, 1, 1]               0
================================================================
Total params: 24,321
Trainable params: 24,321
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.09
Params size (MB): 0.09
Estimated Total Size (MB): 0.18
----------------------------------------------------------------
Generator
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
   ConvTranspose2d-1             [-1, 32, 7, 7]         100,384
              ReLU-2             [-1, 32, 7, 7]               0
   ConvTranspose2d-3           [-1, 16, 14, 14]           4,624
              ReLU-4           [-1, 16, 14, 14]               0
   ConvTranspose2d-5            [-1, 1, 28, 28]             145
              Tanh-6            [-1, 1, 28, 28]               0
================================================================
Total params: 105,153
Trainable params: 105,153
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.08
Params size (MB): 0.40
Estimated Total Size (MB): 0.49
----------------------------------------------------------------
```
## Quickstart
1.  Created a ```base_model``` directory under the ```experiments``` directory. It contains a file ```params.json``` which sets the hyperparameters for the experiment. It looks like
```Json
{
    "learning_rate": 0.008,
    "batch_size": 128,
    "num_epochs": 10,
    "dropout_rate": 0.0,
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
```
2. Train your experiment. Run
```bash
python train.py --data_dir data/ --model_dir experiments/base_model
```
3. Created a new directory ```learning_rate``` in experiments. Run
```bash
python search_hyperparams.py --data_dir data/ --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in ```search_hyperparams.py``` and create a new directory for each experiment under ```experiments/learning_rate/```.
4. Display the results of the hyperparameters search in a nice format
```bash
python synthesize_results.py --parent_dir experiments/learning_rate
```
5. Evaluation on the test set Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```bash
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
## Resources
* For more Project Structure details, please refer to [Deep Learning Project Structure](https://deeps.site/blog/2019/12/07/dl-project-structure/)