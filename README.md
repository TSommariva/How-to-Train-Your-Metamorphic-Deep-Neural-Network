# How to Train Your Metamorphic Deep Neural Network

Neural Metamorphosis (NeuMeta) is a recent paradigm for generating neural networks of varying width and depth. Based on Implicit Neural Representation (INR), NeuMeta learns a continuous weight manifold, enabling the direct generation of compressed models, including those with configurations not seen during training. While promising, the original formulation of NeuMeta proves effective only for the final layers of the undelying model, limiting its broader applicability. In this work, we propose a training algorithm that extends the capabilities of NeuMeta to enable full-network metamorphosis with minimal accuracy degradation. Our approach follows a structured recipe comprising block-wise incremental training, INR initialization, and strategies for replacing batch normalization. The resulting metamorphic networks maintain competitive accuracy across a wide range of compression ratios, offering a scalable solution for adaptable and efficient deployment of deep models.


**How to Train Your Metamorphic Deep Neural Network**

 📝[Paper]()

Thomas Sommariva, Simone Calderara, Angelo Porrello

AImageLab, University of Modena and Reggio Emilia, Italy

## 🏗️ Code Structure

```shell
neumeta/
│
├── config/        # Configuration files for experimental setups
├── models/        # Definitions and variations of NeuMeta models
├── prune/         # Scripts for model pruning and optimization
├── similarity/    # Tools for evaluating model weight similarities
├── utils/         # General utility scripts
│
├── hypermodel.py   # The INR Hypernetwork for NeuMeta
├── smoothing.py/     # Enforces smooth weight transitions across models
└── environment.yml   # Conda Environment

```

## 🚀 Getting Started
To run the NeuMeta project:

1. **Clone the repository**.
2. **Install the dependencies**: `conda env create -f environment.yml`.
3. **Prepare the preatrined checkpoint**: Ensure you have a pretrained model checkpoint for initialization. This will act as the base for Neural Metamorphosis.

4. **Convert to smooth weight**: Use the weight permutation algorithm in `smooth/permute.py` to transform the checkpoint into a smoother weight version for effective morphing.
```python
from smooth.permute import PermutationManager

# Create the model for CIFAR10
model = create_model_cifar10(model_name)
model.eval()  # Set to evaluation mode

# Compute the total variation loss for the network
total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
print("Total Total Variation After Training:", total_tv)

# Apply permutations to the model's layers and check the total variation
input_tensor = torch.randn(1, 3, 32, 32).to(device)
permute_func = PermutationManager(model, input_tensor)
# Compute the permutation matrix for each clique graph, save as a dict
permute_dict = permute_func.compute_permute_dict()
# Apply permutation to the weight
model = permute_func.apply_permutations(permute_dict, ignored_keys=[])
total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
print("Total Total Variation After Permute:", total_tv)
```
  
5. **Train the INR on the checkpoint and dataset**
   - For example, if we want to run experiments on CIFAR10 with resnet20:
  ```shell
  PYTHONOPATH="$PWD" python neumeta/train_cifar10.py --config <CONFIG_PATH>
  ```

Replace `<CONFIG_PATH>` with the path to your specific configuration file tailored for the dataset and model architecture you intend to train.

6. **Weight Sampling for Target Model**
After training the INR, sample weights for any architecture in the same family. For example:
```python
args = parse_args()
#### Load INR model ####
checkpoint_info, hyper_model, _, _, _ = load_checkpoint("test_path", hyper_model, None,None ,None, device=device)
backbone_parameters = checkpoint_info['backbone_parameters']

for hidden_dim in range(16, 65):
    # Create a model for the given hidden dimension
    model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim,
                                 num_param=args.model.num_param,
                                 bottom_up=args.model.bottom_up,
                                 single_block=False,
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth, fuse=args.model.fuse,
                                 prior=False, config_args=args, first_meta_layer=args.model.start_layer, num_layers=3)
        
    # Sample the merged model for K times
    accumulated_model = sample_merge_model(best_hyper_model, model, args, backbone_parameters=backbone_parameters ,K=100, device=device)
    # Validate the merged model
    val_loss, val_acc = validate_single(accumulated_model, val_loader, criterion, args, device=device)

    # Print the results
    print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")        
```

## 🙏 Acknowledgments
This project is adapted from [NeuMeta](https://github.com/Adamdad/neumeta). We extend our gratitude to the original authors for their foundational work.