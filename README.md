# Distributed Implementation for "Natural Gradient Algorithms for Deep Learning"

This repository contains the distributed implementation for my thesis: *"Natural gradient algorithms for training deep neural networks"*. The algorithms presented in the thesis are implemented using PyTorch and can be used as standard optimizers within the PyTorch framework.

[Link to Thesis](https://ora.ox.ac.uk/objects/uuid:0b7ef53b-2192-4332-8641-3b53a7870a98)

## Distributed Optimizers

The distributed optimizers in this repository work as standard PyTorch optimizers. To use them, refer to the main implementation files for usage examples.

- The optimizers are implemented following the PyTorch optimizer API.
- You can plug them into your model training loop just like any other optimizer (e.g., Adam, SGD).
- Check the .py files in the main directory for detailed examples and usage patterns.

## Running Experiments

To run the experiments as described in the thesis or to see how the code can be executed on a SLURM cluster, refer to the .sh script files. These scripts contain the configurations and commands used to launch distributed training on multiple GPUs/nodes.

- The shell scripts show how to configure SLURM jobs for distributed training.
- They also contain detailed experiment setups that align with the experimental results presented in the thesis.

## Caveats
Be careful about running the codes with very many GPUs. While this does not result in any issues, we may get idle GPUs if there's more GPUs than K-factors (may add improvements here later upon request).

## License

GPL-3.0.

## Funding Acknowledgement for this Research
The EPSRC Centre For Doctoral Training in Industrially Focused Mathematical Modelling (EP/L015803/1) in partnership with Numerical Algorithms Group Ltd (NAG) and St Annes provided vital financial support for this research.

## Contact

For questions or further information, feel free to contact me at constantin.puiu@maths.ox.ac.uk.
