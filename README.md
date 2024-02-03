# Neural Ordinary Differential Equations
This is a repository of code developed within the framework of a Bachelor's Thesis at the University of Barcelona on Neural Ordinary Differential Equations. It serves as a final degree project for the joint degrees of Mathematics and Computer Science.

As part of the project, this code provides examples and illustrations of how neural ODEs can be used with various purposes. Additionally, it intends to be a bridge between the theoretical results presented in the main text and the practical applications of this kind of model.

## Folder structure
- `support` contains additional resources used in the creation of the project's main document `memoria.pdf`
- `experiments` contains the code for all the demonstrations and proofs-of-concept used in the project
    - `helpers` is an internal library for training and visualisation
    - `continuous_normalising_flows` contains the experiments about CNF models.
        - `circles_results`, and `triangle_results` have images generated when experimenting with different distributions
        - `results` contains the trained models
        - `cnf_one_images` and `cnf_one_images_2` have `gif` illustrations of the evolution of a one-dimensional CNF
        - `modelling` contains auxiliary classes used in `cnf_mnist.ipynb` to generate hand-written digits
    - `neural_odes` contains examples of simple neural ODEs architectures
        - `adjoint_comparison` compares the efficiency of training neural ODEs using discretise-then-optimise or optimise-then-discretise approaches
        - `augmentation` shows the difference between augmented and unaugmented models
        - `linear_ode` illustrates how neural ODEs can be used to learn a linear continuous dynamical system
