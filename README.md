# Learning-based Intrinsic Reflectional Symmetry Detection
Jittor and matlab code of Learning-based Intrinsic Reflectional Symmetry Detection. Due to the size limit of supplementary materials, only part of our training data is uploaded.

## Usage

  1. Clone this repo.
   ```
   git clone https://github.com/IGLICT/intrinsicSym-Jittor.git
   ```

  2. Set up python3 virtual environment.
   ```
   virtualenv --system-site-packages -p python3 ./env-intrinsym
   source ./env-intrinsym/bin/activate
   cd ./instrinsym
   pip install -r requirements.txt
   ```

  3. Install Jittor, you can refer to the [official repo of Jittor](https://github.com/Jittor/jittor).


  4. Use the network to predict intrinsic symmetry.
   ```
   cd ./network
   python predict.py
   ```
   or train this network
   ```
   python train.py
   ```

  5. Run the scripts (./intrinsym/scripts) for visualization, annotation, or preprocessing with matlab
   ```
   main_visualize_ss   % visualize
   main_selecct_eigen  % annotate
   main_compute_evecs  % preprocess
