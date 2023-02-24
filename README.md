# stochastic-rewards

## installation guide

## rust install
Follow the [rust install guide link](https://www.rust-lang.org/tools/install) .

### Libtorch Manual Install
Get libtorch from the PyTorch website download section and extract the content of the zip file.
For Linux users, add the following to your .bashrc or equivalent, where /path/to/libtorch is the path to the directory that was created when unzipping the file.
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```
The header files location can also be specified separately from the shared library via the following:

#### LIBTORCH_INCLUDE must contains `include` directory.
`export LIBTORCH_INCLUDE=/path/to/libtorch/`
#### LIBTORCH_LIB must contains `lib` directory.
`export LIBTORCH_LIB=/path/to/libtorch/`

For Windows users, assuming that `X:\path\to\libtorch` is the unzipped libtorch directory.

Navigate to Control Panel -> View advanced system settings -> Environment variables.
Create the LIBTORCH variable and set it to `X:\path\to\libtorch`.
Append X:\path\to\libtorch\lib to the Path variable.
