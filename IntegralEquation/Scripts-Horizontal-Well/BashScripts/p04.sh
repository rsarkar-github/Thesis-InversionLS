# Run from within this directory


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.p04_set_true_pert