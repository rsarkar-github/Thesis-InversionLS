# Run from within this directory


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 0 20
python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.p07_perform_update 1 20
