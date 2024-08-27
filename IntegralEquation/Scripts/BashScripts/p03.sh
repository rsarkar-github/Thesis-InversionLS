# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 0 0
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 0 2
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 0 3
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 1 0
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 1 2
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 1 3
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 2 0
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 2 2
python -m Thesis-InversionLS.IntegralEquation.Scripts.p03_compute_true_data 2 3