# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.q05_display_iter_pert -2
python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.q05_display_iter_pert 6
python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.q05_display_iter_pert 9
python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.q05_display_iter_pert 11
python -m Thesis-InversionLS.IntegralEquation.Scripts-Horizontal-Well.q05_display_iter_pert 13