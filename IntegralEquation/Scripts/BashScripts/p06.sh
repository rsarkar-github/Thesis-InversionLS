# Run from within this directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ~
source .bashrc
conda activate py39

cd $SCRIPT_DIR
cd ../../../..

python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 1 0
python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 1 1
python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 1 2
python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 2 0
python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 2 1
python -m Thesis-InversionLS.IntegralEquation.Scripts.p06_plot_images 2 3 2 2
