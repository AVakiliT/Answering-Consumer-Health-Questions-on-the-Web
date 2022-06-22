#SBATCH --time=0-12:00:00
#SBATCH --account=rrg-smucker
#SBATCH --cpus-per-task=4
#SABTCH --ntasks=1
#SBATCH mem-per-cpu=8GB

ipython --ipython-dir=/tmp squadstuff/multico.py
