#SBATCH --account=rrg-smucker
#SBATCH --time=0-03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:v1001:2
#SBATCH --output=slurm/%A_%a.out

module load gcc/9.3.0 arrow scipy-stack
source ~/avakilit/PYTORCH/bin/activate

python data/process.py

