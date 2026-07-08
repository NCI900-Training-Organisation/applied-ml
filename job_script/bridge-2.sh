projects

echo $HOME
echo $PROJECT 
echo $LOCAL

ls /ocean/projects/tra210016p/$USER

squeue -u $USER

module load python/3.8.6
module load cuda/12.6.1

python3 -m venv pytorch-venv
source pytorch-venv/bin/activate

python3 -m pip install --upgrade pip

rsync -rltpDvp -e 'ssh -l jjohn2' ./chest_xray   data.bridges2.psc.edu:/ocean/projects/tra210016p/jjohn2/