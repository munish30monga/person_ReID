conda create -n DINO_env python=3.10
conda activate DINO_env
pip install -r requirements.txt
echo "Installing torhreid..."
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/
pip install -r requirements.txt
python setup.py develop
cd ..