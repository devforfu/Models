echo "Running models training"

arch=$1

echo "[1] ${arch} (no regularization)"
python main.py -id="${arch}_no_reg" -arch=${arch} -i=250,250,3 -n=1000 -pt=0.5 -bn 

echo "[2] ${arch} (regularization)"
python main.py -id="${arch}_reg" -arch=${arch} -i=250,250,3 -n=1000 -mn=3 -l2=0.001 -pt=0.5 -bn

echo "[3] ${arch} (regularization, dropout)"
python main.py -id="${arch}_reg_dropout" -arch=${arch} -i=250,250,3 -n=1000 -mn=3 -l2=0.001 -dr=0.5,0.5,0.25,0.25,0.25 -pt=0.5 -bn

echo "Done!"
