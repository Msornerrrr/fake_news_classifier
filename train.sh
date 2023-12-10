python lstm.py train -s &&
python lstm.py train -dp 0.1 -s &&
python lstm.py train -dp 0.5 -s &&
python lstm.py train -pe -s &&
python lstm.py train -hd 128 -s &&
python lstm.py train -hd 512 -s &&
python lstm.py train -d 1 -s