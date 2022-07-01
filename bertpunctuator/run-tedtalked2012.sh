pip install -r requirements.txt

python preprocess.py
python train.py --dataset tedtalk2012/
python evaluate.py --bert_model ./outputs/

