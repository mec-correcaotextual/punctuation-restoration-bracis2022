pip install -r ../requirements.txt
curl -L https://www.dropbox.com/s/gnxahozkvycfyz7/punct-tedtalk2012.zip?dl=1 > tedtalk2012.zip
unzip tedtalk2012.zip -d ./data/

python train.py --dataset tedtalk2012/
python evaluate.py --bert_model ./outputs/

