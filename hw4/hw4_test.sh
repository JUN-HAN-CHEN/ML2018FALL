wget -O best https://www.dropbox.com/s/4536nkghzw9qovw/best?dl=1
wget -O word2vec.model https://www.dropbox.com/s/rsh5qn1ialkcrsd/word2vec.model?dl=1
wget -O word2vec.model.trainables.syn1neg.npy https://www.dropbox.com/s/ueg259onyh8sfcv/word2vec.model.trainables.syn1neg.npy?dl=1
wget -O word2vec.model.wv.vectors.npy https://www.dropbox.com/s/8plst0ilh7diocv/word2vec.model.wv.vectors.npy?dl=1
python3 hw4_predict.py $1 $2 $3
