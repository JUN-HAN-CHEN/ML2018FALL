import pandas as pd
import numpy as np
import re
import emoji
import sys
import jieba
from gensim.models.word2vec import Word2Vec
from keras import regularizers
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, GRU, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
jieba.set_dictionary(sys.argv[2]) # 繁體字詞庫
#stopwords = [line.strip() for line in open("./stopword.txt", 'r', encoding='utf-8').readlines()]  
stopword=["ㄉ","呃","我","你","ㄟ"," ", "，", "。", "?", "!", "~", "！", "？", "=", "＝", "～", "「", "」", ",", ".", ">", "<", "/", "\\", "、", "^", "＾", "-","+","＋","B","1","2","3","4","5","6","7","8","9","0",".","...",":","ㄅ","ㄇ","ㄈ","X","D","(",")",":","－","』","『","@","＠","with","face","of","x","d","¬","_","ﾉ"]
#stopword+=stopwords
def preprocess(filepath):#讀檔   刪字   斷詞
    df=pd.read_csv(filepath,sep='\r')
    df=np.array(df)
    dfid=[]
    dfcomment=[]
    for i in range(len(df)):
        dfid.append(re.split(r',',df[i][0])[0])
        dfcomment.append(emoji.demojize(re.split(r',',df[i][0])[1]))
#     for i in range(len(dfcomment)) :
#         for j in range(len(stopword)):
#             #dfcomment[i]=dfcomment[i].replace(stopword[j],"")  
    seg_list=[]
    for i in range(len(dfcomment)):
        seg_list.append(list(jieba.cut(dfcomment[i],cut_all=False)))
    dfdic={"id":dfid,"comment":seg_list}
    dfd={"comment":seg_list}
    #print(len(dfid),len(seg_list))
    df=pd.DataFrame(dfd,columns=["comment"])     
    return df
model =Word2Vec.load("word2vec.model")
pretrained_weights = model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print(vocab_size)
print(emdedding_size)
vocab_list = [(word, model.wv[word]) for word, _ in model.wv.vocab.items()]
print("hi",len(vocab_list))
embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
word2idx = {}
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1
def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)
PADDING_LENGTH=64

model = load_model('best')
X_test=preprocess(sys.argv[1])
X_test = text_to_index(X_test.comment)
X_test=pad_sequences(X_test, maxlen=PADDING_LENGTH)
Y_pred = model.predict(X_test,verbose=1)
th=0.5
for i in range(len(Y_pred)):
    if Y_pred[i]>th:
        Y_pred[i]=1
    else:
        Y_pred[i]=0
Y_pred=Y_pred.astype(int)
#Output
a=[]
for i in range(80000):
    a.append(str(i))
Id = pd.DataFrame(a,columns=["id"])
value = pd.DataFrame({"label":[0]*80000})
result = pd.concat([Id, value], axis=1)
result['label'] = Y_pred
result.to_csv(sys.argv[3], index=False, encoding='big5')
