import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import re
import emoji
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
import sys
jieba.set_dictionary(sys.argv[4]) # 繁體字詞庫

stopwords = [line.strip() for line in open("./stopword.txt", 'r', encoding='utf-8').readlines()]  
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
def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df
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
X_train=preprocess(sys.argv[1])
X_test=preprocess(sys.argv[3])
Y_train=pd.read_csv(sys.argv[2])
Y_train=Y_train.label
#Y_train = to_categorical(Y_train)

corpus=pd.concat([X_train.comment,X_test.comment,X_test.comment,X_test.comment])
#corpus = np.vstack([X_train, X_test, X_test])
word_model = Word2Vec(corpus,size=256)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print(vocab_size)
print(emdedding_size)
word_model.save('word2vec.model')
word_model = Word2Vec.load('word2vec.model')

X_val=X_train[-10000:].comment
Y_val=Y_train[-10000:]
X_train=X_train[:-10000].comment
Y_train=Y_train[:-10000]

embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
word2idx={}
vocab_list = [(word, word_model.wv[word]) for word , _ in word_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1
embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)
PADDING_LENGTH = 64
X_train = text_to_index(X_train)    
X_train = pad_sequences(X_train, maxlen=PADDING_LENGTH)    
X_val = text_to_index(X_val)
X_val = pad_sequences(X_val, maxlen=PADDING_LENGTH)
print("Shape:", X_train.shape)
print("Sample:", X_train[10])

model = Sequential()
model.add(embedding_layer)
model.add(GRU(units=128,activation="tanh",dropout=0.3,recurrent_dropout=0.3,return_sequences=True))
model.add(GRU(units=128,activation="tanh",dropout=0.3,recurrent_dropout=0.3,return_sequences=True))
model.add(GRU(units=128,activation="tanh",dropout=0.3,recurrent_dropout=0.3,return_sequences=False))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)

#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["accuracy"])
model.summary()

# Setting callback functions
csv_logger = CSVLogger('training.log')
checkpoint = ModelCheckpoint(filepath='best',
                             verbose=1,
                             save_best_only=True,
                             monitor='val_acc',
                             mode='max')
earlystopping = EarlyStopping(monitor='val_acc', 
                              patience=6, 
                              verbose=1, 
                              mode='max')
                             
# Train the model
train_history = model.fit(X_train, Y_train, 
          validation_data=(X_val, Y_val),
          epochs=100,
          batch_size=500,
          callbacks=[earlystopping, checkpoint, csv_logger])
