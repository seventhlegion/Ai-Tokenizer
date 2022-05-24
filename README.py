#---------------------- libraries -------------------------
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import pandas as pd
import csv
#---------------------- Dataset ---------------------------
df=pd.read_csv('../soha/src/train.csv')
df_t=pd.read_csv('../soha/src/test.csv')
header = ['id','label']
lab = ([])
index = -1
#--------------------- All features of Train ---------------
doc = [row['query'] for index, row in df.iterrows()]
vec = CountVectorizer()
X = vec.fit_transform(doc)
total_features = len(vec.get_feature_names())

for line in df_t['query']:
    new_word_list = word_tokenize(line)
    index += 1
    
    #print (new_word_list)
    #---------------------- Train -----------------------------

    #---------------------- Amoozesh --------------------------
    Amoozesh = [row['query'] for index, row in df.iterrows() if row['label'] == 1]
    vec_a = CountVectorizer()
    X_a = vec_a.fit_transform(Amoozesh)
    tdm_a = pd.DataFrame(X_a.toarray(), columns=vec_a.get_feature_names())

    word_list_a = vec_a.get_feature_names();    
    count_list_a = X_a.toarray().sum(axis=0) 
    freq_a = dict(zip(word_list_a,count_list_a))

    prob_a = []
    for word, count in zip(word_list_a, count_list_a):
        prob_a.append(count/len(word_list_a))
    dict(zip(word_list_a, prob_a))

    total_cnts_features_a = count_list_a.sum(axis=0)

    prob_a_with_ls = []
    for word in new_word_list:
        if word in freq_a.keys():
            count = freq_a[word]
        else:
            count = 0
        prob_a_with_ls.append((count + 1)/(total_cnts_features_a + total_features))
    dict(zip(new_word_list,prob_a_with_ls))
    total_a = 1
    for i in prob_a_with_ls:
        total_a *= i
    #print (total_a)
    #---------------------- MizEtelaat ------------------------
    Miz = [row['query'] for index, row in df.iterrows() if row['label'] == 2]
    vec_m = CountVectorizer()
    X_m = vec_m.fit_transform(Miz)
    tdm_m = pd.DataFrame(X_m.toarray(), columns=vec_m.get_feature_names())

    word_list_m = vec_m.get_feature_names();    
    count_list_m = X_m.toarray().sum(axis=0) 
    freq_m = dict(zip(word_list_m,count_list_m))

    prob_m = []
    for word, count in zip(word_list_m, count_list_m):
        prob_m.append(count/len(word_list_m))
    dict(zip(word_list_m, prob_m))

    total_cnts_features_m = count_list_m.sum(axis=0)

    prob_m_with_ls = []
    for word in new_word_list:
        if word in freq_m.keys():
            count = freq_m[word]
        else:
            count = 0
        prob_m_with_ls.append((count + 1)/(total_cnts_features_m + total_features))
    dict(zip(new_word_list,prob_m_with_ls))
    total_m = 1
    for i in prob_m_with_ls:
        total_m *= i
    #print (total_m)
    #---------------------- Ketabkhoone -----------------------
    Ketabkhoone = [row['query'] for index, row in df.iterrows() if row['label'] == 3]
    vec_k = CountVectorizer()
    X_k = vec_k.fit_transform(Ketabkhoone)
    tdm_k = pd.DataFrame(X_k.toarray(), columns=vec_k.get_feature_names())

    word_list_k = vec_k.get_feature_names();    
    count_list_k = X_k.toarray().sum(axis=0) 
    freq_k = dict(zip(word_list_k,count_list_k))

    prob_k = []
    for word, count in zip(word_list_k, count_list_k):
        prob_k.append(count/len(word_list_k))
    dict(zip(word_list_k, prob_k))

    total_cnts_features_k = count_list_k.sum(axis=0)

    prob_k_with_ls = []
    for word in new_word_list:
        if word in freq_k.keys():
            count = freq_k[word]
        else:
            count = 0
        prob_k_with_ls.append((count + 1)/(total_cnts_features_k + total_features))
    dict(zip(new_word_list,prob_k_with_ls))
    total_k = 1
    for i in prob_k_with_ls:
        total_k *= i
    #print (total_k)
    #---------------------- Enteghad & Pishnehad --------------
    E_P = [row['query'] for index, row in df.iterrows() if row['label'] == 4]
    vec_e = CountVectorizer()
    X_e = vec_e.fit_transform(E_P)
    tdm_e = pd.DataFrame(X_e.toarray(), columns=vec_e.get_feature_names())

    word_list_e = vec_e.get_feature_names();    
    count_list_e = X_e.toarray().sum(axis=0) 
    freq_e = dict(zip(word_list_e,count_list_e))

    prob_e = []
    for word, count in zip(word_list_e, count_list_e):
        prob_e.append(count/len(word_list_e))
    dict(zip(word_list_e, prob_e))

    total_cnts_features_e = count_list_e.sum(axis=0)

    prob_e_with_ls = []
    for word in new_word_list:
        if word in freq_e.keys():
            count = freq_e[word]
        else:
            count = 0
        prob_e_with_ls.append((count + 1)/(total_cnts_features_e + total_features))
    dict(zip(new_word_list,prob_e_with_ls))
    total_e = 1
    for i in prob_e_with_ls:
        total_e *= i
    #print (total_e)
    #---------------------- Sayer -----------------------------
    Sayer = [row['query'] for index, row in df.iterrows() if row['label'] == 5]
    vec_s = CountVectorizer()
    X_s = vec_s.fit_transform(Sayer)
    tdm_s = pd.DataFrame(X_s.toarray(), columns=vec_s.get_feature_names())

    word_list_s = vec_s.get_feature_names();    
    count_list_s = X_s.toarray().sum(axis=0) 
    freq_s = dict(zip(word_list_s,count_list_s))

    prob_s = []
    for word, count in zip(word_list_s, count_list_s):
        prob_s.append(count/len(word_list_s))
    dict(zip(word_list_s, prob_s))

    total_cnts_features_s = count_list_s.sum(axis=0)

    prob_s_with_ls = []
    for word in new_word_list:
        if word in freq_s.keys():
            count = freq_s[word]
        else:
            count = 0
        prob_s_with_ls.append((count + 1)/(total_cnts_features_s + total_features))
    dict(zip(new_word_list,prob_s_with_ls))
    total_s = 1
    for i in prob_s_with_ls:
        total_s *= i
    #print (total_s)



    max_t = max(total_a, total_m, total_k,total_e,total_s)
    if(max_t==total_e):
        max_t = total_e
        lab.append([index,4])
    elif(max_t==total_k):
        max_t = total_k
        lab.append([index,3])
    elif(max_t==total_m):
        max_t= total_m
        lab.append([index,2])
    elif(max_t==total_s):
        max_t = total_s
        lab.append([index,5])
    elif(max_t==total_a):
        max_t = total_a
        lab.append([index,1])
    #print(lab)
    with open('../soha/src/result.csv', 'w') as res:
        writer = csv.writer(res)
        writer.writerow(header)
        writer.writerows(lab)
