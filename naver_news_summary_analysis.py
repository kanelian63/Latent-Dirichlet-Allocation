import requests
from bs4 import BeautifulSoup

titles = []
summaries=[]

url = "http://news.search.naver.com/search.naver?ie=utf8&where=news&query=%EC%BD%94%EB%A1%9C%EB%82%98&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&mynews=0&start={}&refresh_start=0"

for n in range(1, 1000):
    raw = requests.get(url.format(1+n*10),
                       headers={'User-Agent':'Mozilla/5.0'})
    html = BeautifulSoup(raw.text, "html.parser")

    articles = html.select("ul.type01 > li")

    for ar in articles:
        title = ar.select_one("a._sp_each_title").text.strip()
        source = ar.select_one("span._sp_each_source").text.strip()
        summary = ar.select('dd')[1].text.strip()
        titles.append(title)
        summaries.append(summary)

import pandas as pd

df_titles = pd.DataFrame(titles)
df_summaries = pd.DataFrame(summaries)

df_titles.to_csv('D:\\AI\\titles_covid.csv', mode='w', encoding='utf-8', index=False)
df_summaries.to_csv('D:\\AI\\summaries_covid.csv', mode='w', encoding='utf-8', index=False)
#%%
import csv
 
f_t = open('D:\\AI\\titles_covid.csv', 'r', encoding='utf-8')
f_s = open('D:\\AI\\summaries_covid.csv', 'r', encoding='utf-8')
rdr_t = csv.reader(f_t)
rdr_s = csv.reader(f_s)

list_titles = []
list_summaries = []

for line_t, line_s in zip(rdr_t, rdr_s):
    list_titles.extend(line_t)
    list_summaries.extend(line_s)

f_t.close()
f_s.close()

#%%
import MeCab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def getNVM_lemma(text):
    tokenizer = MeCab.Tagger()
    parsed = tokenizer.parse(text)
    word_tag = [w for w in parsed.split("\n")]
    pos = []
    tags = ['NNG','NNP','VV','VA','VX','VCP','VCN', 'MAG']
    for word_ in word_tag[:-2]:
        word = word_.split("\t")
        tag = word[1].split(",")
        if(len(word[0]) < 2):
            continue
        if(tag[-1] != '*'):
            t = tag[-1].split('/')
            if(len(t[0])>1 and ('VV' in t[1] or 'VA' in t[1] or 'VX' in t[1])):
                pos.append(t[0])
        else:
            if(tag[0] in tags):
                pos.append(word[0])
    return pos


tf_vect = CountVectorizer(tokenizer=getNVM_lemma, ngram_range=(2, 2), min_df=2, preprocessor=None, lowercase=False)
#tf_vect = CountVectorizer(tokenizer=getNVM_lemma, preprocessor=None, lowercase=False)
dtm_C = tf_vect.fit_transform(list_summaries)
print(tf_vect.get_feature_names())
print(dtm_C)


tfidf_vect = TfidfVectorizer(tokenizer=getNVM_lemma, ngram_range=(2, 2), min_df=2, preprocessor=None, lowercase=False)
#tfidf_vect = TfidfVectorizer(tokenizer=getNVM_lemma, preprocessor=None, lowercase=False)
dtm_T = tfidf_vect.fit_transform(list_summaries)
print(tfidf_vect.get_feature_names())
print(dtm_T)

"""
tf_vect_sum = []
dtm_C_sum = []
tfidf_vect_sum = []
dtm_T_sum = []

for i in range(len(df_summaries)):
    try:
        tf_vect = CountVectorizer(tokenizer=getNVM_lemma, preprocessor=None, lowercase=False)
        dtm_C = tf_vect.fit_transform(df_summaries.loc[i])
        tf_vect_sum.append(tf_vect)
        dtm_C_sum.append(dtm_C)
        #print(tf_vect.get_feature_names())
        #print(dtm_C)

        tfidf_vect = TfidfVectorizer(tokenizer=getNVM_lemma, preprocessor=None, lowercase=False)
        dtm_T = tfidf_vect.fit_transform(df_summaries.loc[i])
        tfidf_vect_sum.append(tf_vect)
        dtm_T_sum.append(dtm_C)
        #print(tfidf_vect.get_feature_names())
        #print(dtm_T)

    except:
        print('Error 발생 Data {}'.format(i))
"""
#%%
vocab = dict()

for idx, word in enumerate(tf_vect.get_feature_names()):
    vocab[word] = dtm_C.getcol(idx).sum()

words = sorted(vocab.items(), key = lambda x:x[1], reverse = True)

import  matplotlib.pyplot   as  plt
import matplotlib.font_manager as fm

path = 'D:\\AI\\LDP\\NEXON Lv1 Gothic Low OTF.otf'
fontprop = fm.FontProperties(fname=path, size=6)
"""
max = 21
plt.bar(range(1,max), [i[1] for i in words[1:max]])
plt.title('Frequency Top 20', fontproperties=fontprop)
plt.xlabel('Words', fontproperties=fontprop)

ax = plt.subplot()
ax.set_xticks(range(1,max))
ax.set_xticklabels([i[0] for i in words[1:max]], rotation = 40, fontproperties=fontprop)
plt.show()
"""
max = 20
plt.bar(range(max), [i[1] for i in words[0:max]])
plt.title('Frequency Top 20', fontproperties=fontprop)
plt.xlabel('Words', fontproperties=fontprop)

ax = plt.subplot()
ax.set_xticks(range(max))
ax.set_xticklabels([i[0] for i in words[0:max]], rotation = 40, fontproperties=fontprop)
plt.show()

#%%
from sklearn.decomposition import LatentDirichletAllocation
tf_vect = TfidfVectorizer(tokenizer=getNVM_lemma,ngram_range=(1, 2), min_df=2, max_df=20000)
dtm = tf_vect.fit_transform(list_summaries)

n_topics = 20

lda = LatentDirichletAllocation(n_components=n_topics)
lda.fit(dtm)

names = tf_vect.get_feature_names()
topics = dict()

for idx, topic in enumerate(lda.components_):
    vocab = []

    for i in  topic.argsort()[:-(30-1):-1]:
        vocab.append((names[i], topic[i].round(2)))
    print("주제 %d:" % (idx +1))
    print([(names[i], topic[i].round(2)) for i in  topic.argsort()[:-(30-1):-1]])

#%%
import joblib

tf_vect = CountVectorizer(tokenizer=getNVM_lemma,ngram_range=(1, 2), min_df=2, max_df=6000, max_features=25000)
dtm = tf_vect.fit_transform(list_summaries)
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, topic_word_prior=0.01, doc_topic_prior=0.001)
lda.fit(dtm)
saved_model = joblib.dump(dtm, 'LDA_IP.pkl')#학습된 모델 저장
names = tf_vect.get_feature_names()
topics_word = dict()
n_words = 20#주제에 포함된 단어 개수
	#주제에 속한 단어 topics_word에 저장
for idx, topic in enumerate(lda.components_):
    vocab = []
    for i in  topic.argsort()[:-(n_words-1):-1]:
        vocab.append((names[i], topic[i].round(2)))
    topics_word[idx+1] = [(names[i], topic[i].round(2)) for i in  topic.argsort()[:-(n_words-1):-1]]
	#주제당 가장 큰 비중을 차지하는 리뷰 출력
max_dict = dict()
for idx, vec in enumerate(lda.transform(dtm)):
    t = vec.argmax()
    if(t not in max_dict):
        max_dict[t] = (vec[t], idx)
    else:
        if(max_dict[t][0] < vec[t]):
            max_dict[t] = (vec[t], idx)

sorted_review = sorted(max_dict.items(), key = lambda x: x[0], reverse=False)

for key, value in sorted_review:
    print('주제 {}: {}'.format(key+1, topics_word[key+1]))
    print('[주제 {}의 대표 리뷰 :{}]\n{}\n\n'.format(key+1, value[0], list_summaries[value[1]]))

#%%
import pyLDAvis.sklearn

visual = pyLDAvis.sklearn.prepare(lda_model=lda, dtm=dtm, vectorizer=tf_vect)
pyLDAvis.save_html(visual, 'D:\\AI\\LDP\\LDA_Visualization_covid.html')
pyLDAvis.display(visual)
