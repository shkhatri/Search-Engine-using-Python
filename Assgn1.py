import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
import time

#calculate idf
def getidf(token):
     
    if token in df:
        val = math.log10(30/df[token])
    else:
        val =  -1.0   
       
    return(float(val))    

    
#Normalize the vector    
def normalization(weighttf):
    tfidf = {}
    length = 0
    weight = {}
    for words in weighttf:
      tfidf[words] = weighttf[words] * getidf(words)
      length = length +(tfidf[words])**2

    length = math.sqrt(length)
  
    for words in tfidf:
        weight[words] = float(tfidf[words]/length)
        
    return(weight)

    
#Calculate weight    
def getweight(filename,token):
    weighttf = {} 
    val =  0.0  
    weighttf = listOfList.get(filename)
    global weight
    if weighttf:
        weight = normalization(weighttf)
    
        if token in weight:
                val = weight.get(token)
            
    return(float(val))
    

#Get the Top 10 postings  
def topTenList (token):
    topTenwt ={}
    topTenList = []
    topTen = {}
    for filename in listOfList:
        
        topTenwt[filename]=getweight(filename,token)
    
    topTenList = sorted(topTenwt.items(), key=lambda x:-x[1])[:10]
      
    for docs in topTenList:
        topTen[docs[0]]=docs[1] 
    return topTen   
    
    
#Calculate the cosine similarity  
def query(qstring):
    s = time.time()
    sQuery=[]
    q_tf = {}
    q_weight = {}
    d_weight = {}
    length = 0
    sQuery = preprocessing(qstring)  
    q_tf = calculate_tf(sQuery)
    score = 0.0
    
    #normalize query vector
    for words in q_tf:
      length = length +(q_tf[words])**2
    length = math.sqrt(length)
    for words in q_tf:
      q_weight[words] = float(q_tf[words]/length)    
    
    #get the top 10 postings for each token in query  
    for words in sQuery:
       d_weight[words] = topTenList(words) 
    
    #get the intersection of docs in top 10 postings  
    inter = set(list(d_weight.values())[0])
    unionSet = set(list(d_weight.values())[0])
    Totalweight = {}
    highScore = 0
    for docs in d_weight:
        inter = inter.intersection(set(d_weight[docs]))
        unionSet = unionSet.union(set(d_weight[docs]))
        
    #calculate the cosine similarity
    if inter:
         for files in inter: 
             weight =0
             for words in q_weight:
               weight=weight+(q_weight[words]*d_weight[words].get(files)) 
             unionSet.remove(files) 
             Totalweight[files,"AS"]=weight  
         
         # calculate the cosine similarity by upper bound of doc doesn't appear in the top-10 elements of some query token   
         for files in unionSet: 
             weight =0
             for words in q_weight:  
               if d_weight[words].get(files): 
                   weight=weight+(q_weight[words]*d_weight[words].get(files))
               else:
                   fileDict = {}
                   fileDict = d_weight[words]
                   upperBound = min(fileDict, key=lambda key: fileDict[key])
                   weight=weight+(q_weight[words]*fileDict[upperBound])
             Totalweight[files,"UB"]=weight


         #get max value from the calculated cosine similarities
         highScore = max(Totalweight, key=lambda key: Totalweight[key])
         
            
         if highScore[1] is 'AS': 

             #if the max value is 0.0 means query tokens are not in any doc
             if Totalweight[highScore] == float(0): 
                doc = "None"

            #if the highest score for cosine similarity is calculated by Actual score   
             else:
                 score = Totalweight[highScore]
                 doc = highScore[0]

         #if the highest score for cosine similarity is calculated by upper bound
         else:
             doc = "fetch more"
    
    #No doc is common in top 10 postings of query vector     
    else:
         doc = "fetch more"
    print ("Took %f seconds" % (time.time() - s) ) 
    return (doc,float(score))
 
    
#Calculate document frequency 
def calculate_df():
       
    for words in tf:
        if words in df.keys():
            df[words] += 1
        else:
            df[words] = 1 
    return df     
    

#Calculate term frequency    
def calculate_tf(documents):
    #Calculate tf
    global tf
    tf = {}
    global wttf
    wttf = {}
    for words in documents:
        if words in tf.keys():
            tf[words] += 1
        else:
            tf[words] = 1
      

    #Calculate tf = 1 + log(tf)            
    for words in tf:
        wttf[words] = 1 + math.log10(tf[words])
    
    return(wttf)
    
  
    
def preprocessing(doc):
    
    #lower case
    doc = doc.lower()
    
    #Tokenize
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    
    #stopword removal
    stopwordsList = []
    cachedStopWords = stopwords.words("english")
    for word in tokens:
        if word not in cachedStopWords:
         stopwordsList.append(word)
         
    #stemming      
    stemmer = PorterStemmer()
    documents = [stemmer.stem(word) for word in stopwordsList]
    return(documents)             
 
             
             
#initial declarations     
global df
df = {}
global listOfList
listOfList = {}


#Read the text files
corpusroot = 'C:/Users/Shweta/AppData/Local/Programs/Python/Python35/presidential_debates'
s = time.time()
print("Please wait till pre-processing")
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    #call to pre-processing function
    documents = preprocessing(doc)
    listOfList[filename]= calculate_tf(documents)
    calculate_df()
    file.close()   
    
print("Done Pre-processing")
print ("Took %f seconds" % (time.time() - s) ) 




    




     
  
    

    
    
    




