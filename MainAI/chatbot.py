import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import speech_recognition as sr

from ..NNCODE.MainNN import tag_sentences


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


#Reading in the corpus
with open('MainAI/chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    flat = vals.flatten()
   
    flat.sort()
    i = 0
    idx = 0
    x = random.randint(-5 , -2)
    while((flat[x] < 0.2 or flat[x] > 0.9) and i < 30):
        x = random.randint(-5 , -2)
        
        i = i + 1    
    idx=vals.argsort()[0 , x]
    req_tfidf = flat[x]

    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        myresult = tag_sentences([(user_response)])
        print(myresult)
        for f in myresult[0]:
            if ((f[1] == "verb" or f[1] == "pro" or f[1] == "det") or len(myresult[0])<2):
                sent_tokens.pop(-1)
                break

        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        myresult = tag_sentences([(user_response)])
        print(myresult)
        for f in myresult[0]:
            if ((f[1] == "verb" or f[1] == "pron" or f[1] == "det") or len(myresult[0])<2):
                sent_tokens.pop(-1)
                break

        return robo_response
    

def Bot(msg):

    user_response = msg
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            return "You are welcome.."
        elif(user_response == "listen"):
            return ListenToMe()
            
        else:
            if(greeting(user_response)!=None):
                return greeting(user_response)
            else:
                return response(user_response)
    
                
    else:
    
        return "Bye! take care..."
def ListenToMe():
    r = sr.Recognizer() # initialise a recogniser

    # Mytext = "Press on the mic to record"    

    with sr.Microphone() as source: # microphone as source

            
            audio = r.listen(source)

            try:
                
                query = r.recognize_google(audio, language='en-in')
                print(query)
                return Bot(query)
            except:
                return "Not Clear"
