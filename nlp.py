import nltk 
nltk.download()

paragraph = """Choke packet technique is applicable to both virtual networks as well as datagram subnets. A choke packet is a packet sent by a node to the source to inform it of congestion. Each router monitor its resources and the utilization at each of its output lines. whenever the resource utilization exceeds the threshold value which is set by the administrator, the router directly sends a choke packet to the source giving it a feedback to reduce the traffic. The intermediate nodes through which the packets has traveled are not warned about congestion."""

sentences = nltk.sent_tokenize(paragraph)

words = nltk.word_tokenize(paragraph)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    newwords = [stemmer.stem(word) for word in words]
    sentences[i] = ' '.join(newwords)

paragraph2 = """To understand the effect of transportation infrastructure on traffic in cities, we assembled data describing the road network and travel behavior in all U.S. metropolitan areas containing interstate highways for 1980, 1990, and 2000. These data suggest a fundamental law of road congestion: adding 10 percent more lane miles to a city increases vehicle miles traveled by 10 percent. That is, in less than 10 years, new roads cause traffic increases directly proportional to the increase in capacity. This law appears to hold for major urban roads, nonurban interstate highways near major cities, and urban interstates.

The additional traffic caused by a new road has three principal sources. Of these, an increase in driving by current city residents is the most important. In addition, a 10 percent increase in the extent of the interstate network appears to result in about a 20 percent increase in truck traffic (the increase in truck traffic is less important for other roads). We also find that people migrate to cities well provided with roads. Surprisingly, new roads seem not to cause substantial decreases in traffic on old roads."""
sentences2 = nltk.sent_tokenize(paragraph2)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words1 = nltk.word_tokenize(paragraph2)
    newwords1 = [lemmatizer.lemmatize(word) for word in words1]
    sentences2[i] = ' '.join(newwords1)
    
nltk.download('stopwords')

from nltk.corpus import stopwords

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    stop = [word for word in words if word not in stopwords.words('english')]
    sentences[i] = ' '.join(stop)

words = nltk.word_tokenize(paragraph)

tagged_words = nltk.pos_tag(words)

word_tags = []
for tw in tagged_words:
    word_tags.append(tw[0]+"_"+tw[1])
    
tagged_paragraph = ' '.join(word_tags)
