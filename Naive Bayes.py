import numpy as np
#enter dataset path on line 111

def Clean(lines): #Remove punctuations and split lines and labels
	punc = '''!()-[]{};:"\, <>./?@#$%^&*_~'''
	for i in range(len(lines)):
		line =lines[i]
		for char in line:
			if char in punc:
				lines[i]=lines[i].replace(char," ").lower();


	labels=[line[-1] for line in lines]
	lines=[line[:-1] for line in lines]

	return lines,labels

def Predict(lines,probData): #Classify samples
	predicted=[]
	posTable,negTable,posPrior,negPrior=probData

	#Finding probabilties of features not occuring
	posBar=1-np.array(list(posTable.values()))
	negBar=1-np.array(list(negTable.values()))

	for sample in lines:
		#Assuming initial probability to be probability of features not occuring
		pos=np.prod(posBar)
		neg=np.prod(negBar)
		for word in sample.split():
			#Replacing probability of feature not occuring(in the product) to probabilty of feature occuring if feature encountered
			if word in posTable:
				p=posTable[word]
				pos*=p/(1-p)
			if word in negTable:
				p=negTable[word]
				neg*=p/(1-p)

		#Posterior probabilties
		posPosterior=pos*posPrior
		negPosterior=neg*negPrior

		if(posPosterior>negPosterior):
			predicted.append('1')
		else:
			predicted.append('0')

	return predicted


def Keywords(lines,stopwords): #Picking keywords/features that are not stopwords
	keywordLines=[]
	for line in lines:
		words=line.split()
		keywords=[word for word in words if word not in stopwords]
		keywordLines.append(keywords)
	return keywordLines

def ProbData(keywordLines,labels): #Probability Tables for features in positive and negative samples 
	pos={}
	neg={}

	for keywords,label in zip(keywordLines,labels):
		
		#Counting occurences of features in each class
		if label=='1':
			for keyword in keywords:
				if keyword not in pos:
					pos[keyword]=1
				else:
					pos[keyword]+=1

		else:
			for keyword in keywords:
				if keyword not in neg:
					neg[keyword]=1
				else:
					neg[keyword]+=1

	#Taking union of features with non-intersecting features having 0 occurences in non-parent set
	for keywords in keywordLines:
		for keyword in keywords:
			if keyword in pos and keyword not in neg:
				neg[keyword]=0
			if keyword in neg and keyword not in pos:
				pos[keyword]=0

	#Total pos/neg samples
	postot=np.sum(np.array(labels)=='1')
	negtot=np.sum(np.array(labels)=='0')
	
	#Probabilty calculation with Laplacian smoothing
	alpha=1
	K=len(pos.keys())
	posTable={key:(pos[key]+alpha)/(postot+alpha*K) for key in pos}
	negTable={key:(neg[key]+alpha)/(negtot+alpha*K) for key in neg}

	#Prior Probabilities
	posPrior=postot/(postot+negtot)
	negPrior=1-posPrior

	return posTable,negTable,posPrior,negPrior

	
def Accuracy(predictedLabels,labels):
	return np.sum(np.array(predictedLabels)==np.array(labels))/len(labels)


#main
if __name__ == '__main__':
	with open('C:/Users/athar/Downloads/dataset_NB.txt') as data: #enter path here
		lines=data.read().splitlines()


	stopwords=['a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z']

	#K-Fold Cross Validation
	data=lines
	np.random.shuffle(data)
	k=7
	n=len(data)
	t=n/k

	Acc=[]
	for i in range(k):
		start=int(i*t)
		finish=int((i+1)*t)
		test=data[start:finish]
		train=np.delete(data,slice(start,finish),axis=0)


		lines_train,labels_train=Clean(train)
		lines_test,labels_test=Clean(test)
			
		#training
		keywordLines=Keywords(lines_train,stopwords)
		posTable,negTable,posPrior,negPrior=ProbData(keywordLines,labels_train)

		#testing
		predictedLabels=Predict(lines_test,(posTable,negTable,posPrior,negPrior))
		#accuracy
		acc=Accuracy(predictedLabels,labels_test)
		print('Accuracy{}: '.format(i+1)+str(acc*100)+"%")
		Acc.append(acc)
	
	print("Average accuracy: "+str(np.mean(Acc)*100)+"%")
		



