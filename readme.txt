Steps to run the POS Tagger

1. #Install nltk library in the machine
 nltk library contains all the 87 different tags in English language

2. Add the corpus data from the nltk library in the folder that contains POSTaggingUsingHMM.py
corpus data contains 87 tags treebank brown corpus
It also 557166 sentences to train the tagger on so that it can learn and tag for the unknown sentence given

3. Run POSTaggingUsingHMM.py

4. Ouptut contains a test sentence with POS tag given to each word in that sequence
   Each tag is represented using abbreviations for that tag
   like NN for noun, JJ adjective,  VB for verb and so on

//////////////////////////////////////////////////////////////////////////////////////


Steps to run the Question Answering system

1. Run the file QuestionAnsweringSystem.java
2. Once all dependent files are in place 
   that includes:
	a. NLP stanford POSTagger under tagger folder
	b. Sample stories to be displayed
	c. irregular.txt file that contains all the list of irregular verbs in English language
3. Select appropriate file to or sample story and ask a question from the ones listed in the options and be a little easy on the system and ask simple question! 
4. Get amused by the answers given the QA system!