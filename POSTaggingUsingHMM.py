#Implement parts of speech tagging using HMM model in python
#POS tagging is what parts of speech is assigned to a word in a sentence or a doccument
# Basically this is used for natural lanuage process applications
# NLP chat bots like Siri, ELiza use tags for answer formulations and to give answers in natural language
#In our domain i.e Question answering system, we can break the entire questions into tokens and give tags to each tokens
# we can also break our entire text file into tokens and give them tags for answer formulation process

#Vaibhav Shukla, Northeastern University


# Say words = w1....wN
# and tags = t1..tN
#
# then
# P(tags | words) is_proportional_to  product P(ti | t{i-1}) P(wi | ti)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(tags | words)

#Import the nltk library and the brown tagset and the brown corpus which will be sued to give tags for the statements.
import nltk

#import the corpus
from nltk.corpus import brown

# Estimating P(wi | ti) from corpus data using Maximum Likelihood Estimation (MLE):
# P(wi | ti) = count(wi, ti) / count(ti)

#list of all the unique tags from the corpus
brown_word_tags = []

#Manually add a start and an end tag for each statement
for brown_sent in brown.tagged_sents():
    brown_word_tags.append(('START','START'))

    for words,tag in brown_sent:
        brown_word_tags.extend([(tag[:2], words)])

    brown_word_tags.append(("END", "END"))

# get the conditional frequency distribution for the brown word tags
cfd_tag_words = nltk.ConditionalFreqDist(brown_word_tags)
# get the conditional probability distribution for the tag words obtained
cpd_tag_words = nltk.ConditionalProbDist(cfd_tag_words, nltk.MLEProbDist)


print("The probability of an adjective (JJ) being 'smart' is", cpd_tag_words["JJ"].prob("smart"))
print("The probability of a verb (VB) being 'try' is", cpd_tag_words["VB"].prob("try"))


# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
brown_tags = []
for tag, words in brown_word_tags:
    brown_tags.append(tag)

#make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

print
print('The probability of DT occuring after NN is : ', cpd_tags["NN"].prob("DT"))
print('The probability of VB occuring after NN is : ', cpd_tags["NN"].prob("VB"))


###
# putting things together:
# what is the probability of the tag sequence "PP VB NN" for the word sequence "I love food"?
# It is
# P(START) * P(PP|START) * P(I | PP) *
#            P(VB | PP) * P(love | VB) *
#            P(TO | VB) * P(food | NN) *
#            P(END | VB)
#
# We leave aside P(START) for now.
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tag_words["PP"].prob("I") * \
                   cpd_tags["PP"].prob("VB") * cpd_tag_words["VB"].prob("love") * \
                   cpd_tags["VB"].prob("NN") * cpd_tag_words["PP"].prob("food") * \
                   cpd_tags["NN"].prob("END")
print
print("The probability of sentence 'I love food' having the tag sequence 'START PP VB PP END' is : ", prob_tagsequence)


#Implementing the Viterbi algorithm to get the tags of the various tokens generated from a sentence or a text file

# Viterbi:
# If we have a word sequence, what is the best tag sequence?
#
# The method above lets us determine the probability for a single tag sequence.
# But in order to find the best tag sequence, we need the probability
# for _all_ tag sequence.
# What Viterbi gives us is just a good way of computing all those many probabilities
# as fast as possible.

distinct_brown_tags = set(brown_tags)

sample_sentence = ["I", "love", "spicy", "food"]
len_sample_sentence = len(sample_sentence)



viterbi_tag = {}
viterbi_backpointer = {}

for tag in distinct_brown_tags:
    if tag == "START":
        continue
    viterbi_tag[tag] = cpd_tags["START"].prob(tag) * cpd_tag_words[tag].prob(sample_sentence[0])
    viterbi_backpointer[tag] = "START"


# for each step i in 1 .. sentlen,
# store a dictionary
# that maps each tag X
# to the probability of the best tag sequence of length i that ends in X
viterbi_main = []

# for each step i in 1..sentlen,
# store a dictionary
# that maps each tag X
# to the previous tag in the best tag sequence of length i that ends in X
backpointer_main = []


viterbi_main.append(viterbi_tag)
backpointer_main.append(viterbi_backpointer)

current_best = max(viterbi_tag.keys(), key=lambda tag: viterbi_tag[tag])

print
print("Word", "'" + sample_sentence[0] + "'", "current best two-tag sequence:", viterbi_backpointer[current_best], current_best)

for index in range(1,len_sample_sentence):
    curr_viterbi = {}
    curr_backpointer = {}
    prev_viterbi = viterbi_main[-1]

    for brown_tag in distinct_brown_tags:

        if brown_tag != "START":
            # if this tag is X and the current word is w, then
            # find the previous tag Y such that
            # the best tag sequence that ends in X
            # actually ends in Y X
            # that is, the Y that maximizes
            # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
            # The following command has the same notation
            # that you saw in the sorted() command.
            prev_best = max(prev_viterbi.keys(),
                                key=lambda prevtag: \
                                    prev_viterbi[prevtag] * cpd_tags[prevtag].prob(brown_tag) * cpd_tag_words[brown_tag].prob(
                                        sample_sentence[index]))

            curr_viterbi[brown_tag] = prev_viterbi[prev_best] * \
                                cpd_tags[prev_best].prob(brown_tag) * cpd_tag_words[brown_tag].prob(sample_sentence[index])
            curr_backpointer[brown_tag] = prev_best

    current_best = max(curr_viterbi.keys(), key=lambda tag: curr_viterbi[tag])
    print("Word", "'" + sample_sentence[index] + "'", "current best two-tag sequence:", curr_backpointer[current_best], current_best)


    viterbi_main.append(curr_viterbi)
    backpointer_main.append(curr_backpointer)


# now find the probability of each tag
# to have "END" as the next tag,
# and use that to find the overall best sequence
prev_viterbi = viterbi_main[-1]
prev_best = max(prev_viterbi.keys(),
                    key=lambda prev_tag: prev_viterbi[prev_tag] * cpd_tags[prev_tag].prob("END"))

prob_tag_sequence = prev_viterbi[prev_best] * cpd_tags[prev_best].prob("END")


best_tag_sequence = ["END", prev_best]
# invert the list of backpointers
backpointer_main.reverse()

# go backwards through the list of backpointers
# (or in this case forward, because we have inverter the backpointer list)
# in each case:
# the following best tag is the one listed under
# the backpointer for the current best tag
current_best_tag = prev_best
for backpointer in backpointer_main:
    best_tag_sequence.append(backpointer[current_best_tag])
    current_best_tag = backpointer[current_best_tag]


best_tag_sequence.reverse()
print
print "The sentence given is :"
for word in sample_sentence:
    print word,"",

print
print
print "The best tag sequence using HMM for the given sentence is : "


for best_tag in best_tag_sequence:
    print best_tag, "",

print
print
print "The probability of the best tag sequence printed above is given by : ", prob_tag_sequence
