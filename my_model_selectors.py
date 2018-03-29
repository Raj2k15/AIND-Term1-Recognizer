import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # BIC = −2log(L) + plog(N)
        bic_scores=[]
        
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                #print (states)
                model_bic=self.base_model(states)
                #log(L)=log_likelihood
                log_likelihood =model_bic.score(self.X, self.lengths)
                #print('log likelihood is {}' .format(log_likelihood))
                #p is the total number of parameters
                p = states ** 2 + 2 * states * model_bic.n_features - 1
                #print ('p is {}'.format(p))
                #computing BIC score
                BIC=-2*log_likelihood + p* math.log(self.X.shape[0])
                #print ('BIC is {}' .format(BIC))
                bic_scores.append(BIC)
                
            except:
                pass
            
        #print (bic_scores)
        best_state=np.argmin(bic_scores) if bic_scores else self.n_constant
        return self.base_model(best_state+2)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        dic_scores=[]
        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states)
                other_words_scores = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        other_words_scores.append(model.score(X, lengths))
                #1/(M-1)SUM(log(P(X(all but i)) is nothing but means of other words
                DIC = model.score(self.X, self.lengths) - np.mean(other_words_scores)
                dic_scores.append(DIC)
                
            except:
                pass
        
        #print (dic_scores)
        best_state=np.argmax(dic_scores) if dic_scores else self.n_constant
        return self.base_model(best_state+2)
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        
        warnings.filterwarnings("ignore")

        # TODO implement model selection using CV
        best_score = float("-inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            #ignoring all the words with only one sequence
            if len(self.sequences) <= 1:
                continue

            split_method = KFold(n_splits=min(3, len(self.sequences)))

            for train_x, test_x in split_method.split(self.sequences):
                log_likelihood = []
                X_train, Y_train = combine_sequences(train_x, self.sequences)
                X_test, Y_test = combine_sequences(test_x, self.sequences)
                try:
                    _model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X_train, Y_train)
                    log_likelihood.append(_model.score(X_test, Y_test))
                except:
                    pass
                if np.mean(log_likelihood) > best_score:
                    best_score = np.mean(log_likelihood)
                    best_model = _model

        if not best_model:
            best_model = self.base_model(self.n_constant)

        return best_model