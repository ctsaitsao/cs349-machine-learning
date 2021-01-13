import numpy as np
import sklearn
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

class TextClassificationModel:
    def __init__(self):
        self.model = None

    def _convert_lower_case(self, text):
        return np.char.lower(text)

    def _remove_punctuation(self, text):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for symbol in list(symbols):
            text = np.char.replace(text, symbol, ' ')
            text = np.char.replace(text, "  ", " ")
        text = np.char.replace(text, ',', '')
        return text

    def _remove_apostrophe(self, text):
        return np.char.replace(text, "'", "")

    def _remove_stop_words(self, text):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(text))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    def _stemming(self, text):
        stemmer = PorterStemmer()

        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def _convert_numbers(self, text):
        tokens = word_tokenize(str(text))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def _preprocess(self, text):
        text = self._convert_lower_case(text)
        text = self._remove_punctuation(text)  # remove comma seperately
        text = self._remove_apostrophe(text)
        text = self._remove_stop_words(text)
        text = self._convert_numbers(text)
        text = self._stemming(text)
        text = self._remove_punctuation(text)
        text = self._convert_numbers(text)
        text = self._stemming(text)  # needed again as we need to stem the words
        text = self._remove_punctuation(text)  # needed again as num2word is giving few hypens and commas fourty-one
        text = self._remove_stop_words(text)  # needed again as num2word is giving stop words 101 - one hundred and one
        return text

    def _tf_idf(self, texts):

        df_dict = {}

        for i in range(len(texts)):
            text = texts[i]
            for word in text:
                try:
                    df_dict[word].add(i)
                except:
                    df_dict[word] = {i}  # sets bc only counting docs

        for df in df_dict:
            df_dict[df] = len(df_dict[df])

        doc = 0

        tf_idf = {}

        for i in range(len(texts)):

            text = texts[i]
            counter = Counter(text + processed_title[i])
            words_count = len(text + processed_title[i])

            for word in np.unique(text):

                tf = counter[word]/words_count
                try:
                    df = df_dict[word]
                except:
                    pass
                idf = np.log((len(texts) + 1)/(df + 1))
                tf_idf[doc, word] = tf*idf

            doc += 1

    def train(self, texts, labels):
        """
        Trains the model.  The texts are raw strings, so you will need to find
        a way to represent them as feature vectors to apply most ML methods.

        You can implement this using any ML method you like.  You are also
        allowed to use third-party libraries such as sklearn, scipy, nltk, etc,
        with a couple exceptions:

        - The classes in sklearn.feature_extraction.text are *not* allowed, nor
          is any other library that provides similar functionality (creating
          feature vectors from text).  Part of the purpose of this project is
          to do the input featurization yourself.  You are welcome to look
          through sklearn's documentation for text featurization methods to get
          ideas; just don't import them.  Also note that using a library like
          nltk to split text into a list of words is fine.

        - An exception to the above exception is that you *are* allowed to use
          pretrained deep learning models that require specific featurization.
          For example, you might be interested in exploring pretrained
          embedding methods like "word2vec", or perhaps pretrained models like
          BERT.  To use them you have to use the same input features that the
          creators did when pre-training them, which usually means using the
          featurization code provided by the creators.  The rationale for
          allowing this is that we want you to have the opportunity to explore
          cutting-edge ML methods if you want to, and doing so should already
          be enough work that you don't need to also bother with doing
          featurization by hand.

        - When in doubt, ask an instructor or TA if a particular library
          function is allowed or not.

        Hints:
        - Don't reinvent the wheel; a little reading on what techniques are
          commonly used for featurizing text can go a long way.  For example,
          one such method (which has many variations) is TF-Idf:
          https://en.wikipedia.org/wiki/Tf-idf
          https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

        - There are multiple ways to complete the assignment.  With the right
          featurization strategy, you can pass the basic tests with one of the
          ML algorithms you implemented for the previous homeworks.  To pass
          the extra credit tests, you may need to use torch or sklearn unless
          your featurization is exceptionally good or you make some special
          modifications to your previous homework code.

        Arguments:
            texts - A list of strings representing the inputs to the model
            labels - A list of integers representing the class label for each string
        Returns:
            Nothing (just updates the parameters of the model)
        """
        processed_texts = []

        for text in texts:
            processed_texts.append(self._preprocess(text))

        

    def predict(self, texts):
        """Predicts labels for the given texts.
        Arguments:
            texts - A list of strings
        Returns:
            A list of integers representing the corresponding class labels for the inputs
        """
        # print(texts)
