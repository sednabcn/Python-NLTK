import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from utils import text_prepare, load_embeddings, question_to_vec, unpickle_file
import numpy as np

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
       
        question_vec_=question_to_vec(question.strip(),self.word_embeddings,self.embeddings_dim)
        question_vec_=np.array([question_vec_]).reshape(1,-1)
        best_thread = pairwise_distances_argmin(question_vec_,thread_embeddings,metric='cosine')[0]
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        self.chitchat_bot = ChatBot('SiMLengBot',
                            storage_adapter='chatterbot.storage.SQLStorageAdapter',
                            logic_adapters=[{'import_path': 'chatterbot.logic.BestMatch',
                                             'default_response': 'I am sorry, but I do not understand.',
                                             'maximum_similarity_threshold': 0.90}])
        self.trainer = ChatterBotCorpusTrainer(self.chitchat_bot)
        self.trainer.train("chatterbot.corpus.english.computers",
                           "chatterbot.corpus.english.science",
                           "chatterbot.corpus.english.conversations")
        return self.chitchat_bot.get_response(self.question)
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question.strip())
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)
        
        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            self.question=question
            response = self.create_chitchat_bot()
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            print(prepared_question,tag)
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id =self.thread_ranker.get_best_thread(prepared_question,tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

