import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy
import spacy_syllables
from collections import Counter

class FeatureConstructor:
    POS_LIST = [
        "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X", "SPACE"
    ]
    
    def __init__(self):
        self.__nlp = spacy.load("en_core_web_sm")
        self.__nlp.add_pipe("syllables", after="tagger", config={"lang": "en_US"})

        print("Loading excess words list from GitHub...")
        url = 'https://raw.githubusercontent.com/berenslab/chatgpt-excess-words/main/results/excess_words.csv'
        self.__excess_words_df = pd.read_csv(url).drop(columns=["type", "part_of_speech", "comment"])


    def construct_features(self, df):
        feature_data = {
            "sentence_count": [],
            "avg_word_per_sentence": [],
            "avg_word_length": [],
            "paragraph_count": [],
            "avg_sentence_length": [],
            "comma_frequency": [],
            "semicolon_frequency": [],
            "question_mark_frequency": [],
            "exclamation_mark_frequency": [],
            "dash_frequency": [],
            "reading_ease_score": [],
            "flesch_kincaid_grade": [],
            "uppercase_word_ratio": [],
            "title_case_word_ratio": [],
            "stop_word_ratio": [],
            "named_entity_ratio": [],
            "lexical_diversity": [],
            "word_repetition_ratio": [],
            "past_tense_ratio": [],
            "present_tense_ratio": [],
            "future_tense_ratio": [],
            "passive_voice_ratio": [],
            "excess_word_ratio": [],

        }
        
        # Add POS tag columns
        for pos in self.POS_LIST:
            feature_data[f"POS_{pos}"] = []
        
        for row in tqdm(df["text"], desc="Constructing features"):
            text = self.__nlp(row)

            feature_data["sentence_count"].append(self.sentence_count(text))
            feature_data["avg_word_per_sentence"].append(self.avg_word_per_sentence(text))
            feature_data["avg_word_length"].append(self.avg_word_length(text))
            feature_data["paragraph_count"].append(self.paragraph_count(text))
            feature_data["avg_sentence_length"].append(self.avg_sentence_length(text))
            feature_data["comma_frequency"].append(self.comma_frequency(text))
            feature_data["semicolon_frequency"].append(self.semicolon_frequency(text))
            feature_data["question_mark_frequency"].append(self.question_mark_frequency(text))
            feature_data["exclamation_mark_frequency"].append(self.exclamation_mark_frequency(text))
            feature_data["dash_frequency"].append(self.dash_frequency(text))
            feature_data["reading_ease_score"].append(self.reading_ease_score(text))
            feature_data["flesch_kincaid_grade"].append(self.flesch_kincaid_grade(text))
            feature_data["uppercase_word_ratio"].append(self.uppercase_word_ratio(text))
            feature_data["title_case_word_ratio"].append(self.title_case_word_ratio(text))
            feature_data["stop_word_ratio"].append(self.stop_word_ratio(text))
            feature_data["named_entity_ratio"].append(self.named_entity_ratio(text))
            feature_data["lexical_diversity"].append(self.lexical_diversity(text))
            feature_data["word_repetition_ratio"].append(self.word_repetition_ratio(text))
            
            # verb_tense_ratio returns a tuple of 3 values
            past_tense, present_tense, future_tense = self.verb_tense_ratio(text)
            feature_data["past_tense_ratio"].append(past_tense)
            feature_data["present_tense_ratio"].append(present_tense)
            feature_data["future_tense_ratio"].append(future_tense)
            
            feature_data["passive_voice_ratio"].append(self.passive_voice_ratio(text))
            feature_data["excess_word_ratio"].append(self.excess_word_ratio(text))
            
            # POS_tag_distribution returns a dictionary
            pos_distribution = self.POS_tag_distribution(text)
            for pos in self.POS_LIST:
                feature_data[f"POS_{pos}"].append(pos_distribution.get(pos, 0))
        
        features_df = pd.DataFrame(feature_data)
        
        # Drop original text column if present and return
        if 'text' in features_df.columns:
            features_df = features_df.drop(columns=['text'])
        
        return features_df
        
    
    def sentence_count(self, text):

        sentences = list(text.sents)
        return len(sentences)
    
    def avg_word_per_sentence(self, text):

        num_sentences = len(list(text.sents))
        num_words = len(text)
        
        if num_sentences == 0:
            return 0
        
        return num_words / num_sentences
    
    def avg_word_length(self, text):

        words = [token.text for token in text if token.is_alpha]
        
        if len(words) == 0:
            return 0
        
        total_length = sum(len(word) for word in words)
        return total_length / len(words)
    
    def paragraph_count(self, text):

        paragraphs = text.text.split("\n\n")
        return len(paragraphs)
    
    def avg_sentence_length(self, text):

        sentences = list(text.sents)
        
        if len(sentences) == 0:
            return 0
        
        total_length = sum(len(sentence) for sentence in sentences)
        return total_length / len(sentences)
    
    def POS_tag_distribution(self, text):        

        pos_counts = Counter(token.pos_ for token in text)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return {pos: 0 for pos in self.POS_LIST}
        
        return {pos: count / total_tokens for pos, count in pos_counts.items()}

    def comma_frequency(self, text):

        comma_count = sum(1 for token in text if token.text == ",")
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return comma_count / total_tokens
    
    def semicolon_frequency(self, text):

        semicolon_count = sum(1 for token in text if token.text == ";")
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return semicolon_count / total_tokens
    
    def question_mark_frequency(self, text):

        question_mark_count = sum(1 for token in text if token.text == "?")
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return question_mark_count / total_tokens
        
    def exclamation_mark_frequency(self, text):

        exclamation_mark_count = sum(1 for token in text if token.text == "!")
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return exclamation_mark_count / total_tokens
    
    def dash_frequency(self, text):

        dash_count = sum(1 for token in text if token.text == "-")
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return dash_count / total_tokens
    
    def reading_ease_score(self, text):
        # Flesch reading ease score
        
        num_sentences = len(list(text.sents))
        num_words = len(text)
        num_syllables = sum(token._.syllables_count for token in text if token.is_alpha)
        
        if num_sentences == 0 or num_words == 0:
            return 0
        
        return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    
    def flesch_kincaid_grade(self, text):
        # Flesch-Kincaid grade level

        num_sentences = len(list(text.sents))
        num_words = len(text)
        num_syllables = sum(token._.syllables_count for token in text if token.is_alpha)
        
        if num_sentences == 0 or num_words == 0:
            return 0
        
        return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    
    def uppercase_word_ratio(self, text):

        uppercase_count = sum(1 for token in text if token.is_upper)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return uppercase_count / total_tokens
    
    def title_case_word_ratio(self, text):

        title_case_count = sum(1 for token in text if token.is_title)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return title_case_count / total_tokens
    
    def stop_word_ratio(self, text):

        stop_word_count = sum(1 for token in text if token.is_stop)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return stop_word_count / total_tokens
    
    def named_entity_ratio(self, text):

        named_entity_count = len(text.ents)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return named_entity_count / total_tokens
    
    def lexical_diversity(self, text):

        unique_words = set(token.text for token in text if token.is_alpha)
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        return len(unique_words) / total_tokens
    
    def word_repetition_ratio(self, text):

        word_counts = {}
        
        for token in text:
            if token.is_alpha:
                word = token.text.lower()
                word_counts[word] = word_counts.get(word, 0) + 1
        
        total_tokens = len(text)
        
        if total_tokens == 0:
            return 0
        
        repetition_count = sum(count for count in word_counts.values() if count > 1)
        return repetition_count / total_tokens
    
    def verb_tense_ratio(self, text):

        past_tense_count = sum(1 for token in text if token.tag_ in ["VBD", "VBN"])
        present_tense_count = sum(1 for token in text if token.tag_ in ["VB", "VBP", "VBZ"])
        future_tense_count = sum(1 for token in text if token.tag_ == "MD")
        total_verbs = past_tense_count + present_tense_count + future_tense_count
        
        if total_verbs == 0:
            return 0, 0, 0
        
        return past_tense_count / total_verbs, present_tense_count / total_verbs, future_tense_count / total_verbs
    
    def passive_voice_ratio(self, text):

        passive_count = sum(1 for token in text if token.dep_ == "auxpass")
        total_sentences = len(list(text.sents))
        
        if total_sentences == 0:
            return 0
        
        return passive_count / total_sentences
    
    def excess_word_ratio(self, text):
        """
            From https://github.com/berenslab/chatgpt-excess-words study
            that analyzes the frequency of "excess words" in human vs. AI-generated text
            showing that certain words are more commonly used in AI-generated text. This feature calculates
            the ratio of excess words in the input text based on a predefined list of excess words.
        """

        words = [token.text.lower() for token in text if token.is_alpha]
        total_words = len(words)
        
        if total_words == 0:
            return 0
        
        excess_word_count = sum(1 for word in words if word in self.__excess_words_df["word"].values)
        return excess_word_count / total_words