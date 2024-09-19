import os
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from nltk.corpus import stopwords
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#yo wsg i think it works
# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load SpaCy eng mdl
nlp = spacy.load('en_core_web_sm')
# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# like product-specific aspects that i want in my report
aspects = {
    'comfort': ['comfortable', 'comfy', 'cozy', 'cushioned'],
    'durability': ['durable', 'long-lasting', 'sturdy', 'quality'],
    'style': ['stylish', 'fashionable', 'trendy', 'cute'],
    'price': ['expensive', 'cheap', 'affordable', 'overpriced'],
    'fit': ['fit', 'size', 'tight', 'loose']
}

#  tokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

#  load and preprocess data
def load_data(file_path):
    # Read CSV file
    data = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
    
    # Combine all text fields into a single column for each entry
    text_columns = data.select_dtypes(include=[object]).columns.tolist()
    data[text_columns] = data[text_columns].fillna('')
    data['combined_text'] = data[text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return data

#filter data based on keywords
def filter_data_by_keywords(data, keywords):
    # Create a regex pattern for the keywords
    pattern = '|'.join([re.escape(keyword) for keyword in keywords])
    # Filter data to include only entries mentioning any of the keywords
    filtered_data = data[data['combined_text'].str.contains(pattern, case=False, na=False)]
    return filtered_data

# clean data shi
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with space
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to preprocess text: Tokenization, Lemmatization, and Stopword Removal
def preprocess_text(text, nlp, stop_words):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words and not token.is_punct and not token.is_space
    ]
    return ' '.join(tokens)

#load NRC Emotion Lexicon
def load_nrc_lexicon():
    nrc_wordlevel = {}
    nrc_senselevel = {}

    wordlevel_file = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
    senselevel_file = 'NRC-Emotion-Lexicon-Senselevel-v0.92.txt'

    # Load the Wordlevel lexicon
    if os.path.exists(wordlevel_file):
        try:
            with open(wordlevel_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    word, emotion, association = parts
                    if association == '1':  # Only keep associations with '1'
                        if word not in nrc_wordlevel:
                            nrc_wordlevel[word] = []
                        nrc_wordlevel[word].append(emotion)
        except Exception as e:
            print(f"Error reading Wordlevel Lexicon: {e}")
    else:
        print(f"Wordlevel lexicon file '{wordlevel_file}' not found.")

    # Load the Senselevel lexicon
    if os.path.exists(senselevel_file):
        try:
            with open(senselevel_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    sense, emotion, association = parts
                    if association == '1':  # Only keep associations with '1'
                        if sense not in nrc_senselevel:
                            nrc_senselevel[sense] = []
                        nrc_senselevel[sense].append(emotion)
        except Exception as e:
            print(f"Error reading Senselevel Lexicon: {e}")
    else:
        print(f"Senselevel lexicon file '{senselevel_file}' not found.")

    return nrc_wordlevel, nrc_senselevel

# Gettt emotions from text - ooo cooooooollllllll
def get_emotions(text, nrc_wordlevel, nrc_senselevel):
    emotions = []
    words = tokenizer.tokenize(text)
    
    for word in words:
        word = word.lower()

        # Check Wordlevel lexicon
        if word in nrc_wordlevel:
            emotions.extend(nrc_wordlevel[word])

        # Check Senselevel lexicon (optional based on your usage)
        for sense in nrc_senselevel:
            if word in sense:  # Simple matching, you can refine this
                emotions.extend(nrc_senselevel[sense])

    return emotions

# Extract like the primary emotion
def get_primary_emotion(emotion_list):
    if emotion_list:
        emotion_counts = Counter(emotion_list)
        primary_emotion = emotion_counts.most_common(1)[0][0]
        return primary_emotion
    else:
        return 'neutral'

# Perform sentiment analysis using VADER -  seems slightly bs but idk adds some value i think
def get_vader_sentiment(text, vader_analyzer):
    score = vader_analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, compound

# Function to detect sarcasm - prolly need to improve this
def detect_sarcasm(text):
    if '/s' in text.lower() or 'sarcasm' in text.lower():
        return True
    return False

# Function to perform Named Entity Recognition (NER)
def get_entities(text, nlp):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract representative comments
def get_representative_comments(df, label_col, label_value, num_comments=3):
    comments = df[df[label_col] == label_value]['cleaned_text'].head(num_comments).tolist()
    return comments

# Function to extract product-specific aspects
def extract_aspects(text):
    found_aspects = []
    for aspect, keywords in aspects.items():
        if any(keyword in text.lower() for keyword in keywords):
            found_aspects.append(aspect)
    return found_aspects

# Function to perform aspect-based sentiment analysis
def aspect_sentiment(row):
    aspect_sentiments = {}
    for aspect in row['aspects']:
        aspect_keywords = aspects[aspect]
        text = row['cleaned_text']
        words = text.split()
        aspect_words = [word for word in words if word in aspect_keywords]
        aspect_text = ' '.join(aspect_words)
        if aspect_text:
            sentiment, _ = get_vader_sentiment(aspect_text, vader_analyzer)
            aspect_sentiments[aspect] = sentiment
        else:
            aspect_sentiments[aspect] = 'Neutral'
    return aspect_sentiments
    

# Main function to run the analysis
def main():
    # File path to the CSV file
    file_path = 'data3.csv'  # Update this to your actual file path

    # Load data
    data = load_data(file_path)

    # List of keywords to filter comments
    keywords = [
    'birkenstock', 'sandals', 'shoes', 'footwear', 'clogs', 'slippers', 'flip-flops', 
    'sneakers', 'boots', 'comfort', 'fit', 'size', 'arch support', 'cushioning', 
    'durability', 'quality', 'leather', 'suede', 'material', 'design', 'style', 'fashion', 
    'trendy', 'classic', 'casual', 'outdoor', 'hiking', 'everyday wear', 'price', 
    'cost', 'affordable', 'expensive', 'cheap', 'deal', 'sale', 'discount', 'value', '$', 
    'worth', 'luxury', 'premium', 'overpriced', 'buy', 'bought', 'purchase', 'ordered', 
    'opinion', 'review', 'feedback', 'recommend', 'customer', 'experience', 'return', 
    'exchange', 'refund', 'comfortable', 'discomfort', 'pain', 'blisters', 'satisfaction', 
    'unsatisfied', 'brand', 'name', 'trust', 'reliable', 'high quality', 'low quality', 
    'support', 'sturdy', 'long-lasting', 'repair', 'maintenance', 'wear', 'tear', 
    'lightweight', 'heavy', 'flexible', 'rigid', 'waterproof', 'weatherproof', 'color', 
    'fit well', 'tight', 'loose', 'adjustable', 'straps', 'buckle', 'velcro', 'laces',

    # Birkenstock model names
    'arizona', 'gizeh', 'madrid', 'boston', 'mayari', 'yara', 'zurich', 'milano', 
    'kyoto', 'honolulu', 'florida', 'siena', 'kumba', 'ramsey', 'franca', 'buckley', 
    'barbados', 'bend', 'davos', 'buckley', 'zana', 'essentials', 'evelyn', 'dunham', 
    'super-birki', 'a630', 'professional', 'super-grip',

    # Broadly related terms
    'foot care', 'pedorthic', 'orthotic', 'insole', 'foot health', 'pronation', 
    'supination', 'plantar fasciitis', 'flat feet', 'high arches', 'posture', 
    'alignment', 'joint pain', 'foot pain', 'heel pain', 'ankle pain', 'walking', 
    'standing', 'pressure relief', 'motion control', 'anatomical', 'bio-mechanical', 
    'ergonomic', 'sustainable', 'eco-friendly', 'recycled materials', 'environmentally friendly', 

    # Customer sentiment and experience
    'love', 'hate', 'like', 'dislike', 'happy', 'unhappy', 'excited', 'disappointed', 
    'great', 'terrible', 'amazing', 'awful', 'best', 'worst', 'perfect', 'poor', 
    'recommend', 'suggest', 'five-star', 'one-star', 'rating', 'favorite', 'prefer', 
    'customer service', 'delivery', 'shipping', 'slow shipping', 'fast shipping', 
    'return policy', 'warranty', 'guarantee', 'money back', 'exchange',

    # Market and consumer behavior
    'competition', 'market share', 'demand', 'supply', 'trend', 'fashion trend', 
    'seasonal', 'limited edition', 'new collection', 'product launch', 'in stock', 
    'out of stock', 'available', 'pre-order', 'best-seller', 'top-selling', 
    'consumer', 'buyer', 'shopper', 'retailer', 'online store', 'physical store'
]



    # Filter data by keywords
    data = filter_data_by_keywords(data, keywords)

    # If no data remains after filtering, exit
    if data.empty:
        print(f"No comments mentioning any of the keywords {keywords} found in the data.")
        exit()

    # Clean the combined text
    data['cleaned_text'] = data['combined_text'].apply(clean_text)

    # Preprocess text
    stop_words = set(stopwords.words('english'))
    data['processed_text'] = data['cleaned_text'].apply(lambda x: preprocess_text(x, nlp, stop_words))

    # Load NRC Emotion Lexicon
    nrc_lexicon = load_nrc_lexicon()

    # Get emotions
    nrc_wordlevel, nrc_senselevel = load_nrc_lexicon()
    data['emotions'] = data['cleaned_text'].apply(lambda x: get_emotions(x, nrc_wordlevel, nrc_senselevel))

    # Get primary emotion
    data['primary_emotion'] = data['emotions'].apply(get_primary_emotion)

    # Perform sentiment analysis
    data[['sentiment_label', 'sentiment_score']] = data['cleaned_text'].apply(
        lambda x: pd.Series(get_vader_sentiment(x, vader_analyzer))
    )

    # Adjust sentiment for sarcasm
    data['is_sarcastic'] = data['cleaned_text'].apply(detect_sarcasm)
    data.loc[data['is_sarcastic'], 'sentiment_label'] = 'Sarcasm'

    # Named Entity Recognition (NER)
    data['entities'] = data['cleaned_text'].apply(lambda x: get_entities(x, nlp))

    # Extract product-specific aspects
    data['aspects'] = data['cleaned_text'].apply(extract_aspects)

    # Perform aspect-based sentiment analysis
    data['aspect_sentiments'] = data.apply(aspect_sentiment, axis=1)

    # Save the DataFrame with analysis results
    data.to_csv('sentiment_analysis_results.csv', index=False)
    print("Analysis results saved to 'sentiment_analysis_results.csv'.")

    # Aggregate results
    overall_sentiment = data['sentiment_label'].value_counts(normalize=True).to_dict()
    aspect_distribution = Counter([aspect for aspects in data['aspects'] for aspect in aspects])
    aspect_sentiments = {aspect: Counter() for aspect in aspect_distribution.keys()}

    for _, row in data.iterrows():
        for aspect, sentiment in row['aspect_sentiments'].items():
            aspect_sentiments[aspect][sentiment] += 1

    # Generate report
    report = "Birkenstock Product Sentiment Analysis Report\n"
    report += "==========================================\n\n"
    
    report += "1. Overall Sentiment:\n"
    total_comments = len(data)
    for sentiment, count in data['sentiment_label'].value_counts().items():
        percentage = count / total_comments
        report += f"   {sentiment}: {percentage:.2%}\n"
    
    report += "\n2. Most Discussed Aspects:\n"
    for aspect, count in aspect_distribution.most_common():
        report += f"   {aspect.capitalize()}: {count} mentions\n"
    
    report += "\n3. Aspect-Based Sentiment:\n"
    for aspect, sentiments in aspect_sentiments.items():
        total = sum(sentiments.values())
        report += f"   {aspect.capitalize()}:\n"
        for sentiment, count in sentiments.items():
            percentage = count / total
            report += f"      {sentiment}: {percentage:.2%}\n"
    
    report += "\n4. Representative Comments:\n"
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        report += f"   {sentiment} Comments:\n"
        comments = get_representative_comments(data, 'sentiment_label', sentiment)
        for comment in comments:
            report += f"      - {comment}\n"
    
    # Save the report
    with open('birkenstock_sentiment_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis complete. Report saved as 'birkenstock_sentiment_report.txt'.")

if __name__ == "__main__":
    main()