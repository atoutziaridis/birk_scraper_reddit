# birk_scraper_reddot
Sentiment analysis tool for Birkenstock.
Here's a complete example of a `README.md` file that you can include in your GitHub repository:

---

# Birkenstock Sentiment Analysis

This project performs sentiment analysis on customer reviews and social media data related to Birkenstock products. It uses a combination of lexicon-based sentiment analysis and keyword-based filtering to gauge consumer sentiment towards specific aspects of Birkenstock products, such as comfort, durability, and style.

## What the Code Does

- **Text Preprocessing**: The code cleans and preprocesses raw text data by removing unnecessary characters, stopwords, and performing lemmatization.
- **Keyword Filtering**: Keywords relevant to Birkenstock products are used to filter relevant data for analysis. It also supports fuzzy matching to capture variations in spelling.
- **Sentiment Analysis**: Sentiment analysis is performed using both the **VADER** sentiment analysis tool and the **NRC Emotion Lexicon**, identifying overall sentiment as well as specific emotions (e.g., joy, sadness, anger).
- **Aspect Extraction**: Product-specific aspects like comfort, style, price, and fit are extracted from the reviews to perform aspect-based sentiment analysis.
- **Data Aggregation**: The results are saved in CSV format, and a comprehensive report of the sentiment analysis is generated, highlighting the overall sentiment, frequently mentioned product aspects, and representative comments.

## How to Use the Code

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up the Virtual Environment

Create a virtual environment to keep the dependencies isolated from your global Python installation:

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **For macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

- **For Windows**:

  ```bash
  venv\Scripts\activate
  ```

### 3. Install the Required Packages

Install the following Python packages:
- **nltk**: Natural Language Toolkit for text processing.
- **pandas**: For data manipulation and analysis.
- **spacy**: For advanced natural language processing.
- **fuzzywuzzy**: For fuzzy string matching.
- **vaderSentiment**: For sentiment analysis.
- **requests**: To download additional resources (NRC Emotion Lexicon).


Also install (can find all of these in the respective website 1-2.NRC website, 3 go to the github repo and find the file for the latest version: NRC-Emotion-Lexicon-Senselevel-v0.92
NRC-Emotion-Lexicon-Wordlevel-v0.92
en_core_web_trf-3.7.3-py3-none-any.whl


### 4. Download Necessary NLTK Data

Ensure that the necessary NLTK data, such as stopwords and the Punkt tokenizer, is downloaded:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 5. Run the Script

With everything set up, you can now run the sentiment analysis script:

```bash
python3 birk_sentiment_analysis_advanced.py
```

### 6. Deactivate the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```text
birk_sentiment_analysis/
├── birk_sentiment_analysis_advanced.py  # Main script for sentiment analysis
├── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt  # NRC Emotion Lexicon (word-level)
├── NRC-Emotion-Lexicon-Senselevel-v0.92.txt  # NRC Emotion Lexicon (sense-level)
├── data3.csv  # Input data (customer reviews)
├── README.md  # Instructions for using the project
└── requirements.txt  # List of Python packages required for the project
```

## Requirements

- Python 3.7+
- nltk
- pandas
- spacy
- fuzzywuzzy
- vaderSentiment
- requests

These packages will be installed when you run `pip install -r requirements.txt`.

## Notes

- Make sure to have your review data in `data3.csv` in the appropriate format (with text fields).
- You can customize the keyword list or product-specific aspects in the code to fit your use case.
  
---
