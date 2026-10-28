from collections import defaultdict
from community import community_louvain
from pyvis.network import Network
import networkx as nx
import os
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unicodedata
import re
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from collections import Counter
from transformers import BertForSequenceClassification, BertTokenizer
import torch


# Download necessary NLTK and Spacy models
def download_models():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    spacy.cli.download('nl_core_news_lg')
    # st.write("Downloaded models")
    # spacy.cli.download('en_core_web_lg')
    pass


# Define function to load stop words based on language
def stop_words_language(language):
    if language == "NL":
        nlp = spacy.load('nl_core_news_lg')
        stop_words = set(stopwords.words('dutch'))
    else:
        nlp = spacy.load('en_core_web_lg')
        stop_words = set(stopwords.words('english'))
    return nlp, stop_words


def read_pdf(pdf_file, priority, exclude_pages=[], filter_amount=2):
    if exclude_pages is None:
        exclude_pages = []

    # Adjust for 0-based indexing
   # exclude_pages = [page - 1 for page in exclude_pages]

    all_text = ""
    page_numbers = []

    try:
        with open(pdf_file, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            # Iterate through pages except the last 'filter_amount' pages
            for page_num in range(num_pages - filter_amount):
                if page_num in exclude_pages:
                    print(
                        f"Skipping excluded page {page_num + 1} of {pdf_file}.")
                    continue

                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                all_text += page_text

                # Extract page number for each sentence on this page
                sentences = sent_tokenize(page_text)
                num_sentences = len(sentences)

                for _ in range(num_sentences):
                    page_numbers.append(page_num + 1)

            # Split text into sentences
            sentences = sent_tokenize(all_text)
            # Add sentences and their corresponding page numbers to DataFrame
            df_data = {'Phrase': sentences, 'Page': page_numbers[:len(sentences)], 'Priority': [
                priority]*len(sentences)}
            df = pd.DataFrame(df_data)
            return df

    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
        return pd.DataFrame()


def restore_split_words(sentence, nlp, stop_words):
    doc = nlp(sentence)
    new_tokens = []
    skip_next = False
    restored_words = []  # List to keep track of restored words for debugging

    # Loop through tokens except for the last one
    for i, token in enumerate(doc[:-1]):
        if skip_next:
            skip_next = False
            continue
        next_token = doc[i + 1]
        # Check if either token is a stopword or a single character
        if (token.text.lower() in stop_words or next_token.text.lower() in stop_words or
                len(token.text) == 1 or len(next_token.text) == 1):
            # Append first token only, skip combining
            new_tokens.append(token.text)
        else:
            combined_text = token.text + next_token.text
            # Check if the combined text is out-of-vocabulary
            if nlp.vocab[combined_text].is_oov:
                new_tokens.append(token.text)
            else:
                new_tokens.append(combined_text)
                # Add the original and restored word to the list
                restored_words.append(
                    (token.text, next_token.text, combined_text))
                skip_next = True  # Skip the next token since it has been combined with the current one
    if not skip_next and doc:  # Append the last token if it wasn't combined
        new_tokens.append(doc[-1].text)

    # Reconstruct sentence from new tokens
    sentence_after = ' '.join(new_tokens)

    # if restored_words:  # Print restoration changes if any
   #     print("Restoration changes:")
   #     for original1, original2, combined in restored_words:
    #        print(f"{original1} + {original2} -> {combined}")

    return sentence_after


def process_text(df, nlp, stop_words):
    processed_data = {'Phrase': [], 'Page': [], 'Priority': []}
    for _, row in df.iterrows():
        sentence = row['Phrase']

        sentence = unicodedata.normalize('NFKD', sentence)
        sentence = ''.join(
            [c for c in sentence if not unicodedata.combining(c)])
        sentence = sentence.replace('\n', '').lower()
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        sentence = sentence.replace("rightarrow", "").replace("leftarrow", "")

        # Restore potentially split words before removing stop_words
        restored_sentence = restore_split_words(
            sentence, nlp, stop_words)

        # Named entity recognition (NER) to filter out personal names
        doc = nlp(restored_sentence)  # Rebuild sentence for NER
        filtered_names = [
            ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        # if filtered_names:
        #   print("Filtered out personal names:", filtered_names)
        for name in filtered_names:
            restored_sentence = restored_sentence.replace(name, '')

        words = word_tokenize(restored_sentence)
        processed_words = [w for w in words if w.lower(
        ) not in stop_words and len(w) < 30 and len(w) > 1]

        # Store processed words as a list
        processed_data['Phrase'].append(processed_words)
        processed_data['Page'].append(row['Page'])
        processed_data['Priority'].append(row['Priority'])

    processed_df = pd.DataFrame(processed_data)
    return processed_df


def generate_word_pairs(df):
    # Define the inner function to generate word pairs for each phrase
    def generate_pairs(tokens):
        # Generate all possible combinations of word pairs
        pairs = list(combinations(tokens, 2))
        return [(word1, word2) for word1, word2 in pairs if word1 != word2]

    # Apply the generate_pairs function to each phrase to generate word pairs
    df['word_pairs'] = df['Phrase'].apply(generate_pairs)

    # Flatten the list of word pairs
    flat_pairs = [pair for sublist in df['word_pairs'] for pair in sublist]

    # Count occurrences of each word pair
    word_pair_counts = Counter(flat_pairs)

    # Create a DataFrame to store word pairs and their counts
    word_pair_df = pd.DataFrame(
        word_pair_counts.items(), columns=['Word Pair', 'Count'])
    word_pair_df['Word1'] = word_pair_df['Word Pair'].apply(lambda x: x[0])
    word_pair_df['Word2'] = word_pair_df['Word Pair'].apply(lambda x: x[1])

    # Concatenate Word1 and Word2 columns to create a new column for sorting and eliminate duplicates
    word_pair_df['Sorted Pair'] = word_pair_df.apply(
        lambda x: tuple(sorted([x['Word1'], x['Word2']])), axis=1)
    word_pair_df = word_pair_df.drop_duplicates(
        subset='Sorted Pair').drop(columns='Sorted Pair')

    return word_pair_df[['Word Pair', 'Word1', 'Word2', 'Count']]


def predict_category(word_df, Label, batch_size=256):
    label_mapping = {
        0: 'Water',
        1: 'Mobiliteit',
        2: 'Duurzaamheid',
        3: 'Overige'
    }

    # Setup GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
    model = BertForSequenceClassification.from_pretrained(
        'model_6_bertje_df_all')
    model.to(device)  # Move model to GPU

    word1_predictions = []
    word2_predictions = []

    num_batches = (len(word_df) + batch_size - 1) // batch_size

    for i in range(0, len(word_df), batch_size):
        batch_df = word_df.iloc[i:i+batch_size]

        tokens_word1 = tokenizer(batch_df['Word1'].tolist(
        ), padding=True, truncation=True, return_tensors='pt').to(device)
        tokens_word2 = tokenizer(batch_df['Word2'].tolist(
        ), padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs_word1 = model(**tokens_word1)
            outputs_word2 = model(**tokens_word2)

        probabilities_word1 = torch.softmax(outputs_word1.logits, dim=1)
        probabilities_word2 = torch.softmax(outputs_word2.logits, dim=1)

        for j in range(len(batch_df)):
            predicted_label_index_word1 = torch.argmax(
                probabilities_word1[j], dim=0).item()
            predicted_label_word1 = label_mapping[predicted_label_index_word1]
            word1_predictions.append(predicted_label_word1)

            predicted_label_index_word2 = torch.argmax(
                probabilities_word2[j], dim=0).item()
            predicted_label_word2 = label_mapping[predicted_label_index_word2]
            word2_predictions.append(predicted_label_word2)

        # Print progress
        progress = (i // batch_size) + 1
        print(f'Processed batch {progress}/{num_batches}')

    result_df = pd.DataFrame({
        'Word_Pair': word_df['Word Pair'],
        'Word1': word_df['Word1'],
        'Word1_Category': word1_predictions,
        'Word2': word_df['Word2'],
        'Word2_Category': word2_predictions,
        'Count': word_df['Count']
    })

    # Filter rows where either Word1 or Word2 matches the specified label
    result_df = result_df[(result_df['Word1_Category'] == Label) | (
        result_df['Word2_Category'] == Label)]

    return result_df


class PriorityTfidfVectorizer(TfidfVectorizer):
    def __init__(self, priorities=None, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=float, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern,
                         ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.priorities = priorities

    def fit_transform(self, raw_documents, y=None):
        tfidf_matrix = super().fit_transform(raw_documents, y)
        # Multiply tf-idf scores by the priority of the document
        for i, priority in enumerate(self.priorities):
            tfidf_matrix[i, :] *= priority
        return tfidf_matrix

# Create TF-IDF vectors


def create_tfidf_vectors(documents, priorities):
    vectorizer = PriorityTfidfVectorizer(priorities=priorities)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


def select_top_words_by_tfidf(processed_df, top_n):
    # Join the phrases by page number into single strings
    grouped_text = processed_df.groupby('Page')['Phrase'].apply(
        lambda x: ' '.join(map(str, x))).tolist()
    priorities = processed_df['Priority'].unique()

    # Calculate the TF-IDF matrix with priorities
    tfidf_matrix, feature_names = create_tfidf_vectors(
        grouped_text, priorities)

    # Create a DataFrame with the scores per page
    weightdf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    weightdf = weightdf.T  # Transpose to have words as rows
    weightdf['TF-IDF_max'] = weightdf.max(axis=1)

    # Sort the DataFrame by TF-IDF_max in descending order
    sorted_weightdf = weightdf.sort_values(by='TF-IDF_max', ascending=False)

    # Select the top N words
    top_words = sorted_weightdf.head(top_n).index.tolist()
    return top_words


# Adjust top_n if needed
def select_top_words_by_tfidf_cat(processed_df, word_df, top_n):
    # Join the phrases by page number into single strings
    grouped_text = processed_df.groupby('Page')['Phrase'].apply(
        lambda x: ' '.join(map(str, x))).tolist()
    priorities = processed_df['Priority'].unique()

    # Calculate the TF-IDF matrix with priorities
    tfidf_matrix, feature_names = create_tfidf_vectors(
        grouped_text, priorities)

    # Create a DataFrame with the scores per page
    weightdf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    weightdf = weightdf.T  # Transpose to have words as rows
    weightdf['TF-IDF_max'] = weightdf.max(axis=1)

    # Sort the DataFrame by TF-IDF_max in descending order
    sorted_weightdf = weightdf.sort_values(by='TF-IDF_max', ascending=False)

    top_words_filtered = sorted_weightdf[sorted_weightdf.index.isin(
        word_df['Word1']) | sorted_weightdf.index.isin(word_df['Word2'])]
    # Select the top N words
    top_words = top_words_filtered.head(top_n).index.tolist()
    return top_words


def create_social_graph_from_top_words(word_df, top_words, processed_df):
    social_graph_df = word_df[(word_df['Word1'].isin(top_words)) & (
        word_df['Word2'].isin(top_words))].sort_values('Word1')

    # Calculate occurrence count for each word, considering priorities
    list_of_words = list(set(list(
        social_graph_df['Word1'].unique()) + list(social_graph_df['Word2'].unique())))
    occurrence_df = pd.DataFrame({'Word': list_of_words, 'Occurrence': [
        sum(word in tokens for tokens, priority in zip(processed_df['Phrase'], processed_df['Priority']) for _ in range(int(priority))) for word in list_of_words
    ]})
    # Merge occurrences back into the social graph DataFrame
    social_graph_df = pd.merge(social_graph_df, occurrence_df, left_on='Word1', right_on='Word', how='left').rename(
        columns={'Occurrence': 'Word1_occurrence'}).drop('Word', axis=1)
    social_graph_df = pd.merge(social_graph_df, occurrence_df, left_on='Word2', right_on='Word', how='left').rename(
        columns={'Occurrence': 'Word2_occurrence'}).drop('Word', axis=1)

    # Calculate weighted count using the PMI formula
    social_graph_df['Count_weighted'] = social_graph_df['Count'] / \
        (social_graph_df['Word1_occurrence']
         * social_graph_df['Word2_occurrence'])

    return social_graph_df


def main(pdf_files, doc_weights, top_n, filter_pages, language):
    # Initialize df_data dictionary
    df_data = {'Phrase': [], 'Page': [], 'Priority': []}
    selected_pages = filter_pages
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            # Pass the selected pages to read_pdf
            exclude_pages = selected_pages.get(pdf_file, [])
            text_df = read_pdf(
                pdf_file, doc_weights[pdf_file], exclude_pages=exclude_pages)
            if text_df is not None:  # Check if read_pdf returned a valid DataFrame
                df_data['Phrase'].extend(text_df['Phrase'])
                df_data['Page'].extend(text_df['Page'])
                df_data['Priority'].extend(text_df['Priority'])
            else:
                print(f"Error reading text from {pdf_file}")
        else:
            print(f"File '{pdf_file}' does not exist.")

    df = pd.DataFrame(df_data)  # Create DataFrame from df_data
    nlp, stop_words = stop_words_language(language)
    # Process the text DataFrame
    processed_df = process_text(df, nlp, stop_words)
    # Generate word pairs df on processed data
    word_df = generate_word_pairs(processed_df)

    # Select the top 300 words based on TF-IDF scores
    top_n_words = select_top_words_by_tfidf(processed_df, top_n)

    # Create the social graph from the top 300 words
    social_graph_df = create_social_graph_from_top_words(
        word_df, top_n_words, processed_df)

    return social_graph_df, processed_df


def main_cat(pdf_files, doc_weights, Label, top_n, filter_pages, language):
    # Initialize df_data dictionary
    df_data = {'Phrase': [], 'Page': [], 'Priority': []}
    selected_pages = filter_pages
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            exclude_pages = selected_pages.get(pdf_file, [])
            text_df = read_pdf(pdf_file, doc_weights[pdf_file], exclude_pages)
            if text_df is not None:  # Check if read_pdf returned a valid DataFrame
                df_data['Phrase'].extend(text_df['Phrase'])
                df_data['Page'].extend(text_df['Page'])
                df_data['Priority'].extend(text_df['Priority'])
            else:
                print(f"Error reading text from {pdf_file}")
        else:
            print(f"File '{pdf_file}' does not exist.")

    df = pd.DataFrame(df_data)  # Create DataFrame from df_data
    nlp, stop_words = stop_words_language(language)
    # Process the text DataFrame
    processed_df = process_text(df, nlp, stop_words)
    # Generate word pairs df on processed data
    word_df = generate_word_pairs(processed_df)

    # Choices are 'Water', 'Mobiliteit', 'Duurzaamheid', 'Armoede', 'Overige'
    word_df_cat = predict_category(word_df, Label)

    # Select the top 300 words based on TF-IDF scores
    top_n_words = select_top_words_by_tfidf_cat(
        processed_df, word_df_cat, top_n)

    social_graph = create_social_graph_from_top_words(
        word_df_cat, top_n_words, processed_df)

    return social_graph, processed_df


def create_network(data):
    # Create a NetworkX graph
    G = nx.from_pandas_edgelist(data, 'Source', 'Target', edge_attr=True)

    # Initialize node importance based on occurrences
    node_importance = {node: 0 for node in G.nodes()}

    # Update the node importance based on unique Source_Occurrence and Target_Occurrence values
    for _, row in data.iterrows():
        if row['Source'] in node_importance:
            node_importance[row['Source']] = row['Source_Occurrence']
        if row['Target'] in node_importance:
            node_importance[row['Target']] = row['Target_Occurrence']

    # Min-max normalization
    max_importance = max(node_importance.values())
    min_importance = min(node_importance.values())
    scaling_factor = 40
    threshold = 0.005  # Set a small threshold to filter near-zero importance nodes

    for node in node_importance:
        if max_importance != min_importance:  # To avoid division by zero
            node_importance[node] = ((node_importance[node] - min_importance) /
                                     (max_importance - min_importance)) * scaling_factor
        else:
            # If all values are the same, set them to a default normalized value
            node_importance[node] = 1

    # Filter out nodes below the importance threshold
    filtered_nodes = {node: size for node,
                      size in node_importance.items() if size > threshold}

    # Detect communities
    partition = community_louvain.best_partition(G)

    # Filter partition to only include nodes that passed the filtering
    filtered_partition = {node: comm for node,
                          comm in partition.items() if node in filtered_nodes}

    # Community leaders remain for analysis purposes
    community_leader = {}
    for node, comm in partition.items():
        if node in filtered_nodes and (comm not in community_leader or node_importance[node] > node_importance[community_leader[comm]]):
            community_leader[comm] = node

    # Create a DataFrame to hold community information
    community_data = []

    # # Define a dictionary to map color codes to color names
    # color_dict = {
    #     '#FF0000': 'Red',
    #     '#ffdd00': 'Yellow',
    #     '#00FFFF': 'Cyan',
    #     '#00FF00': 'Green',
    #     '#ed4ed0': 'Pink',
    #     '#0335fc': 'Blue',
    #     '#8900d3': 'Purple',
    #     '#f296a2': 'Light Pink'
    # }

    # Dutch version
    # Define a dictionary to map color codes to color names
    color_dict = {
        '#FF0000': 'Rood',
        '#ffdd00': 'Geel',
        '#00FFFF': 'Cyaan',
        '#00FF00': 'Groen',
        '#ed4ed0': 'Roze',
        '#0335fc': 'Blauw',
        '#8900d3': 'Paars',
        '#f296a2': 'Licht roze'
    }
    # Create a list of color codes
    color_list = list(color_dict.keys())

    # Create community colors with the reordered color list
    community_colors = {com: color_list[i % len(
        color_list)] for i, com in enumerate(set(filtered_partition.values()))}

    # Collect community colors with names and community leaders
    for com, color in community_colors.items():
        leader = community_leader[com]
        community_data.append({
            "Community": com,
            "Leader": leader,
            "Color": color_dict[color],
        })

    # Dictionary to hold clusters with filtered nodes
    clusters = defaultdict(list)

    # Populate the dictionary with nodes grouped by their community and sorted by importance
    for node, comm in filtered_partition.items():
        clusters[comm].append((node, node_importance[node]))

    # Sort each cluster's nodes by importance in descending order
    for comm in clusters:
        clusters[comm].sort(key=lambda x: x[1], reverse=True)

    # Collect each cluster and its nodes using community leader names
    for comm in sorted(clusters.keys()):
        leader = community_leader[comm]
        for node, importance in clusters[comm]:
            community_data.append({
                "Community": comm,
                "Leader": leader,
                "Node": node,
                "Node_Weight": importance,
                "Color": color_dict[community_colors[comm]],
            })

    # Create a DataFrame from the community data
    community_df = pd.DataFrame(community_data)

    # Initialize PyVis network with ForceAtlas2 and improved label visibility
    net = Network(height="750px", width="100%", bgcolor="#222222",
                  font_color="white", notebook=True)

    # Set options with adjusted label visibility
    options = """
    {
        "nodes": {
            "font": {
                "size": 16,
                "face": "arial"
            },
            "scaling": {
                "label": {
                    "enabled": true,
                    "min": 8,
                    "max": 30
                }
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": false
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -1500,
                "centralGravity": 0.1,
                "springLength": 100,
                "springStrength": 0.08,
                "damping": 0.4
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "tooltipDelay": 200,
            "hideEdgesOnDrag": false
        },
        "configure": {
                    "enabled": true,
                    "filter": "physics"
        }
    }
    """
    net.set_options(options)

    # Add filtered nodes to the PyVis network
    for node, size in filtered_nodes.items():
        net.add_node(node, title=node, size=size,
                     color=community_colors[filtered_partition[node]])

    # Add edges only between nodes present in the filtered nodes list
    for source, target, weight in G.edges(data=True):
        if source in filtered_nodes and target in filtered_nodes:
            net.add_edge(source, target, value=weight['Edge_Weight'])

    # Save the network to an HTML file
    net.show("water_network.html")

    return community_df, "water_network.html"
