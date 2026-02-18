from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pandas as pd
from textblob import TextBlob
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Load the BERT model (this is a version fine-tuned for sentence embeddings)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def get_bert_similarity(project_description, introductions_df):
    user_messages = introductions_df['Messages'].tolist()
    
    # Embed the user messages and the project description
    message_embeddings = bert_model.encode(user_messages, convert_to_tensor=True)
    project_embedding = bert_model.encode(project_description, convert_to_tensor=True)

    # Compute the cosine similarities
    similarities = util.pytorch_cos_sim(project_embedding, message_embeddings)[0].cpu().numpy()
    
    # Filter out users below the threshold
    threshold = 0.3
    possible_indices = np.where(similarities > threshold)[0]

    # Create a dataframe with the possible teammates and their similarity scores
    possible_teammates_df = introductions_df.iloc[possible_indices].copy()
    possible_teammates_df['Similarity'] = similarities[possible_indices]

    print(similarities)

    return possible_teammates_df


# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
distil_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def distilbert_analyze_messages_for_trait(messages, trait):
    combined_messages = ' '.join(messages)
    
    # Tokenize the message and get the output from the model
    inputs = tokenizer(combined_messages, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = distil_model(**inputs)
    logits = outputs.logits
    scores = torch.nn.functional.softmax(logits, dim=1)[0].tolist()  # Convert logits to probabilities
    
    # The model provides scores for negative (index 0) and positive (index 1) sentiments
    positive_score = scores[1]
    
    # Convert score to a scale of 1-9
    rating = int(positive_score * 9)
    
    # Adjust rating based on trait
    if trait == "participation":
        # Assuming more participation for more positive messages
        return rating
    elif trait == "work done":
        # Assuming more work done for more neutral messages (middle of the scale)
        return 5 if rating >= 4 and rating <= 6 else (rating - 1 if rating > 6 else rating + 1)
    elif trait == "compatibility":
        # Assuming more compatibility for more positive messages
        return rating
    else:  # "adaptibility"
        # Assuming more adaptability for more neutral messages
        return 5 if rating >= 4 and rating <= 6 else (rating - 2 if rating > 6 else rating + 2)

def get_trait_scores_from_messages_distilbert(possible_teammates):
    # Load the entire message history
    slack_messages_df = pd.read_csv(os.path.join(_DATA_DIR, 'slack_messages.csv'))

    # Filter out messages only from possible teammates
    filtered_messages = slack_messages_df[slack_messages_df['User ID'].isin(possible_teammates['User ID'])]

    # Group messages by User ID and combine them into a single string
    combined_messages = filtered_messages.groupby('User ID')['Messages'].apply(' '.join).reset_index()

    trait_scores = []

    for _, row in combined_messages.iterrows():
        user_id = row['User ID']
        messages = row['Messages'].split(' ')  # Assuming you're storing multiple messages as space-separated

        traits = ["participation", "work done", "compatibility", "adaptibility"]
        scores = {}
        for trait in traits:
            score = distilbert_analyze_messages_for_trait(messages, trait)
            scores[trait] = score
        
        trait_scores.append([user_id] + list(scores.values()))

    # Save the trait scores to a CSV
    df = pd.DataFrame(trait_scores, columns=['user_id', 'participation', 'work done', 'compatibility', 'adaptibility'])
    df.to_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'), index=False)

    return df
