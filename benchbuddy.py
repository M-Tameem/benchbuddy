# filename: benchbuddy.py
from scipy.spatial import distance
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import csv
import os
import time
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from collections import defaultdict
import re
from project_intro_sim import *
import shutil

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_fake_data():
    try:
        shutil.copy2(os.path.join(_DATA_DIR, 'sample_introductions.csv'), os.path.join(_DATA_DIR, 'slack_introductions.csv'))
        shutil.copy2(os.path.join(_DATA_DIR, 'sample_messages.csv'), os.path.join(_DATA_DIR, 'slack_messages.csv'))
        return True
    except Exception as e:
        print(f"Error loading fake data: {e}")
        return False

def fetch_introduction_messages(token, channel_name='introductions'):
    client = WebClient(token=token)
    users = fetch_all_users(token)
    intro_messages = []

    try:
        # First, find the channel ID for #introductions
        channels = client.conversations_list()
        intro_channel_id = None
        for channel in channels['channels']:
            if channel['name'] == channel_name:
                intro_channel_id = channel['id']
                break

        if not intro_channel_id:
            print(f"Channel {channel_name} not found.")
            return []

        # Now fetch messages for each user from #introductions
        for user_id, user_name in users.items():
            messages = fetch_user_messages(user_id, intro_channel_id, token)
            for msg in messages:
                intro_messages.append([user_id, user_name, msg])

    except SlackApiError as e:
        print(f"Error fetching introduction messages: {e}")

    return intro_messages

def save_introduction_messages(token):
    messages = fetch_introduction_messages(token)
    with open(os.path.join(_DATA_DIR, 'slack_introductions.csv'), "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "User Name", "Messages"])
        writer.writerows(messages)

def clean_message_text(text):
    # Remove user mentions
    cleaned_text = re.sub(r'<@[^>]+>', '', text)
    # Remove URLs
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    return cleaned_text.strip()

def load_user_messages(filename=None):
    if filename is None:
        filename = os.path.join(_DATA_DIR, 'slack_messages.csv')
    user_messages = defaultdict(list)
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            user_messages[row[1]].append(row[2])
    return user_messages

def fetch_all_users(token):
    client = WebClient(token=token)
    try:
        response = client.users_list()
        return {member['id']: member['real_name'] for member in response['members'] if not member['is_bot'] and member['id'] != 'USLACKBOT'}
    except SlackApiError as e:
        print(f"Error fetching users: {e}")
        return {}

def fetch_user_messages(user_id, channel_id, token):
    client = WebClient(token=token)
    try:
        response = client.conversations_history(channel=channel_id, user=user_id)        
        # Filter messages based on user_id
        user_messages = [clean_message_text(message['text']) for message in response['messages'] if 'subtype' not in message and message['user'] == user_id]
        
        return user_messages
    except SlackApiError as e:
        print(f"Error fetching messages for user {user_id} in channel {channel_id}: {e}")
        return []

def fetch_channels_for_user(user_id, token):
    client = WebClient(token=token)
    try:
        response = client.users_conversations(user=user_id)
        return [channel['id'] for channel in response['channels']]
    except SlackApiError as e:
        print(f"Error fetching channels for user {user_id}: {e}")
        return []

def fetch_slack_data(token):
    users = fetch_all_users(token)
    with open(os.path.join(_DATA_DIR, 'slack_messages.csv'), "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["User ID", "User Name", "Messages"])
        all_messages = []
        for user_id, user_name in users.items():
            all_messages.clear()  # Explicitly clear the list
            channels = fetch_channels_for_user(user_id, token)
            for channel_id in channels:
                messages = fetch_user_messages(user_id, channel_id, token)
                all_messages.extend(messages)
                time.sleep(1)  # Avoid rate limits
            writer.writerow([user_id, user_name, " ".join(all_messages)])

def get_team_recommendation(number_of_teammates):
    df = pd.read_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'))
    X = df.drop(['user_id'], axis=1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Adjust the number of clusters based on the sample size
    num_clusters = min(len(X), max(number_of_teammates * 2, 10))

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(X_scaled)
    df['cluster'] = kmeans.labels_

    # Compute the global average
    global_avg = X_scaled.mean(axis=0)

    # Sort clusters by their proximity to the global average
    clusters_sorted_by_proximity = sorted([(i, distance.euclidean(centroid, global_avg)) for i, centroid in enumerate(kmeans.cluster_centers_)], key=lambda t: t[1])

    recommended_users_df = pd.DataFrame()

    for cluster_id, _ in clusters_sorted_by_proximity:
        current_cluster_df = df[df['cluster'] == cluster_id]
        recommended_users_df = pd.concat([recommended_users_df, current_cluster_df])
        
        if len(recommended_users_df) >= number_of_teammates:
            break

    # If the number of users in the chosen clusters is greater than required, select the users closest to the cluster centroid
    if len(recommended_users_df) > number_of_teammates:
        centroid_of_combined_clusters = recommended_users_df.drop(['user_id', 'cluster'], axis=1).mean().values
        recommended_users_df['dist_to_centroid'] = recommended_users_df.apply(lambda row: distance.euclidean(row.drop(['user_id', 'cluster']).values, centroid_of_combined_clusters), axis=1)
        recommended_users_df = recommended_users_df.nsmallest(number_of_teammates, 'dist_to_centroid')
    print(len(recommended_users_df[['user_id']]))
    return recommended_users_df[['user_id']]

def main():
    possible_teammates_df = pd.DataFrame()
    st.title("BenchBuddy")
    menu = st.sidebar.selectbox("Choose a function", ["Main App", "View CSV Data", "Refresh Data", "View Documentation"])

    if menu == "Main App":
        st.subheader("An AI based team optimizaton selector for BenchSci")
        st.write("Developed by Muhammad-Tameem Mughal")
        # Ensure slack_introductions.csv and slack_messages.csv exist (fallback to sample data)
        if not os.path.exists(os.path.join(_DATA_DIR, 'slack_introductions.csv')):
            load_fake_data()

        # Display slack_introductions and slack_messages CSVs with beta_expander
        with st.expander("Slack Introductions CSV"):
            introductions_df = pd.read_csv(os.path.join(_DATA_DIR, 'slack_introductions.csv'))
            st.write(introductions_df)

        with st.expander("Slack Messages CSV"):
            messages_df = pd.read_csv(os.path.join(_DATA_DIR, 'slack_messages.csv'))
            st.write(messages_df)

        # Input for project description and button to calculate possible teammates
        st.subheader("Project Description")
        project_description = st.text_area("Enter the project description:")
        if not project_description.strip():
            st.warning("Please provide a project description first.")

        if st.button("Find Possible Teammates"):
            possible_teammates_df = get_bert_similarity(project_description, introductions_df)
            possible_teammates_df.to_csv(os.path.join(_DATA_DIR, 'possible_teammates.csv'), index=False)
            st.success("Filtered teammates saved to data/possible_teammates.csv!")

        # Display possible_teammates CSV
        try:
            with st.expander("Possible Teammates CSV"):
                possible_teammates = pd.read_csv(os.path.join(_DATA_DIR, 'possible_teammates.csv'))
                st.write(possible_teammates)
        except:
            st.write("No teammates filtered yet.")

        # Button to calculate trait scores
        st.subheader("Trait Scores")
        if st.button("Calculate Trait Scores"):
            trait_scores_df = get_trait_scores_from_messages_distilbert(possible_teammates)  # Use the new function
            trait_scores_df.to_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'), index=False)
            #trait_scores_df = get_trait_scores_from_messages(possible_teammates)
            #trait_scores_df.to_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'), index=False)
            st.success("Trait scores calculated and saved to data/output_trait_scores.csv!")

        # Display output_trait_scores CSV
        try:
            with st.expander("Trait Scores CSV"):
                trait_scores_df = pd.read_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'))
                st.write(trait_scores_df)
        except:
            st.write("No trait scores calculated yet.")

        # Input for the number of teammates and button to select best team
        st.subheader("Select Best Team")
        num_teammates = st.number_input("Enter the number of teammates required:", min_value=1, value=5)

        if st.button("Select Best Team"):
            # Assuming you have a function to get the best teammates based on trait scores and num_teammates
            recommended_team = get_team_recommendation(num_teammates)
            st.write(recommended_team)


    elif menu == "View CSV Data":
            slack_message = pd.read_csv(os.path.join(_DATA_DIR, 'slack_messages.csv'))
            slack_intro = pd.read_csv(os.path.join(_DATA_DIR, 'slack_introductions.csv'))
            output_trait = pd.read_csv(os.path.join(_DATA_DIR, 'output_trait_scores.csv'))
            possible_team = pd.read_csv(os.path.join(_DATA_DIR, 'possible_teammates.csv'))
            st.subheader("Slack Messages")
            st.write(slack_message)
            st.subheader("Slack Intros")
            st.write(slack_intro)
            st.subheader("Possible Teammates")
            st.write(possible_team)
            st.subheader("Trait Scores")
            st.write(output_trait)

    elif menu == "Refresh Data":
        slack_token = st.text_input("Enter Slack Token:")
        if st.button("Refresh Slack Messages"):
            fetch_slack_data(slack_token)
            st.success("Data fetched from Slack!")
        st.write("Note: please wait between refreshing slack messages and introduction messages to avoid ratelimiting by slack API")
        if st.button("Refresh Introduction Messages"):
            save_introduction_messages(slack_token)
            st.success("Introduction messages fetched from Slack!")
        # Add this new button for loading fake data
        if st.button("Load Fake Data"):
            if load_fake_data():
                st.success("Fake data loaded successfully!")
            else:
                st.error("Failed to load fake data.")
    
    elif menu == "View Documentation":
        st.title("Features")
        st.subheader("Loading of Fake Data:")
        st.write("The program can load fake data which may be useful for testing or demonstrations.")
        st.subheader("Fetching Slack Data:")
        st.write("Directly communicates with the Slack API to retrieve all user messages and introduction messages.")
        st.subheader("Cleaning and Storing Messages:")
        st.write("Conversations and introduction messages fetched from Slack are stored in CSV files.")
        st.write("Messages are cleaned by removing user mentions and URLs.")
        st.subheader("BERT-based Similarity Calculation:")

        st.write("Uses the well-known BERT model to compute cosine similarities between a given project description and users' introduction messages.")
        st.subheader("DistilBERT-based Trait Analysis:")

        st.write("Uses a specific version of BERT (the DistilBERT model) for sentiment analysis to derive user traits from chat messages.")
        st.write("Current traits include participation, work done, compatibility, and adaptability.")
        st.subheader("Clustering for Team Recommendations:")

        st.write("Implements KMeans clustering to group users based on their trait scores.")
        st.write("Recommends teammates based on cluster centroids' proximity to the global average of traits.")

        st.title("Areas of improvement")
        st.write("There's a couple things that can be improved in this program. For one, the Slack API call could be converted to a batch call that only takes in messages from a certain period of time to avoid ratelimiting or overworking the internal servers and computer power.")
        st.write("Additionally, the models - though a working proof of concept, could all use much better fine-tuning, provided a small team were assigned to them dedicated to fine-tuning and creating labelled datasets to improve the models further. An idea to be explored could be using OpenAI's preexisting GPT/LLMs and calling their API to analyze traits - though they would have to be fine tuned very heavily to ensure the proper output.")
        st.write("Moreover, a feature that was developed but not able to be fully implemented would be an automated creation of a slack channel, google doc, and notion page for all the teammates that were selected by the model.")

if __name__ == "__main__":
    main()