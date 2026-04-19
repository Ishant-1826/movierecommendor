from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. LOAD DATA
movies_dict = pickle.load(open('movie_list.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
spm = pickle.load(open('spm.pkl', 'rb'))

# 2. LOAD OR INITIALIZE AGENT MEMORY (User Weights)
if os.path.exists('user_weights.pkl'):
    user_weights = pickle.load(open('user_weights.pkl', 'rb'))
else:
    # Initialize a 1 x 10000 vector of zeros
    user_weights = np.zeros((1, 10000))

def fetch_poster(movie_title):
    api_key = "1a660faa"
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data['Poster']
    except Exception:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster+Found"

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    movie_list = movies['title'].values
    
    if request.method == 'POST':
        selected_movie = request.form.get('movie_name')
        try:
            # Find the index of the searched movie
            idx = movies[movies['title'] == selected_movie].index[0]
            
            # --- HYBRID SCORING LOGIC ---
            # A. Calculate Personal Preference (1 x N_movies)
            user_score = spm.dot(user_weights.T).flatten()
            
            # B. Calculate Content Similarity (1 x N_movies)
            similarity_score = cosine_similarity(spm[idx], spm)[0]
            
            # C. Combine them (Alpha=0.05 gives user preference a 'nudge' effect)
            alpha = 0.1
            final_distances = similarity_score + (alpha * user_score)
            
            # Sort and select top 6 (excluding the searched movie itself)
            movie_indices = sorted(list(enumerate(final_distances)), reverse=True, key=lambda x: x[1])
            
            for i in movie_indices[1:7]:
                title = movies.iloc[i[0]].title
                recommendations.append({
                    'title': title,
                    'poster': fetch_poster(title)
                })
        except Exception as e:
            print(f"Error: {e}")
            
    return render_template('index.html', movie_list=movie_list, recommendations=recommendations)

# 3. FEEDBACK ROUTE (The 'Learning' Bridge)
@app.route('/feedback', methods=['POST'])
def feedback():
    global user_weights
    data = request.json
    movie_title = data.get('title')
    action = data.get('action') # 'like', 'dislike', or 'watchlist'

    # Rewards for our mini-Reinforcement Learning loop
    rewards = {'like': 0.05, 'dislike': -0.1, 'watchlist': 2}
    reward = rewards.get(action, 0)

    try:
        idx = movies[movies['title'] == movie_title].index[0]
        # Convert sparse row to dense for vector addition
        movie_vec = spm[idx].toarray()
        
        # Update the weights (The Learning Step)
        user_weights += reward * movie_vec
        
        # Save the updated memory
        pickle.dump(user_weights, open('user_weights.pkl', 'wb'))
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)