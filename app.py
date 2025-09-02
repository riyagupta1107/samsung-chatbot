import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZATION ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Initialize Pinecone Client (using modern v3.x syntax)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the .env file.")
# This is the modern way to initialize
pc = Pinecone(api_key=pinecone_api_key)


# Initialize Gemini Client
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")
genai.configure(api_key=gemini_api_key)


# --- CONNECT TO PINECONE INDEX ---
index_name = "samsung-wm"
# The modern client lists indexes as objects, so we get their names.
if index_name not in [index.name for index in pc.list_indexes()]:
    raise KeyError(f"Pinecone index '{index_name}' does not exist. Please ensure it's created.")
index = pc.Index(index_name)


# --- HELPER FUNCTION ---
def get_embedding(text, model="models/embedding-001"):
   """Generates a vector embedding for the given text using Gemini."""
   text = text.strip().replace("\n", " ")
   try:
       # Call the Gemini API to get the embedding
       result = genai.embed_content(model=model, content=text)
       return result['embedding']
   except Exception as e:
       print(f"Error calling Gemini Embedding API: {e}")
       return None

# --- API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests by querying the Pinecone index."""
    data = request.get_json()
    user_query = data.get('query')
    product_category = data.get('category')

    if not user_query or not product_category:
        return jsonify({"error": "A 'query' and 'category' must be provided."}), 400

    # 1. Get the embedding for the user's query
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return jsonify({"error": "Failed to generate embedding for the query."}), 500

    # 2. Query Pinecone
    try:
        query_response = index.query(
            namespace=product_category,
            vector=query_embedding,
            top_k=1, # Find the single most similar result
            include_metadata=True
        )

        # 3. Formulate the response
        if query_response.matches:
            match = query_response.matches[0]
            # Based on your upsert script, the solution text is the metadata itself
            solution = match.metadata
            if isinstance(solution, str):
                 response_text = f"Based on your issue, I found this potential solution:\n\n{solution}"
            else:
                 # Fallback in case metadata is a dict, e.g. {'text': 'solution...'}
                 response_text = f"Based on your issue, I found this potential solution:\n\n{json.dumps(solution)}"
        else:
            response_text = "I'm sorry, I couldn't find a matching solution in our database. Could you please try rephrasing the problem?"

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return jsonify({"error": "An error occurred while searching for a solution."}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

