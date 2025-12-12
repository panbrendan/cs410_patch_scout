import pandas as pd
import numpy as np
import pickle
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATA_FILE = "osrs_master_dataset.csv"

class PatchScoutTool:
    def __init__(self, csv_file):
        print("Initializing PatchScout System...")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Could not find {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        self.df['raw_text'] = self.df['raw_text'].astype(str)
        self.df = self.df.dropna(subset=['label'])
        self.documents = self.df['raw_text'].tolist()
        self.labels = self.df['label'].tolist()
        
        print(" -> Building Search Indices (BM25 + FAISS)...")
        # BM25
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # FAISS (Semantic)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.embedder.encode(self.documents, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        print(" -> Training Real-time Classifier (SVM)...")
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        X_vectors = self.vectorizer.fit_transform(self.documents)
        
        self.classifier = LinearSVC(class_weight='balanced', random_state=42, dual='auto')
        self.classifier.fit(X_vectors, self.labels)
        
        print("\nSystem Ready! Loaded", len(self.documents), "patch notes.")

    def search(self, query, mode="hybrid", filter_label=None, top_k=5):
        ## Main Search Function
        results = []
        
        if mode == "keyword":
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k*3]
            
        elif mode == "semantic":
            query_vector = self.embedder.encode([query])
            distances, indices = self.index.search(query_vector, top_k*3)
            top_indices = indices[0]
            
        else:
            query_vector = self.embedder.encode([query])
            distances, indices = self.index.search(query_vector, top_k*3)
            top_indices = indices[0]

        for idx in top_indices:
            if idx >= len(self.documents): continue
            
            text = self.documents[idx]
            label = self.labels[idx]
            
            if filter_label and filter_label.lower() not in label.lower():
                continue
            
            results.append({"text": text, "label": label})
            
            if len(results) >= top_k:
                break
                
        return results

    def predict_new(self, text):
        ## Classifies user input using the trained model
        vec = self.vectorizer.transform([text])
        return self.classifier.predict(vec)[0]

def main():
    app = PatchScoutTool(DATA_FILE)
    
    while True:
        print("\n" + "="*50)
        print("PATCHSCOUT MAIN MENU")
        print("="*50)
        print("1. Search Patch Notes")
        print("2. Classify New Text (Demo)")
        print("Q. Quit")
        
        choice = input("Select option: ").lower()
        
        if choice == '1':
            query = input("\nEnter search query: ")
            
            print("\nFilter by Category?")
            print("[1] No Filter")
            print("[2] Bug Fixes")
            print("[3] XP/Progression")
            print("[4] Combat Balance")
            print("[5] Mobile/UI")
            print("[6] Quest/Lore")
            cat_choice = input("Choice: ")
            
            cat_map = {'2': 'Bug Fix', '3': 'XP', '4': 'Combat', '5': 'Mobile', '6': 'Quest'}
            selected_filter = cat_map.get(cat_choice, None)
            
            results = app.search(query, mode="semantic", filter_label=selected_filter)
            
            print(f"\n--- Results for '{query}'" + (f" (Filter: {selected_filter})" if selected_filter else "") + " ---")
            if not results:
                print("No results found.")
            
            for i, res in enumerate(results):
                print(f"[{i+1}] [{res['label']}] {res['text']}")
                
        elif choice == '2':
            text = input("\nEnter a theoretical patch note:\n> ")
            prediction = app.predict_new(text)
            print(f"\nAI Prediction: This is a [{prediction}]")
            
        elif choice == 'q':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()