import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
from graphviz import Digraph
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

nltk.download('stopwords')

class TreeMaker:
    
    @staticmethod
    def preprocess_census_metadata(path):
        df = pd.read_json(path).T
        filtered_data = df[df["type"] == "Total"]
        print("The unique values for the type were: ", pd.unique(df["type"]), "and now it is: ", pd.unique(filtered_data["type"]))
        filtered_data = filtered_data.reset_index()
        filtered_data = filtered_data.rename(columns={"index": "vector"})
        return filtered_data

    @staticmethod
    def census_similarity(sentence1, sentence2):
        tokens1 = TreeMaker.process_census_text(sentence1)
        tokens2 = TreeMaker.process_census_text(sentence2)
        
        if not tokens1 and not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    @staticmethod
    def process_census_text(text):
        stop_words = set(stopwords.words('english'))
        # Normalize ranges first
        text = TreeMaker.normalize_ranges(text)
        
        # Extract tokens
        # Split on whitespace and punctuation, but preserve numbers and ranges
        tokens = re.findall(r'\b\d+(?:-\d+)?\b|\b[a-zA-Z]+\b', text.lower())
        
        # Filter stopwords from alphabetic tokens only
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in stop_words:
                filtered_tokens.append(token)
            elif not token.isalpha():  # Keep numbers and ranges
                filtered_tokens.append(token)
        
        return set(filtered_tokens)

    @staticmethod
    def normalize_ranges(text):
        # Convert "80,000 to 100,000" to "80000-100000"
        text = re.sub(r'(\d{1,3}(?:,\d{3})*)\s+to\s+(\d{1,3}(?:,\d{3})*)', 
                        lambda m: f"{m.group(1).replace(',', '')}-{m.group(2).replace(',', '')}", 
                        text)
        return text

    @staticmethod
    def match_descriptions_jaccard(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9):
        """
        Map descriptions in df_base to matching vectors in df_cmp.
        - Exact description → exact vector.
        - Otherwise Jaccard(token_set) ≥ threshold → first matching vector.

        Expects both DataFrames to have columns ['vector', 'description'].
        Returns a DataFrame with columns [description, vector_base, vector_cmp].
        """

        source_data = source_df.copy()
        compare_data = compare_df.copy()

        # 1) First pass: exact matches
        mapping_records = []
        matched_indices = set()
        for _, source_row in source_data.iterrows():
            source_description, source_vector = source_row['description'], source_row['vector']
            exact_matches = compare_data[(compare_data['description'] == source_description) & (~compare_data.index.isin(matched_indices))]
            if not exact_matches.empty:
                compare_idx = exact_matches.index[0]
                mapping_records.append({
                    'description': source_description,
                    'vector_base': source_vector,
                    'vector_cmp': exact_matches.iloc[0]['vector']
                })
                matched_indices.add(compare_idx)
            else:
                # Mark for second pass
                mapping_records.append({
                    'description': source_description,
                    'vector_base': source_vector,
                    'vector_cmp': None,  # To be filled in second pass if possible
                })

        # 2) Second pass: similarity matches for unmatched
        for record in mapping_records:
            if record['vector_cmp'] is not None:
                continue  # Already matched exactly
            source_description = record['description']
            best_similarity = 0
            best_match_idx = None
            best_match_vector = None
            
            for compare_idx, compare_row in compare_data[~compare_data.index.isin(matched_indices)].iterrows():
                compare_description = compare_row['description']
                similarity_score = TreeMaker.census_similarity(source_description, compare_description)
                if similarity_score >= similarity_threshold and similarity_score > best_similarity:
                    best_similarity = similarity_score
                    best_match_idx = compare_idx
                    best_match_vector = compare_row['vector']
            if best_match_idx is not None:
                record['vector_cmp'] = best_match_vector
                matched_indices.add(best_match_idx)
                

        return pd.DataFrame(mapping_records)

    @staticmethod
    def match_descriptions_transformer(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
        # 1. Exact matches
        compare_desc_to_info = {}
        for compare_idx, compare_row in compare_df.iterrows():
            desc = compare_row['description']
            if desc not in compare_desc_to_info:
                compare_desc_to_info[desc] = []
            compare_desc_to_info[desc].append((compare_idx, compare_row['vector']))

        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer(model_name)
        mapping_records = []
        used_compare_indices = set()
        for source_idx, source_row in source_df.iterrows():
            source_description, source_vector = source_row['description'], source_row['vector']
            if source_description in compare_desc_to_info:
                for compare_idx, compare_vector in compare_desc_to_info[source_description]:
                    if compare_idx not in used_compare_indices:
                        mapping_records.append({
                            'description': source_description,
                            'vector_base': source_vector,
                            'vector_cmp': compare_vector
                        })
                        used_compare_indices.add(compare_idx)
                        break
                else:
                    mapping_records.append({
                        'description': source_description,
                        'vector_base': source_vector,
                        'vector_cmp': None
                    })
            else:
                mapping_records.append({
                    'description': source_description,
                    'vector_base': source_vector,
                    'vector_cmp': None
                })
        num_exact = sum(1 for rec in mapping_records if rec['vector_cmp'] is not None)
        print(f"Number of exact matches: {num_exact}")

        # 2. Sentence transformer similarity for unmatched
        unmatched_source = [rec for rec in mapping_records if rec['vector_cmp'] is None]
        if unmatched_source:
            unmatched_source_df = pd.DataFrame(unmatched_source)
            unmatched_compare_df = compare_df.loc[~compare_df.index.isin(used_compare_indices)]

            
            # model = SentenceTransformer('all-mpnet-base-v2')
            
            source_embeddings = model.encode(unmatched_source_df['description'].tolist(), show_progress_bar=True)
            compare_embeddings = model.encode(unmatched_compare_df['description'].tolist(), show_progress_bar=True)
            sim_matrix = cosine_similarity(source_embeddings, compare_embeddings)

            for i, rec in enumerate(unmatched_source):
                similarities = sim_matrix[i]
                sorted_indices = np.argsort(-similarities)
                match_found = False
                for idx in sorted_indices:
                    if similarities[idx] < similarity_threshold:
                        break
                    compare_idx = unmatched_compare_df.index[idx]
                    if compare_idx not in used_compare_indices:
                        rec['vector_cmp'] = unmatched_compare_df.iloc[idx]['vector']
                        used_compare_indices.add(compare_idx)
                        match_found = True
                        break
                if not match_found:
                    rec['vector_cmp'] = None
        num_exact = sum(1 for rec in mapping_records if rec['vector_cmp'] is not None)
        print(f"Number of exact matches: {num_exact}")
        return pd.DataFrame(mapping_records)

    @staticmethod
    def match_descriptions_details_sentence_transformer( source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
        # 1. Pre-encode ALL descriptions at once
        model = SentenceTransformer(model_name)
        
        print("Encoding all source descriptions...")
        source_embeddings = model.encode(source_df['details'].tolist(), show_progress_bar=True)
        
        print("Encoding all compare descriptions...")
        compare_embeddings = model.encode(compare_df['details'].tolist(), show_progress_bar=True)
        
        # Create a mapping from description to embedding index
        source_desc_to_embedding_idx = {desc: idx for idx, desc in enumerate(source_df['details'])}
        compare_desc_to_embedding_idx = {desc: idx for idx, desc in enumerate(compare_df['details'])}
        
        # 2. Rest of your logic, but use pre-computed embeddings
        compare_desc_to_info = {}
        for compare_idx, compare_row in compare_df.iterrows():
            desc = compare_row['description']
            if desc not in compare_desc_to_info:
                compare_desc_to_info[desc] = []
            compare_desc_to_info[desc].append((compare_idx, compare_row['vector']))

        mapping_records = []
        used_compare_indices = set()
        
        for source_idx, source_row in source_df.iterrows():
            source_description, source_vector = source_row['description'], source_row['vector']
            source_details = source_row['details']
            
            if source_description in compare_desc_to_info:
                candidates = [
                    (compare_idx, compare_vector)
                    for compare_idx, compare_vector in compare_desc_to_info[source_description]
                    if compare_idx not in used_compare_indices
                ]
                if candidates:
                    # Get pre-computed embeddings

                    # Find the index of the source details in the pre-computed embeddings array
                    source_embedding_idx = source_desc_to_embedding_idx[source_details]
                    # Extract the embedding for this specific source details (as a 2D array for cosine_similarity)
                    source_embedding = source_embeddings[source_embedding_idx:source_embedding_idx+1]
                    
                    # Extract just the compare_df indices from the candidates list
                    candidate_indices = [compare_idx for compare_idx, _ in candidates]
                    # For each candidate, find the index of their details in the pre-computed embeddings
                    candidate_embedding_indices = [compare_desc_to_embedding_idx[compare_df.loc[idx, 'details']] for idx in candidate_indices]
                    # Use the indices to get the actual embeddings for all candidates at once
                    candidate_embeddings_subset = compare_embeddings[candidate_embedding_indices]
                    
                    # Compute similarities (fast!)
                    similarities = cosine_similarity(source_embedding, candidate_embeddings_subset)[0]
                    
                    best_idx_in_candidates = int(np.argmax(similarities))
                    best_idx, best_vector = candidates[best_idx_in_candidates]
                    
                    mapping_records.append({
                        'description': source_description,
                        'vector_base': source_vector,
                        'vector_cmp': best_vector
                    })
                    used_compare_indices.add(best_idx)
                else:
                    mapping_records.append({
                        'description': source_description,
                        'vector_base': source_vector,
                        'vector_cmp': None
                    })
            else:
                mapping_records.append({
                    'description': source_description,
                    'vector_base': source_vector,
                    'vector_cmp': None
                })

        # 2. Sentence transformer similarity for unmatched
        unmatched_source = [rec for rec in mapping_records if rec['vector_cmp'] is None]
        if unmatched_source:
            unmatched_source_df = pd.DataFrame(unmatched_source)
            unmatched_compare_df = compare_df.loc[~compare_df.index.isin(used_compare_indices)]

            
            # model = SentenceTransformer('all-mpnet-base-v2')
            
            source_embeddings = model.encode(unmatched_source_df['description'].tolist(), show_progress_bar=True)
            compare_embeddings = model.encode(unmatched_compare_df['description'].tolist(), show_progress_bar=True)
            sim_matrix = cosine_similarity(source_embeddings, compare_embeddings)

            for i, rec in enumerate(unmatched_source):
                similarities = sim_matrix[i]
                sorted_indices = np.argsort(-similarities)
                match_found = False
                for idx in sorted_indices:
                    if similarities[idx] < similarity_threshold:
                        break
                    compare_idx = unmatched_compare_df.index[idx]
                    if compare_idx not in used_compare_indices:
                        rec['vector_cmp'] = unmatched_compare_df.iloc[idx]['vector']
                        used_compare_indices.add(compare_idx)
                        match_found = True
                        break
                if not match_found:
                    rec['vector_cmp'] = None

        return pd.DataFrame(mapping_records)

    @staticmethod
    def merge_mappings(map_descriptions, *mappings_dfs):
        """
        Start with base DataFrame (2021) and for each description,
        collect all matching vectors from the mapping DataFrames.

        Args:
            base_df: Base DataFrame (e.g., data_2021) with columns ['description', 'vector']
            *mappings: Mapping DataFrames with columns ['description', 'vector_base', 'vector_cmp']

        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp_list']
        """
        merged_mappings = []

        # For each description in the base DataFrame (2021)
        for _, source_row in map_descriptions.iterrows():
            source_description = source_row['description']
            source_vector = source_row['vector']

            # Collect all matching vectors from all mappings
            target_vectors = []

            for mapping_df in mappings_dfs:
                mapping_df = mapping_df[mapping_df['vector_cmp'].notnull()]
                # Find rows in this mapping that match the vector_base
                matching_rows = mapping_df[mapping_df['vector_base'] == source_vector]

                # Add all vector_cmp values from this mapping
                for _, match_row in matching_rows.iterrows():
                    target_vectors.append(match_row['vector_cmp'])

            # Add to result (even if vector_cmp_list is empty)
            merged_mappings.append({
                'description': source_description,
                'vector_base': source_vector,
                'vector_cmp_list': target_vectors
            })

        result_df = pd.DataFrame(merged_mappings)

        # Filter out rows with empty vector_cmp_list
        result_df = result_df[result_df['vector_cmp_list'].apply(len) > 0]

        return result_df

    @staticmethod
    def build_tree(source_data, merged_df, tree_name, path = None):
        """
        Create a colored tree visualization based on year matches.

        Args:
            merged_df: DataFrame from merge_mappings_from_base with columns
                    ['description', 'vector_base', 'vector_cmp_list']
            total_only_df: DataFrame with parent-child relationships
                        with columns ['parent_vector', 'vector']
        """

        # Step 1: Create color and label mapping based on year matches
        color_map = {}
        node_labels = {}

        for _, row in merged_df.iterrows():
            vector = row['vector_base']
            description = row['description']
            matches = row['vector_cmp_list']

            # Extract years and actual column names from matches
            matched_info = []
            for match in matches:
                if 'v_CA16_' in match:
                    matched_info.append(('2016', match))
                elif 'v_CA11' in match:
                    matched_info.append(('2011', match))
                elif 'v_CA06_' in match:
                    matched_info.append(('2006', match))
            matched_info.append(('2021', vector))
            # Remove duplicates and sort by year
            matched_info = sorted(list(set(matched_info)),reverse=True)
            num_matches = len(matched_info) - 1

            # Determine color based on number of matches
            if num_matches == 0:
                color = 'white'
            elif num_matches == 1:
                color = 'salmon'
            elif num_matches == 2:
                color = 'yellow'
            elif num_matches >= 3:
                color = 'lightgreen'

            color_map[vector] = color

            # Create node label with description and matching column names
            if matched_info:
                matches_str = '\\n'.join([f"{year}: {col}" for year, col in matched_info])
            else:
                matches_str = '2021 only'

            # Truncate description if too long
            desc_short = description[:20] + '...' if len(description) > 20 else description
            node_labels[vector] = f"{desc_short}\\n{matches_str}"

        # Step 2: Build a mapping from parent to children (your original code)
        tree = defaultdict(list)

        for _, row in source_data.iterrows():
            parent = row['parent_vector']
            child = row['vector']
            tree[parent].append(child)

        # Step 3: Create the Graphviz diagram (enhanced version of your original)
        dot = Digraph()
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='box', style='filled')
        dot.attr(splines='ortho') 

        # First, add all nodes with colors and labels
        all_nodes = set()
        for parent, children in tree.items():
            if parent is not None:
                all_nodes.add(parent)
            for child in children:
                all_nodes.add(child)

        for node in all_nodes:
            color = color_map.get(node, 'lightgray')  # Default color for nodes not in merged_df
            label = node_labels.get(node, node)  # Use vector name if no custom label
            dot.node(node, label=label, fillcolor=color)

        # Then add edges (your original logic)
        for parent, children in tree.items():
            for child in children:
                if parent is not None:
                    dot.edge(parent, child)
                else:
                    dot.node(child)  # root nodes
        if path is not None:
            path = Path(path)
        else:
            path = Path.cwd()

        dot.render(tree_name, path, format="svg")
        return dot
