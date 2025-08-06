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
from typing import Dict, List, Tuple, Set, Optional
# import sys
# import os

# # Add "src" to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .multithreaded_mapping import thread_match_descriptions_multithreaded
nltk.download('stopwords')


class VariableLinker:
    """
    A class for processing census metadata and creating tree visualizations.
    
    This class provides functionality for:
    - Preprocessing census metadata from JSON files
    - Computing similarity between census descriptions using various methods
    - Matching descriptions across different census years
    - Building hierarchical tree visualizations with color coding
    """
    
    @staticmethod
    def tree_preprocess_census_metadata(path, type_filter = "Total"):
        """
        Preprocess census metadata from a JSON file.
        
        Reads a JSON file containing census metadata, filters for specified type records targetting the type that represents the main skeleton of the tree,
        and restructures the data for further processing.
        
        Args:
            path (str): Path to the JSON file containing census metadata
            type_filter (str): Type of records to filter for (default: "Total")
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with columns ['vector', 'type', 'description', ...]
                         where 'vector' contains the original index values
        """
        df = pd.read_json(path).T
        filtered_data = df[df["type"] == type_filter]
        print("The unique values for the type were: ", pd.unique(df["type"]), "and now it is: ", pd.unique(filtered_data["type"]))
        filtered_data = filtered_data.reset_index()
        filtered_data = filtered_data.rename(columns={"index": "vector"})
        return filtered_data

    @staticmethod
    def jaccard_similarity(sentence1, sentence2):
        """
        Compute Jaccard similarity between two census descriptions.
        
        This function processes two census description strings and computes their
        Jaccard similarity based on token overlap.
        
        Args:
            sentence1 (str): First census description
            sentence2 (str): Second census description
            
        Returns:
            float: Jaccard similarity score between 0.0 and 1.0
        """
        tokens1 = VariableLinker.process_discription_text(sentence1)
        tokens2 = VariableLinker.process_discription_text(sentence2)
        
        if not tokens1 and not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    @staticmethod
    def process_discription_text(text):
        """
        Process and tokenize census text for similarity comparison.
        
        This function normalizes text, extracts meaningful tokens (words and numbers),
        removes stopwords, and preserves numeric ranges and values.
        
        Args:
            text (str): Raw census description text
            
        Returns:
            set: Set of processed tokens (words and numbers, excluding stopwords)
        """
        stop_words = set(stopwords.words('english'))
        # Normalize ranges first
        text = VariableLinker.normalize_ranges(text)
        
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
        """
        Normalize numeric ranges in text for consistent processing.
        
        Converts text like "80,000 to 100,000" to "80000-100000" format.
        
        Args:
            text (str): Text containing potential numeric ranges
            
        Returns:
            str: Text with normalized numeric ranges
        """
        # Convert "80,000 to 100,000" to "80000-100000"
        text = re.sub(r'(\d{1,3}(?:,\d{3})*)\s+to\s+(\d{1,3}(?:,\d{3})*)', 
                        lambda m: f"{m.group(1).replace(',', '')}-{m.group(2).replace(',', '')}", 
                        text)
        return text

    @staticmethod
    def tree_match_descriptions_jaccard(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9):
        """
        Match descriptions between two DataFrames using Jaccard similarity.
        
        This function performs a two-pass matching approach:
        1. First pass: Exact description matches
        2. Second pass: Jaccard similarity matches for unmatched descriptions
        
        Args:
            source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
            compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description']
            similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                         containing the mapping between source and comparison vectors
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
                similarity_score = VariableLinker.jaccard_similarity(source_description, compare_description)
                if similarity_score >= similarity_threshold and similarity_score > best_similarity:
                    best_similarity = similarity_score
                    best_match_idx = compare_idx
                    best_match_vector = compare_row['vector']
            if best_match_idx is not None:
                record['vector_cmp'] = best_match_vector
                matched_indices.add(best_match_idx)
                

        return pd.DataFrame(mapping_records)

    @staticmethod
    def tree_match_descriptions_transformer(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
        """
        Match descriptions between two DataFrames using sentence transformers.
        
        This function uses pre-trained sentence transformers to compute semantic similarity
        between census descriptions. 
        
        Performs a two-pass matching approach:
        1. First pass: Exact description matches
        2. Second pass: Sentence transformer similarity matches for unmatched descriptions
        
        Args:
            source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
            compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description']
            similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
            model_name (str): Name of the sentence transformer model to use (default: 'all-mpnet-base-v2')
            
        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                         containing the mapping between source and comparison vectors
        """
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
    def tree_match_descriptions_details_sentence_transformer( source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
        """
        Match descriptions using sentence transformers with optimized pre-encoding.
        
        Optimized version of match_descriptions_transformer that pre-encodes all descriptions
        at once for better performance. Performs exact matching on 'description' column first,
        then uses the 'details' column to find the best match when multiple exact matches exist.
        
        Args:
            source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description', 'details']
            compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description', 'details']
            similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
            model_name (str): Name of the sentence transformer model to use (default: 'all-mpnet-base-v2')
            
        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                         containing the mapping between source and comparison vectors
        """
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
    def tree_match_descriptions_multithreaded(
        source_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        similarity_threshold: float = 0.9,
        max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Multithreaded version of map_descriptions with enhanced similarity matching.
        
        Performs a two-pass matching approach with multithreaded similarity processing:
        1. First pass: Exact description matches (single-threaded)
        2. Second pass: Similarity matches for unmatched descriptions (multithreaded)
        
        Changes in behavior compared to single-threaded version:
        1. Uses mutual exclusion to ensure thread-safe mapping
        2. Processes similarity matching in parallel
        
        Args:
            source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
            compare_df (pd.DataFrame): Compare DataFrame with columns ['vector', 'description']
            similarity_threshold (float): Minimum similarity threshold (default: 0.9)
            max_workers (int): Maximum number of worker threads (default: 4)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                        containing the mapping between source and comparison vectors
        """
        return thread_match_descriptions_multithreaded(source_df= source_df, 
                                                       compare_df=compare_df, 
                                                       similarity_threshold=similarity_threshold, 
                                                       max_workers=max_workers)


    @staticmethod
    def tree_merge_mappings(map_descriptions, *mappings_dfs):
        """
        Merge multiple mapping DataFrames into a single consolidated mapping.
        
        Takes a base DataFrame (typically 2021 data, or the latest year) and multiple mapping DataFrames,
        then consolidates all matching vectors for each description into a single list.
        
        Args:
            map_descriptions (pd.DataFrame): Base DataFrame with columns ['description', 'vector']
            *mappings_dfs: Variable number of mapping DataFrames with columns 
                          ['description', 'vector_base', 'vector_cmp']
            
        Returns:
            pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp_list']
                         where vector_cmp_list contains all matching vectors from all mappings
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
    def tree_build_tree(source_data, merged_df, tree_name, path = None):
        """

        
        Creates a hierarchical tree visualization using Graphviz, where nodes
        are colored based on how many census years they appear in. The tree structure is
        determined by parent-child relationships in the source_data.
        
        Color coding:
        - Gray: Source year only (no matches in other years)
        - Salmon: Matches in 1 other year
        - Yellow: Matches in 2 other years  
        - Light green: Matches in 3+ other years
        
        Args:
            source_data (pd.DataFrame): DataFrame with parent-child relationships
                                      containing columns ['parent_vector', 'vector']
            merged_df (pd.DataFrame): DataFrame from merge_mappings with columns
                                    ['description', 'vector_base', 'vector_cmp_list']
            tree_name (str): Name for the output tree file (without extension)
            path (str, optional): Directory path for saving the tree file (default: current directory)
            
        Returns:
            Digraph: Graphviz Digraph object representing the tree visualization
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

    @staticmethod
    def tree_parse_tree_to_dict(filepath):
        """
        Parse a Graphviz tree file into a dictionary.
        
        This function reads a Graphviz tree file and extracts node information including
        descriptions and year-specific vector mappings. It parses the node labels which
        contain multi-line information about each node.
        
        Args:
            filepath (str): Path to the Graphviz tree file
            
        Returns:
            Dict: Dictionary mapping node IDs to their information
        """
        # Step 1: Initialize empty dictionary to store parsed node information
        tree_dict = {}
        node_pattern = re.compile(
            r'\s*(\w+)\s+\[label=("([^"]+)"|([^\s]+))\s+fillcolor=([^\]]+)\]'
        )

        # Step 3: Read and parse each line of the tree file
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                # Try to match the current line against our regex pattern
                match = node_pattern.match(line.strip())
                if match:
                    # Step 4: Extract matched groups from regex
                    groups = match.groups()
                    node_id = groups[0]  # First group is always the node ID
                    fillcolor = groups[-1]  # Last group is always the fillcolor
                    
                    # Step 5: Handle both quoted and unquoted labels
                    # groups[2] contains quoted label content (if present)
                    # groups[3] contains unquoted label content (if present)
                    if groups[2] is not None:  # Quoted label found
                        label = groups[2]  # Use quoted content
                    else:  # Unquoted label found
                        label = groups[3]  # Use unquoted content
                    
                    # Step 6: Parse the label content
                    # Split the label into lines (handles multi-line labels)
                    label_lines = label.split('\\n')
                    description = label_lines[0]  # First line is always the description
                    
                    # Step 7: Extract year mappings from remaining lines
                    year_map = {}
                    for entry in label_lines[1:]:  # Skip first line (description)
                        # Look for year:vector patterns like "2021: v_CA21_4728"
                        if ':' in entry:
                            year, val = entry.split(':', 1)  # Split on first colon
                            year_map[year.strip()] = val.strip()  # Store year->vector mapping
                    
                    # Step 8: Store parsed information in tree dictionary
                    tree_dict[node_id] = {
                        "description": description,  # Human-readable description
                        **year_map,                 # Year-specific vector mappings
                        "fillcolor": fillcolor     # Color for visualization
                    }
        
        # Return the complete parsed tree dictionary
        return tree_dict

    @staticmethod
    def tree_extract_parent_child_relationships(filepath: str) -> Dict[str, List[str]]:
        """
        Extract parent-child relationships from tree file edges.
        
        This function reads a Graphviz tree file and extracts all parent-child relationships
        defined by edges (lines containing " -> "). It creates a mapping where each parent
        node is associated with a list of its child nodes.
        
        Args:
            filepath (str): Path to the tree file (Graphviz format)
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping parent nodes to their children
        """
        # Initialize a defaultdict to store parent->children mappings
        # defaultdict(list) automatically creates an empty list for new keys
        parent_child_relationships = defaultdict(list)
        
        # Open the tree file for reading with UTF-8 encoding
        with open(filepath, encoding='utf-8') as f:
            # Read the file line by line
            for line in f:
                # Remove leading/trailing whitespace from the line
                line = line.strip()
                
                # Check if this line contains an edge definition (parent -> child)
                # Graphviz edge syntax uses " -> " to connect nodes
                if ' -> ' in line:
                    # Split the line at " -> " to separate parent and child
                    parts = line.split(' -> ')
                    
                    # Ensure we have exactly 2 parts (parent and child)
                    if len(parts) == 2:
                        # Extract parent node ID and remove any whitespace
                        parent = parts[0].strip()
                        # Extract child node ID and remove any whitespace
                        child = parts[1].strip()
                        
                        # Add the child to the parent's list of children
                        # defaultdict automatically creates a list if parent doesn't exist
                        parent_child_relationships[parent].append(child)
        
        # Convert defaultdict back to regular dict and return
        return dict(parent_child_relationships)

    @staticmethod
    def tree_predict_parent_nodes(tree_dict: Dict, parent_child_relationships: Dict[str, List[str]], 
                            target_years: List[str] = ['2016', '2011', '2006']) -> Dict[str, List[str]]:
        """
        Predict parent nodes in other years using the additive property.
        
        This function implements the core prediction algorithm. It identifies parent nodes
        that exist in some years but not others, and determines if they can be predicted
        in missing years by checking if all their children exist in those years.
        
        The additive property means: Parent_Value = Sum(Child_Values)
        If all children exist in a target year, we can predict the parent for that year.
        
        Args:
            tree_dict (Dict): Parsed tree dictionary with node info and year mappings
            parent_child_relationships (Dict[str, List[str]]): Parent to children mapping
            target_years (List[str]): Years to predict parents for (default: ['2016', '2011', '2006'])
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping parent nodes to years they can be predicted in
        """
        # Initialize empty dictionary to store predictions
        # Key: parent node ID, Value: list of years where parent can be predicted
        predictions = {}
        
        # Iterate through each parent node and its children
        for parent_node, children in parent_child_relationships.items():
            # Step 0: Validate that parent exists in tree dictionary
            # Skip if parent node doesn't exist in the tree dictionary
            # This could happen if the tree file is incomplete or corrupted
            if parent_node not in tree_dict:
                continue
                
            # Get the parent node's information (description, year mappings, etc.)
            parent_info = tree_dict[parent_node]
            
            # Step 1: Find which years the parent currently exists in
            # Initialize empty set to store years where parent exists
            parent_years = set()
            
            # Check each key-value pair in parent_info
            for key, value in parent_info.items():
                # Only consider numeric keys (years) - missing years simply don't have keys
                # This identifies years where the parent node actually exists
                if key.isdigit():
                    parent_years.add(key)
            
            # Step 2: Check each target year to see if parent can be predicted
            # Initialize empty list to store years where parent can be predicted
            predictable_years = []
            
            # Check each target year (2016, 2011, 2006)
            for target_year in target_years:
                # Skip if parent already exists in this target year
                # No need to predict something that already exists
                if target_year in parent_years:
                    continue
                    
                # Step 3: Check if ALL children exist in the target year
                # Start with assumption that we can predict (optimistic approach)
                can_predict = True
                
                # Check each child of this parent
                for child_node in children:
                    # Skip if child node doesn't exist in tree dictionary
                    if child_node not in tree_dict:
                        can_predict = False
                        break
                        
                    # Get child node's information
                    child_info = tree_dict[child_node]
                    
                    # Check if child exists in target year
                    # If child doesn't exist in target year, we cannot predict parent
                    if target_year not in child_info:
                        can_predict = False
                        break
                
                # If all children exist in target year, add it to predictable years
                if can_predict:
                    predictable_years.append(target_year)
            
            # Step 4: Add parent to predictions if it can be predicted in at least one year
            if predictable_years:
                predictions[parent_node] = predictable_years
        
        # Return the final predictions dictionary
        return predictions

