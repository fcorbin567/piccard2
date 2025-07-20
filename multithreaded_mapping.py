import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import time
from typing import List, Tuple, Dict, Set
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from nltk.corpus import stopwords
import re
from collections import Counter

class ThreadSafeMappingManager:
    """Thread-safe manager for handling mapping operations with mutual exclusion"""
    
    def __init__(self, compare_data: pd.DataFrame):
        self.compare_data = compare_data
        self.used_indices: Set[int] = set()
        self.lock = threading.Lock()
        self.mapping_results: List[Dict] = []
        self.lock_results = threading.Lock()
    
    def find_best_available_match(self, similarity_candidates: List[Tuple[float, int, str]]) -> Tuple[bool, str | None]:
        """
        Thread-safe method to find the best available match from similarity candidates.
        
        Args:
            similarity_candidates: List of (similarity_score, compare_idx, compare_vector) sorted by score desc
            
        Returns:
            Tuple of (success, matched_vector) where success is True if match found
        """
        with self.lock:  # Critical section - mutual exclusion
            for similarity_score, compare_idx, compare_vector in similarity_candidates:
                if compare_idx not in self.used_indices:
                    # Found available match
                    self.used_indices.add(compare_idx)
                    return True, compare_vector
            
            # No available matches found
            return False, None
    
    def add_mapping_result(self, mapping_record: Dict):
        """Thread-safe method to add mapping result"""
        with self.lock_results:
            self.mapping_results.append(mapping_record)


def census_similarity(sentence1, sentence2):    
    tokens1 = process_census_text(sentence1)
    tokens2 = process_census_text(sentence2)
    
    if not tokens1 and not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def process_census_text(text):
        stop_words = set(stopwords.words('english'))
        # Normalize ranges first
        text = normalize_ranges(text)
        
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

def normalize_ranges(text):
    # Convert "80,000 to 100,000" to "80000-100000"
    text = re.sub(r'(\d{1,3}(?:,\d{3})*)\s+to\s+(\d{1,3}(?:,\d{3})*)', 
                  lambda m: f"{m.group(1).replace(',', '')}-{m.group(2).replace(',', '')}", 
                  text)
    return text




def find_similarity_candidates(source_description: str, 
                             compare_data: pd.DataFrame, 
                             similarity_threshold: float) -> List[Tuple[float, int, str]]:
    """
    Find all similarity candidates above threshold for a given source description.
    
    Returns:
        List of (similarity_score, compare_idx, compare_vector) sorted by score descending
    """
    candidates = []
    
    for compare_idx, compare_row in compare_data.iterrows():
        compare_description = compare_row['description']
        similarity_score = census_similarity(source_description, compare_description)
        
        if similarity_score >= similarity_threshold:
            candidates.append((similarity_score, compare_idx, compare_row['vector']))
    
    # Sort by similarity score (highest first) and return
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def process_similarity_mapping(record: Dict, 
                             compare_data: pd.DataFrame, 
                             mapping_manager: ThreadSafeMappingManager,
                             similarity_threshold: float) -> Dict:
    """
    Process similarity mapping for a single record (thread worker function).
    
    Args:
        record: Mapping record with source description and vector
        compare_data: DataFrame to match against
        mapping_manager: Thread-safe manager for mapping operations
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Updated mapping record
    """
    source_description = record['description']
    
    # Step 1: Find all similarity candidates above threshold
    similarity_candidates = find_similarity_candidates(
        source_description, compare_data, similarity_threshold
    )
    
    if not similarity_candidates:
        # No candidates found, keep as None
        return record
    
    # Step 2: Use mutual exclusion to find best available match
    success, matched_vector = mapping_manager.find_best_available_match(similarity_candidates)
    
    if success:
        record['vector_cmp'] = matched_vector
        # logger.info(f"Matched: {record['vector_base']} -> {matched_vector} (score: {similarity_candidates[0][0]:.3f})")
    # else:
        # logger.info(f"No available match for: {record['vector_base']} (all candidates used)")
    
    return record


def match_descriptions_multithreaded(
    source_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    similarity_threshold: float = 0.9,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Multithreaded version of map_descriptions with enhanced similarity matching.
    
    Changes in behavior:
    1. Finds ALL similar descriptions above threshold (not just first)
    2. Sorts by similarity score (highest first)
    3. Uses mutual exclusion to ensure thread-safe mapping
    4. Processes similarity matching in parallel
    
    Args:
        source_df: Source DataFrame with columns ['vector', 'description']
        compare_df: Compare DataFrame with columns ['vector', 'description']
        similarity_threshold: Minimum similarity threshold
        max_workers: Maximum number of worker threads
        
    Returns:
        DataFrame with columns [description, vector_base, vector_cmp]
    """
    
    source_data = source_df.copy()
    compare_data = compare_df.copy()
    
    # Step 1: Handle exact matches first (single-threaded for simplicity)
    mapping_records = []
    matched_indices = set()
    
    # Pre-create exact match lookup dictionary
    compare_desc_to_info = {}
    for compare_idx, compare_row in compare_data.iterrows():
        desc = compare_row['description']
        if desc not in compare_desc_to_info:
            compare_desc_to_info[desc] = []
        compare_desc_to_info[desc].append((compare_idx, compare_row['vector']))
    time_start = time.time()
    # Process exact matches
    for source_idx, source_row in source_data.iterrows():
        source_description, source_vector = source_row['description'], source_row['vector']
        
        if source_description in compare_desc_to_info:
            exact_matches = compare_desc_to_info[source_description]
            matched = False
            
            for compare_idx, compare_vector in exact_matches:
                if compare_idx not in matched_indices:
                    mapping_records.append({
                        'description': source_description,
                        'vector_base': source_vector,
                        'vector_cmp': compare_vector
                    })
                    matched_indices.add(compare_idx)
                    matched = True
                    break
            
            if matched:
                continue
        
        # Mark for similarity matching
        mapping_records.append({
            'description': source_description,
            'vector_base': source_vector,
            'vector_cmp': None
        })
    time_end = time.time()
    logger.info(f"Exact matching completed in {time_end - time_start:.2f} seconds")
    # Step 2: Multithreaded similarity matching
    # Pre-filter unmatched compare data
    unmatched_compare_data = compare_data[~compare_data.index.isin(matched_indices)]
    
    # Create thread-safe mapping manager
    mapping_manager = ThreadSafeMappingManager(pd.DataFrame(unmatched_compare_data))
    
    # Filter records that need similarity matching
    similarity_records = [record for record in mapping_records if record['vector_cmp'] is None]
    
    logger.info(f"Processing {len(similarity_records)} records with similarity matching using {max_workers} threads")
    
    # Process similarity matching in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all similarity matching tasks
        future_to_record = {
            executor.submit(
                process_similarity_mapping, 
                record, 
                pd.DataFrame(unmatched_compare_data), 
                mapping_manager, 
                similarity_threshold
            ): record for record in similarity_records
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_record):
            try:
                updated_record = future.result()
                # Update the original record in mapping_records
                for i, record in enumerate(mapping_records):
                    if record['vector_base'] == updated_record['vector_base']:
                        mapping_records[i] = updated_record
                        break
            except Exception as exc:
                logger.error(f"Error processing record: {exc}")
    
    end_time = time.time()
    logger.info(f"Similarity matching completed in {end_time - start_time:.2f} seconds")
    
    return pd.DataFrame(mapping_records)
