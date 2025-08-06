# Import required libraries for multithreaded processing and data manipulation
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
import time
from typing import List, Tuple, Dict, Set
import logging

# Set up logging for debugging and monitoring multithreaded operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import VariableLinker to use its utility functions
# Note: This import is moved to function level to avoid circular imports

class ThreadSafeMappingManager:
    """
    Thread-safe manager for handling mapping operations with mutual exclusion.
    
    This class provides thread-safe operations for managing mapping results
    and used indices during multithreaded similarity matching.
    
    Attributes:
        compare_data (pd.DataFrame): Comparison DataFrame for matching
        used_indices (Set[int]): Set of indices that have been matched
        lock (threading.Lock): Lock for thread-safe operations
        mapping_results (List[Dict]): List to store mapping results
        lock_results (threading.Lock): Lock for thread-safe result storage
    """
    
    def __init__(self, compare_data: pd.DataFrame):
        """
        Initialize the thread-safe mapping manager.
        
        Args:
            compare_data (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description']
        """
        self.compare_data = compare_data
        self.used_indices: Set[int] = set()
        self.lock = threading.Lock()
        self.mapping_results: List[Dict] = []
        self.lock_results = threading.Lock()
    
    def find_best_available_match(self, similarity_candidates: List[Tuple[float, int, str]]) -> Tuple[bool, str | None]:
        """
        Thread-safe method to find the best available match from similarity candidates.
        
        Uses mutual exclusion to ensure thread-safe selection of matches.
        
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
        """
        Thread-safe method to add mapping result.
        
        Args:
            mapping_record (Dict): Mapping record to add
        """
        with self.lock_results:
            self.mapping_results.append(mapping_record)


def find_similarity_candidates(source_description: str, 
                             compare_data: pd.DataFrame, 
                             similarity_threshold: float) -> List[Tuple[float, int, str]]:
    """
    Find all similarity candidates above threshold for a given source description.
    
    Evaluates all descriptions in compare_data against the source description
    and returns candidates that meet the similarity threshold, sorted by score.
    
    Args:
        source_description (str): Source description to match against
        compare_data (pd.DataFrame): DataFrame to search for matches
        similarity_threshold (float): Minimum similarity threshold
        
    Returns:
        List of (similarity_score, compare_idx, compare_vector) sorted by score descending
    """
    # Import here to avoid circular import
    from linking.variable_linker import VariableLinker
    
    candidates = []
    
    for compare_idx, compare_row in compare_data.iterrows():
        compare_description = compare_row['description']
        similarity_score = VariableLinker.jaccard_similarity(source_description, compare_description)
        
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
    
    This function is designed to be called by worker threads to process
    similarity matching for individual records in parallel.
    
    Args:
        record (Dict): Mapping record with source description and vector
        compare_data (pd.DataFrame): DataFrame to match against
        mapping_manager (ThreadSafeMappingManager): Thread-safe manager for mapping operations
        similarity_threshold (float): Minimum similarity threshold
        
    Returns:
        Dict: Updated mapping record with vector_cmp field populated if match found
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


def thread_match_descriptions_multithreaded(
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
