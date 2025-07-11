"""
Embeddings loader for vector space analysis.
Extracts embeddings from ChromaDB and groups by node type.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingsLoader:
    """Load and organize embeddings from ChromaDB."""
    
    def __init__(self, vector_store):
        """
        Initialize with REMVectorStore instance.
        
        Args:
            vector_store: REMVectorStore instance
        """
        self.vector_store = vector_store
        self.collection = vector_store.collection
        
    def load_all_embeddings(self, limit: Optional[int] = None) -> Dict[str, Dict]:
        """
        Load all embeddings grouped by node_type.
        
        Args:
            limit: Optional limit on total documents to load
            
        Returns:
            Dict with structure:
            {
                'chunk': {
                    'embeddings': np.ndarray,  # (n_samples, embedding_dim)
                    'metadata': List[Dict],     # metadata for each embedding
                    'ids': List[str]           # document IDs
                },
                'synthesis': {...},
                ...
            }
        """
        print("Loading embeddings from ChromaDB...")
        
        # Get all documents with embeddings
        if limit:
            results = self.collection.get(
                limit=limit,
                include=['embeddings', 'metadatas', 'documents']
            )
        else:
            # Get count first
            total_count = self.collection.count()
            print(f"Total documents in collection: {total_count}")
            
            # Load in batches to handle large collections
            batch_size = 1000
            all_results = {
                'ids': [],
                'embeddings': [],
                'metadatas': [],
                'documents': []
            }
            
            for offset in tqdm(range(0, total_count, batch_size), desc="Loading batches"):
                batch = self.collection.get(
                    limit=min(batch_size, total_count - offset),
                    offset=offset,
                    include=['embeddings', 'metadatas', 'documents']
                )
                
                all_results['ids'].extend(batch['ids'])
                all_results['embeddings'].extend(batch['embeddings'])
                all_results['metadatas'].extend(batch['metadatas'])
                all_results['documents'].extend(batch['documents'])
            
            results = all_results
        
        # Group by node_type
        grouped_data = defaultdict(lambda: {
            'embeddings': [],
            'metadata': [],
            'ids': [],
            'documents': []
        })
        
        for i, metadata in enumerate(results['metadatas']):
            node_type = metadata.get('node_type', 'unknown')
            
            grouped_data[node_type]['embeddings'].append(results['embeddings'][i])
            grouped_data[node_type]['metadata'].append(metadata)
            grouped_data[node_type]['ids'].append(results['ids'][i])
            grouped_data[node_type]['documents'].append(results['documents'][i])
        
        # Convert embeddings lists to numpy arrays
        final_data = {}
        for node_type, data in grouped_data.items():
            if data['embeddings']:
                final_data[node_type] = {
                    'embeddings': np.array(data['embeddings']),
                    'metadata': data['metadata'],
                    'ids': data['ids'],
                    'documents': data['documents']
                }
                print(f"Loaded {len(data['embeddings'])} {node_type} embeddings")
        
        return final_data
    
    def get_embeddings_by_type(self, node_type: str) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """
        Get embeddings and metadata for specific node type.
        
        Args:
            node_type: Type of node to retrieve
            
        Returns:
            Tuple of (embeddings, metadata, ids)
        """
        results = self.collection.get(
            where={"node_type": node_type},
            include=['embeddings', 'metadatas']
        )
        
        if not results['embeddings']:
            return np.array([]), [], []
        
        embeddings = np.array(results['embeddings'])
        metadata = results['metadatas']
        ids = results['ids']
        
        print(f"Retrieved {len(embeddings)} {node_type} embeddings")
        return embeddings, metadata, ids
    
    def get_embeddings_by_year(self, year: int) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a specific year, grouped by node type.
        
        Args:
            year: Year to filter by
            
        Returns:
            Dict mapping node_type to embeddings array
        """
        results = self.collection.get(
            where={"year": year},
            include=['embeddings', 'metadatas']
        )
        
        # Group by node_type
        grouped = defaultdict(list)
        for i, metadata in enumerate(results['metadatas']):
            node_type = metadata.get('node_type', 'unknown')
            grouped[node_type].append(results['embeddings'][i])
        
        # Convert to numpy arrays
        return {
            node_type: np.array(embeddings) 
            for node_type, embeddings in grouped.items()
            if embeddings
        }
    
    def get_sample_embeddings(self, n_samples: int = 100, node_type: Optional[str] = None) -> Dict:
        """
        Get a random sample of embeddings for quick analysis.
        
        Args:
            n_samples: Number of samples per node type
            node_type: Optional specific node type to sample
            
        Returns:
            Dict with sampled embeddings by type
        """
        if node_type:
            # Sample specific type
            results = self.vector_store.sample(
                n=n_samples,
                filter={"node_type": node_type}
            )
            
            if results['documents']:
                embeddings = self.collection.get(
                    ids=results['ids'],
                    include=['embeddings']
                )['embeddings']
                
                return {
                    node_type: {
                        'embeddings': np.array(embeddings),
                        'metadata': results['metadatas'],
                        'ids': results['ids']
                    }
                }
            return {}
        
        # Sample all types
        node_types = ['chunk', 'summary', 'learning', 'synthesis', 'rem']
        sampled_data = {}
        
        for nt in node_types:
            results = self.vector_store.sample(
                n=n_samples,
                filter={"node_type": nt}
            )
            
            if results['documents']:
                embeddings = self.collection.get(
                    ids=results['ids'],
                    include=['embeddings']
                )['embeddings']
                
                sampled_data[nt] = {
                    'embeddings': np.array(embeddings),
                    'metadata': results['metadatas'],
                    'ids': results['ids']
                }
                print(f"Sampled {len(embeddings)} {nt} embeddings")
        
        return sampled_data
    
    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the embeddings.
        
        Returns:
            Dict with counts and basic stats
        """
        stats = self.vector_store.get_stats()
        
        # Add node type counts
        node_type_counts = {}
        for node_type in ['chunk', 'summary', 'learning', 'synthesis', 'rem']:
            count = len(self.collection.get(
                where={"node_type": node_type},
                limit=1,
                include=[]
            )['ids'])
            if count > 0:
                node_type_counts[node_type] = count
        
        stats['node_type_counts'] = node_type_counts
        return stats