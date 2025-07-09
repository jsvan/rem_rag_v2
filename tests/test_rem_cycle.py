"""
Test the REM (Random Episodic Memory) cycle implementation.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.core.rem_cycle import REMCycle, REMSample
from src.llm import LLMClient
from src.vector_store import REMVectorStore


class TestREMCycle:
    """Test suite for REM cycle functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client"""
        llm = Mock(spec=LLMClient)
        llm.complete = Mock(return_value="Test response")
        return llm
    
    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store"""
        store = Mock(spec=REMVectorStore)
        store.add_texts = Mock(return_value=["test-node-id"])
        return store
    
    @pytest.fixture
    def rem_cycle(self, mock_llm, mock_store):
        """Create a REM cycle instance with mocks"""
        return REMCycle(mock_llm, mock_store)
    
    def test_rem_sample_creation(self):
        """Test REMSample dataclass creation"""
        sample = REMSample(
            node_id="test-1",
            content="Test content",
            metadata={"year": 1962},
            timestamp=datetime(1962, 10, 1)
        )
        
        assert sample.node_id == "test-1"
        assert sample.content == "Test content"
        assert sample.metadata["year"] == 1962
        assert sample.timestamp.year == 1962
    
    def test_node_to_sample_conversion(self, rem_cycle):
        """Test converting a node dict to REMSample"""
        node = {
            "id": "node-1",
            "documents": ["This is test content"],
            "metadata": {
                "year": 1962,
                "timestamp": "1962-10-01T00:00:00",
                "article_title": "Test Article"
            }
        }
        
        sample = rem_cycle._node_to_sample(node)
        
        assert sample.node_id == "node-1"
        assert sample.content == "This is test content"
        assert sample.metadata["year"] == 1962
        assert sample.timestamp.year == 1962
    
    def test_find_implicit_question(self, rem_cycle):
        """Test finding implicit questions from samples"""
        samples = [
            REMSample("1", "Content about nuclear policy", {"year": 1962, "article_title": "Nuclear Strategy"}),
            REMSample("2", "Content about Soviet relations", {"year": 1963, "article_title": "USSR Today"}),
            REMSample("3", "Content about European unity", {"year": 1961, "article_title": "EEC Formation"})
        ]
        
        rem_cycle.llm.complete.return_value = "What role does nuclear deterrence play in shaping alliance structures?"
        
        question = rem_cycle._find_implicit_question(samples)
        
        assert "nuclear deterrence" in question
        assert rem_cycle.llm.complete.called
        
        # Check the prompt includes all passages
        call_args = rem_cycle.llm.complete.call_args[0][0]
        assert "Nuclear Strategy" in call_args
        assert "USSR Today" in call_args
        assert "EEC Formation" in call_args
    
    def test_generate_synthesis(self, rem_cycle):
        """Test synthesis generation"""
        samples = [
            REMSample("1", "Nuclear content", {"year": 1962}),
            REMSample("2", "Soviet content", {"year": 1963}),
            REMSample("3", "Europe content", {"year": 1961})
        ]
        
        question = "How do military alliances evolve?"
        expected_synthesis = "Military alliances evolve through shared threats..."
        rem_cycle.llm.complete.return_value = expected_synthesis
        
        synthesis = rem_cycle._generate_synthesis(samples, question)
        
        assert synthesis == expected_synthesis
        
        # Check prompt construction
        call_args = rem_cycle.llm.complete.call_args[0][0]
        assert question in call_args
        assert "Nuclear content" in call_args
        assert "1962" in call_args
    
    def test_store_rem_node(self, rem_cycle):
        """Test storing REM synthesis as a node"""
        samples = [
            REMSample("node-1", "Content 1", {"year": 1962}),
            REMSample("node-2", "Content 2", {"year": 1963}),
            REMSample("node-3", "Content 3", {"year": 1961})
        ]
        
        synthesis = "This is a test synthesis"
        question = "Test question?"
        
        node_id = rem_cycle._store_rem_node(synthesis, question, samples)
        
        assert node_id == "test-node-id"
        
        # Check store was called correctly
        rem_cycle.store.add_texts.assert_called_once()
        
        # Verify metadata
        call_args = rem_cycle.store.add_texts.call_args
        metadata = call_args[1]["metadatas"][0]
        
        assert metadata["node_type"] == "rem"
        assert metadata["implicit_question"] == question
        assert metadata["source_node_ids"] == ["node-1", "node-2", "node-3"]
        assert metadata["source_years"] == [1962, 1963, 1961]
        assert metadata["year_min"] == 1961
        assert metadata["year_max"] == 1963
    
    def test_sample_nodes_with_current_year(self, rem_cycle):
        """Test node sampling with current year preference"""
        # Mock nodes from different years
        all_nodes = [
            {"id": "1", "documents": ["Content 1"], "metadata": {"year": 1962}},
            {"id": "2", "documents": ["Content 2"], "metadata": {"year": 1962}},
            {"id": "3", "documents": ["Content 3"], "metadata": {"year": 1961}},
            {"id": "4", "documents": ["Content 4"], "metadata": {"year": 1963}},
            {"id": "5", "documents": ["Content 5"], "metadata": {"year": 1961}},
        ]
        
        rem_cycle.store.get_all_nodes.return_value = all_nodes
        
        # Sample with current year = 1962
        samples = rem_cycle._sample_nodes(current_year=1962)
        
        assert len(samples) == 3
        
        # At least one should be from 1962
        years = [s.metadata.get("year") for s in samples]
        assert 1962 in years
    
    def test_sample_nodes_insufficient_data(self, rem_cycle):
        """Test sampling when insufficient nodes available"""
        # Only 2 nodes available
        rem_cycle.store.get_all_nodes.return_value = [
            {"id": "1", "documents": ["Content 1"], "metadata": {}},
            {"id": "2", "documents": ["Content 2"], "metadata": {}}
        ]
        
        samples = rem_cycle._sample_nodes()
        
        assert len(samples) == 0  # Should return empty list
    
    def test_run_cycle_basic(self, rem_cycle):
        """Test running a basic REM cycle"""
        # Mock sufficient nodes for sampling
        mock_nodes = [
            {"id": f"node-{i}", "documents": [f"Content {i}"], "metadata": {"year": 1960 + i}}
            for i in range(10)
        ]
        rem_cycle.store.get_all_nodes.return_value = mock_nodes
        
        # Mock LLM responses
        rem_cycle.llm.complete.side_effect = [
            "What connects these events?",  # Question
            "They all show the evolution of Cold War dynamics."  # Synthesis
        ] * 5  # For 5 dreams
        
        # Run small cycle
        node_ids = rem_cycle.run_cycle(num_dreams=5, current_year=1962)
        
        assert len(node_ids) == 5
        assert all(nid == "test-node-id" for nid in node_ids)
        
        # Verify store was called 5 times
        assert rem_cycle.store.add_texts.call_count == 5
    
    def test_query_rem_insights(self, rem_cycle):
        """Test querying REM insights"""
        # Mock search results
        mock_results = [
            Mock(
                page_content="Question: How do alliances form?\n\nSynthesis: Alliances emerge from shared threats.",
                metadata={
                    "implicit_question": "How do alliances form?",
                    "source_years": [1961, 1962, 1963],
                    "score": 0.95
                }
            ),
            Mock(
                page_content="Question: What drives conflict?\n\nSynthesis: Conflicts arise from competing interests.",
                metadata={
                    "implicit_question": "What drives conflict?", 
                    "source_years": [1960, 1962],
                    "score": 0.87
                }
            )
        ]
        
        rem_cycle.store.similarity_search.return_value = mock_results
        
        insights = rem_cycle.query_rem_insights("alliance formation", top_k=2)
        
        assert len(insights) == 2
        assert insights[0]["question"] == "How do alliances form?"
        assert "shared threats" in insights[0]["synthesis"]
        assert insights[0]["source_years"] == [1961, 1962, 1963]
        
        # Verify search was called with REM filter
        rem_cycle.store.similarity_search.assert_called_once_with(
            query="alliance formation",
            k=2,
            where={"node_type": "rem"}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])