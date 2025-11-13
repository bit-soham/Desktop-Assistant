import unittest
import os
import tempfile
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.rag_llm import RAGLLMProcessor


class TestRAGLLMProcessor(unittest.TestCase):
    """Test cases for RAG LLM processing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = os.path.join(self.temp_dir, 'notes')
        self.embeddings_dir = os.path.join(self.temp_dir, 'embeddings')
        os.makedirs(self.notes_dir)
        os.makedirs(self.embeddings_dir)

        # Mock models
        self.mock_llm_tokenizer = Mock()
        self.mock_llm_model = Mock()
        self.mock_embedding_model = Mock()

        # Setup tokenizer mocks
        self.mock_llm_tokenizer.pad_token = None
        self.mock_llm_tokenizer.eos_token = "</s>"
        self.mock_llm_tokenizer.pad_token_id = 2
        self.mock_llm_tokenizer.eos_token_id = 2

        # Setup model mock
        self.mock_llm_model.device = 'cpu'

        # Setup embedding model mock
        self.mock_embedding_model.encode.return_value = np.random.rand(1, 384)

        self.rag_processor = RAGLLMProcessor(
            self.mock_llm_tokenizer,
            self.mock_llm_model,
            self.mock_embedding_model,
            self.notes_dir,
            self.embeddings_dir
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_relevant_context_no_notes(self):
        """Test context retrieval when no notes exist"""
        result = self.rag_processor.get_relevant_context("test query", threshold=0.7)
        self.assertEqual(result, [])

    def test_get_relevant_context_with_notes(self):
        """Test context retrieval with existing notes"""
        # Create a test note
        note_path = os.path.join(self.notes_dir, "test_note.txt")
        with open(note_path, 'w') as f:
            f.write("This is a test note about artificial intelligence and machine learning.")

        # Create corresponding embedding
        emb_path = os.path.join(self.embeddings_dir, "test_note.pt")
        test_embedding = torch.randn(1, 384)
        torch.save(test_embedding, emb_path)

        # Mock embedding similarity to return high similarity
        with patch('models.rag_llm.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.9]])  # High similarity

            result = self.rag_processor.get_relevant_context("AI and ML", threshold=0.7)

            self.assertEqual(len(result), 1)
            self.assertIn("artificial intelligence", result[0].lower())

    def test_get_relevant_context_below_threshold(self):
        """Test context retrieval when similarity is below threshold"""
        # Create a test note
        note_path = os.path.join(self.notes_dir, "test_note.txt")
        with open(note_path, 'w') as f:
            f.write("This is a test note.")

        # Create corresponding embedding
        emb_path = os.path.join(self.embeddings_dir, "test_note.pt")
        test_embedding = torch.randn(1, 384)
        torch.save(test_embedding, emb_path)

        # Mock embedding similarity to return low similarity
        with patch('models.rag_llm.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.5]])  # Low similarity

            result = self.rag_processor.get_relevant_context("unrelated query", threshold=0.7)

            self.assertEqual(result, [])

    def test_chatgpt_streamed_basic_response(self):
        """Test basic LLM response generation"""
        # Mock the tokenizer and model responses
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        # Create a mock that has a .to() method returning the dict
        mock_template_result = Mock()
        mock_template_result.to.return_value = mock_inputs
        self.mock_llm_tokenizer.apply_chat_template.return_value = mock_template_result

        # Mock model generate to return a simple response
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Longer than input
        self.mock_llm_model.generate.return_value = mock_output

        # Mock tokenizer decode to return clean response
        self.mock_llm_tokenizer.decode.return_value = "This is a test response from the LLM."

        result = self.rag_processor.chatgpt_streamed(
            "Hello", "You are a helpful assistant", [], "Bot", threshold=0.7
        )

        self.assertEqual(result, "This is a test response from the LLM.")
        self.mock_llm_model.generate.assert_called_once()

    def test_chatgpt_streamed_with_context(self):
        """Test LLM response with relevant context"""
        # Create a test note
        note_path = os.path.join(self.notes_dir, "context_note.txt")
        with open(note_path, 'w') as f:
            f.write("Important context about the topic.")

        # Create corresponding embedding
        emb_path = os.path.join(self.embeddings_dir, "context_note.pt")
        test_embedding = torch.randn(1, 384)
        torch.save(test_embedding, emb_path)

        # Mock high similarity
        with patch('models.rag_llm.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.9]])

            # Mock the tokenizer and model
            mock_inputs = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_template_result = Mock()
            mock_template_result.to.return_value = mock_inputs
            self.mock_llm_tokenizer.apply_chat_template.return_value = mock_template_result

            mock_output = torch.tensor([[1, 2, 3, 4, 5]])
            self.mock_llm_model.generate.return_value = mock_output
            self.mock_llm_tokenizer.decode.return_value = "Response with context."

            result = self.rag_processor.chatgpt_streamed(
                "topic query", "You are a helpful assistant", [], "Bot", threshold=0.7
            )

            # Verify context was included in the input
            call_args = self.mock_llm_tokenizer.apply_chat_template.call_args
            messages = call_args[0][0]  # First positional argument
            user_message = messages[-1]['content']  # Last message should be user with context
            self.assertIn("Important context", user_message)

    def test_chatgpt_streamed_eot_id_filtering(self):
        """Test that <|eot_id|> tokens are filtered from responses"""
        # Mock the tokenizer and model
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_template_result = Mock()
        mock_template_result.to.return_value = mock_inputs
        self.mock_llm_tokenizer.apply_chat_template.return_value = mock_template_result

        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_llm_model.generate.return_value = mock_output

        # Mock tokenizer decode to return response with <|eot_id|> token
        self.mock_llm_tokenizer.decode.return_value = "This is a response with <|eot_id|> token that should be removed."

        result = self.rag_processor.chatgpt_streamed(
            "test", "system prompt", [], "Bot", threshold=0.7
        )

        # Verify <|eot_id|> was removed
        self.assertNotIn("<|eot_id|>", result)
        self.assertEqual(result, "This is a response with  token that should be removed.")

    def test_load_notes_and_embeddings(self):
        """Test loading notes and embeddings"""
        # Create test notes
        note1_path = os.path.join(self.notes_dir, "note1.txt")
        note2_path = os.path.join(self.notes_dir, "note2.txt")

        with open(note1_path, 'w') as f:
            f.write("Content of note 1")

        with open(note2_path, 'w') as f:
            f.write("Content of note 2")

        # Create embeddings
        emb1_path = os.path.join(self.embeddings_dir, "note1.pt")
        emb2_path = os.path.join(self.embeddings_dir, "note2.pt")

        torch.save(torch.randn(1, 384), emb1_path)
        torch.save(torch.randn(1, 384), emb2_path)

        vault_content, vault_titles, vault_embeddings = self.rag_processor.load_notes_and_embeddings()

        self.assertEqual(len(vault_content), 2)
        self.assertEqual(len(vault_titles), 2)
        self.assertEqual(vault_embeddings.shape[0], 2)  # 2 notes

    def test_is_expired_no_metadata(self):
        """Test expiration check for note without metadata"""
        note_path = os.path.join(self.notes_dir, "regular_note.txt")
        with open(note_path, 'w') as f:
            f.write("Regular note content")

        result = self.rag_processor.is_expired(note_path)
        self.assertFalse(result)

    def test_is_expired_with_metadata_not_expired(self):
        """Test expiration check for note with metadata that hasn't expired"""
        note_path = os.path.join(self.notes_dir, "future_note.txt")
        import json
        from datetime import datetime, timedelta

        future_time = datetime.now() + timedelta(days=1)
        metadata = {"expiration": future_time.isoformat()}

        with open(note_path, 'w') as f:
            f.write(f'{json.dumps(metadata)}\nFuture note content')

        result = self.rag_processor.is_expired(note_path)
        self.assertFalse(result)

    def test_is_expired_with_metadata_expired(self):
        """Test expiration check for note with metadata that has expired"""
        note_path = os.path.join(self.notes_dir, "expired_note.txt")
        import json
        from datetime import datetime, timedelta

        past_time = datetime.now() - timedelta(days=1)
        metadata = {"expiration": past_time.isoformat()}

        with open(note_path, 'w') as f:
            f.write(f'{json.dumps(metadata)}\nExpired note content')

        result = self.rag_processor.is_expired(note_path)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()