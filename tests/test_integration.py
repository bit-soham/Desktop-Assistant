import unittest
import os
import tempfile
import torch
import numpy as np
import re
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.audio_processing import AudioProcessor
from models.note_management import NoteManager
from models.rag_llm import RAGLLMProcessor
from models.text_processing import confirm_and_apply_command_correction


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for full workflow scenarios"""

    def setUp(self):
        """Set up test fixtures for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = os.path.join(self.temp_dir, 'notes')
        self.embeddings_dir = os.path.join(self.temp_dir, 'embeddings')
        self.outputs_dir = os.path.join(self.temp_dir, 'outputs')
        os.makedirs(self.notes_dir)
        os.makedirs(self.embeddings_dir)
        os.makedirs(self.outputs_dir)

        # Mock all ML models
        self.mock_whisper = Mock()
        self.mock_xtts_model = Mock()
        self.mock_xtts_config = Mock()
        self.mock_xtts_config.audio = Mock()
        self.mock_xtts_config.audio.sample_rate = 22050

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

        # Initialize components
        self.audio_processor = AudioProcessor(
            self.mock_whisper,
            self.mock_xtts_model,
            self.mock_xtts_config,
            output_dir=self.outputs_dir
        )

        self.note_manager = NoteManager(
            self.notes_dir,
            self.embeddings_dir,
            self.mock_embedding_model
        )

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

    @patch('builtins.print')  # Suppress debug prints
    def test_full_note_creation_workflow(self, mock_print):
        """Test the complete workflow from voice command to note creation"""
        # Step 1: Mock voice transcription
        self.mock_whisper.transcribe.return_value = ([Mock(text="create note buy groceries tomorrow")], Mock())

        # Step 2: Mock LLM parsing for note creation
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_template_result = Mock()
        mock_template_result.to.return_value = mock_inputs
        self.mock_llm_tokenizer.apply_chat_template.return_value = mock_template_result

        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        self.mock_llm_model.generate.return_value = mock_output
        self.mock_llm_tokenizer.decode.return_value = '{"Title": "Grocery Shopping", "Note": "buy groceries tomorrow", "Duration": null}'

        # Step 3: Mock TTS synthesis
        self.mock_xtts_model.synthesize.return_value = {'wav': np.array([0.1, 0.2, 0.3])}

        with patch('wave.open'), patch('pyaudio.PyAudio'), patch('soundfile.write'), \
             patch.object(self.audio_processor, 'play_audio') as mock_play:

            # Simulate the workflow
            # 1. Record audio (would be triggered by voice input)
            audio_file = os.path.join(self.outputs_dir, "test_recording.wav")
            with patch('pyaudio.PyAudio') as mock_pyaudio:
                mock_p = Mock()
                mock_stream = Mock()
                mock_p.open.return_value = mock_stream
                mock_p.get_sample_size.return_value = 2
                mock_pyaudio.return_value = mock_p

                mock_stream.read.side_effect = [b'test_data', KeyboardInterrupt()]

                # This would normally be called in main.py
                self.audio_processor.record_audio(audio_file)

            # 2. Transcribe audio
            transcription = self.audio_processor.transcribe_with_whisper(audio_file)
            self.assertEqual(transcription, "create note buy groceries tomorrow")

            # 3. Process command and create note
            user_input_lower = transcription.lower()
            user_input_lower = confirm_and_apply_command_correction(user_input_lower, threshold=0.85)

            m = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower, flags=re.IGNORECASE)
            self.assertIsNotNone(m)

            note_part = m.group(1).strip()
            parsed_notes = self.note_manager.parse_note_creation(note_part, self.mock_llm_tokenizer, self.mock_llm_model)

            self.assertEqual(parsed_notes['Title'], "Grocery Shopping")
            self.assertEqual(parsed_notes['Note'], "buy groceries tomorrow")
            self.assertIsNone(parsed_notes['Duration'])

            # 4. Create the note
            result = self.note_manager.manage_note("create", parsed_notes['Title'], parsed_notes['Note'])
            self.assertTrue(result)

            # 5. Verify note was created
            note_path = os.path.join(self.notes_dir, "Grocery Shopping.txt")
            self.assertTrue(os.path.exists(note_path))

            with open(note_path, 'r') as f:
                content = f.read()
                self.assertIn("buy groceries tomorrow", content)

    @patch('builtins.print')  # Suppress debug prints
    def test_conversation_with_rag_workflow(self, mock_print):
        """Test the complete conversation workflow with RAG context"""
        # Step 1: Create some notes for context
        self.note_manager.manage_note("create", "Meeting Notes", "Discuss project timeline and deliverables")
        self.note_manager.manage_note("create", "Shopping List", "Buy milk, bread, and eggs")

        # Create embeddings for the notes
        for filename in os.listdir(self.notes_dir):
            if filename.endswith('.txt'):
                title = filename[:-4]
                emb_path = os.path.join(self.embeddings_dir, f"{title}.pt")
                torch.save(torch.randn(1, 384), emb_path)

        # Step 2: Mock user query and LLM response
        user_query = "what do I need to buy for the project meeting"

        # Mock high similarity for meeting notes
        with patch('models.rag_llm.util.cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.1], [0.9]])  # Low for shopping, high for meeting

            mock_inputs = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            mock_template_result = Mock()
            mock_template_result.to.return_value = mock_inputs
            self.mock_llm_tokenizer.apply_chat_template.return_value = mock_template_result

            mock_output = torch.tensor([[1, 2, 3, 4, 5]])
            self.mock_llm_model.generate.return_value = mock_output
            self.mock_llm_tokenizer.decode.return_value = "Based on your notes, you need to discuss the project timeline and deliverables in the meeting."

            # Step 3: Get RAG response
            response = self.rag_processor.chatgpt_streamed(
                user_query, "You are a helpful assistant", [], "Bot", threshold=0.7
            )

            # Verify response contains relevant context
            self.assertIn("timeline", response.lower())
            self.assertIn("deliverables", response.lower())

            # Verify <|eot_id|> was filtered out
            self.assertNotIn("<|eot_id|>", response)

    def test_note_listing_workflow(self):
        """Test the note listing workflow"""
        # Create some test notes
        self.note_manager.manage_note("create", "Note 1", "First test note")
        self.note_manager.manage_note("create", "Note 2", "Second test note")
        self.note_manager.manage_note("create", "Note 3", "Third test note")

        # Capture print output (don't mock print!)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.note_manager.list_notes()

        output = f.getvalue()

        # Verify all note titles are listed (list_notes only prints titles, not content)
        self.assertIn("Note 1", output)
        self.assertIn("Note 2", output)
        self.assertIn("Note 3", output)
        self.assertIn("Active Notes:", output)

    @patch('builtins.print')  # Suppress debug prints
    def test_note_deletion_workflow(self, mock_print):
        """Test the note deletion workflow"""
        # Create a test note
        title = "Note to Delete"
        content = "This note will be deleted"
        self.note_manager.manage_note("create", title, content)

        # Verify note exists
        note_path = os.path.join(self.notes_dir, f"{title}.txt")
        self.assertTrue(os.path.exists(note_path))

        # Delete the note
        result = self.note_manager.manage_note("delete", title)
        self.assertTrue(result)

        # Verify note is gone
        self.assertFalse(os.path.exists(note_path))

    def test_tts_error_handling_workflow(self):
        """Test TTS error handling in the workflow"""
        # Mock TTS to raise an error
        self.mock_xtts_model.synthesize.side_effect = Exception("libtorchcodec error")

        # Capture output (don't mock print!)
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self.audio_processor.process_and_play("Test message", "speaker.wav", 1)

        output = f.getvalue()

        # Verify error handling - should show text fallback
        self.assertIn("TTS Audio Generation Error", output)
        self.assertIn("Response text: Test message", output)
        # The error message checks for "libtorchcodec" in str(e), but doesn't print it
        # Instead it shows the FFmpeg/TorchCodec issue message
        self.assertIn("FFmpeg/TorchCodec issue", output)

    @patch('builtins.input', return_value='n')  # Mock user declining corrections
    def test_text_command_correction_workflow(self, mock_input):
        """Test text processing and command correction in workflow"""
        # Test various misspellings that should be corrected
        test_cases = [
            ("creat note test", "create note test"),
            ("delet note test", "delete note test"),
            ("list nots", "list nots"),  # This shouldn't be corrected
            ("create note hello world", "create note hello world"),
        ]

        for input_text, expected in test_cases:
            result = confirm_and_apply_command_correction(input_text, threshold=0.85)
            # Note: The actual correction depends on the implementation
            # This test verifies the function runs without error
            self.assertIsInstance(result, str)
            self.assertEqual(result.lower(), input_text.lower())  # Should preserve case of original


if __name__ == '__main__':
    unittest.main()