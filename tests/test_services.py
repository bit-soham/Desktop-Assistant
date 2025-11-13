import unittest
import os
import tempfile
import wave
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import io

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.audio_processing import AudioProcessor
from models.note_management import NoteManager
from models.rag_llm import RAGLLMProcessor
from models.text_processing import confirm_and_apply_command_correction


class TestAudioProcessing(unittest.TestCase):
    """Test cases for audio processing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the models since we don't want to load actual ML models in tests
        self.mock_whisper = Mock()
        self.mock_xtts_model = Mock()
        self.mock_xtts_config = Mock()
        self.mock_xtts_config.audio = Mock()
        self.mock_xtts_config.audio.sample_rate = 22050

        # Create AudioProcessor instance
        self.audio_processor = AudioProcessor(
            self.mock_whisper,
            self.mock_xtts_model,
            self.mock_xtts_config,
            output_dir=tempfile.mkdtemp()
        )

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up any test files
        import shutil
        if os.path.exists(self.audio_processor.output_dir):
            shutil.rmtree(self.audio_processor.output_dir)

    @patch('pyaudio.PyAudio')
    @patch('wave.open')
    def test_play_audio_success(self, mock_wave_open, mock_pyaudio):
        """Test successful audio playback"""
        # Setup mocks
        mock_wf = Mock()
        mock_wf.getsampwidth.return_value = 2
        mock_wf.getnchannels.return_value = 1
        mock_wf.getframerate.return_value = 44100
        mock_wf.readframes.side_effect = [b'data', b'']  # First call returns data, second returns empty

        mock_wave_open.return_value = mock_wf

        mock_p = Mock()
        mock_stream = Mock()
        mock_p.open.return_value = mock_stream
        mock_p.get_format_from_width.return_value = 8  # paInt16 format
        mock_pyaudio.return_value = mock_p

        # Test
        test_file = "test.wav"
        self.audio_processor.play_audio(test_file, output_device_index=1)

        # Verify
        mock_wave_open.assert_called_once_with(test_file, 'rb')
        mock_p.open.assert_called_once()
        mock_stream.write.assert_called_once_with(b'data')
        mock_stream.close.assert_called_once()
        mock_p.terminate.assert_called_once()

    @patch('pyaudio.PyAudio')
    @patch('wave.open')
    @patch('builtins.print')  # Suppress debug prints
    def test_record_audio_success(self, mock_print, mock_wave_open, mock_pyaudio):
        """Test successful audio recording"""
        # Setup mocks
        mock_p = Mock()
        mock_stream = Mock()
        mock_p.open.return_value = mock_stream
        mock_p.get_sample_size.return_value = 2
        mock_pyaudio.return_value = mock_p

        mock_wf = Mock()
        mock_wave_open.return_value = mock_wf

        # Mock KeyboardInterrupt to stop recording
        mock_stream.read.side_effect = [b'test_data', KeyboardInterrupt()]

        # Test
        test_file = os.path.join(self.audio_processor.output_dir, "test_record.wav")
        self.audio_processor.record_audio(test_file, input_device_index=1)

        # Verify
        mock_wave_open.assert_called_once_with(test_file, 'wb')
        mock_wf.setnchannels.assert_called_once_with(1)
        mock_wf.setsampwidth.assert_called_once_with(2)
        mock_wf.setframerate.assert_called_once_with(16000)
        mock_wf.writeframes.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_p.terminate.assert_called_once()

    def test_transcribe_with_whisper_success(self):
        """Test successful audio transcription"""
        # Setup mock
        mock_segment1 = Mock()
        mock_segment1.text = "Hello"
        mock_segment2 = Mock()
        mock_segment2.text = "world"
        mock_info = Mock()

        self.mock_whisper.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)

        # Test
        result = self.audio_processor.transcribe_with_whisper("test.wav")

        # Verify
        self.assertEqual(result, "Hello world")
        self.mock_whisper.transcribe.assert_called_once_with("test.wav", beam_size=5)

    def test_transcribe_with_whisper_empty(self):
        """Test transcription with no segments"""
        # Setup mock
        mock_info = Mock()
        self.mock_whisper.transcribe.return_value = ([], mock_info)

        # Test
        result = self.audio_processor.transcribe_with_whisper("test.wav")

        # Verify
        self.assertEqual(result, "")

    @patch('soundfile.write')
    @patch('os.path.exists')
    def test_process_and_play_short_text(self, mock_exists, mock_sf_write):
        """Test speech synthesis and playback with short text"""
        # Setup mocks
        mock_exists.return_value = True
        self.mock_xtts_model.synthesize.return_value = {'wav': np.array([0.1, 0.2, 0.3])}

        # Test
        test_prompt = "Hello world"
        speaker_file = "speaker.wav"

        with patch.object(self.audio_processor, 'play_audio') as mock_play:
            self.audio_processor.process_and_play(test_prompt, speaker_file, 1)

        # Verify
        self.mock_xtts_model.synthesize.assert_called_once()
        mock_sf_write.assert_called_once()
        mock_play.assert_called_once()

    @patch('soundfile.write')
    @patch('os.path.exists')
    def test_process_and_play_long_text_chunking(self, mock_exists, mock_sf_write):
        """Test speech synthesis with text chunking for long text"""
        # Setup mocks
        mock_exists.return_value = True
        self.mock_xtts_model.synthesize.return_value = {'wav': np.array([0.1, 0.2, 0.3])}

        # Test with long text that should be chunked
        long_text = "This is a very long text that should be split into multiple chunks because it exceeds the maximum length limit for the TTS system. " * 10
        speaker_file = "speaker.wav"

        with patch.object(self.audio_processor, 'play_audio') as mock_play:
            self.audio_processor.process_and_play(long_text, speaker_file, 1)

        # Verify that synthesize was called multiple times (chunking occurred)
        self.assertGreater(self.mock_xtts_model.synthesize.call_count, 1)
        mock_sf_write.assert_called_once()
        mock_play.assert_called_once()

    @patch('soundfile.write')
    @patch('os.path.exists')
    def test_process_and_play_xtts_error_handling(self, mock_exists, mock_sf_write):
        """Test error handling in speech synthesis"""
        # Setup mocks to raise an exception
        mock_exists.return_value = True
        self.mock_xtts_model.synthesize.side_effect = Exception("libtorchcodec error")

        # Capture stdout to verify error message
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            self.audio_processor.process_and_play("Test prompt", "speaker.wav", 1)

        # Verify error handling - should not crash and should show error message
        output = captured_output.getvalue()
        self.assertIn("TTS Audio Generation Error", output)
        self.assertIn("Response text: Test prompt", output)


class TestNoteManagement(unittest.TestCase):
    """Test cases for note management functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = os.path.join(self.temp_dir, 'notes')
        self.embeddings_dir = os.path.join(self.temp_dir, 'embeddings')
        os.makedirs(self.notes_dir)
        os.makedirs(self.embeddings_dir)

        # Mock embedding model
        self.mock_embedding_model = Mock()
        self.mock_embedding_model.encode.return_value = np.random.rand(1, 384)  # Mock embedding

        self.note_manager = NoteManager(self.notes_dir, self.embeddings_dir, self.mock_embedding_model)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_note_basic(self):
        """Test basic note creation"""
        title = "Test Note"
        content = "This is a test note"
        duration = 3600  # 1 hour

        result = self.note_manager.manage_note("create", title, content, seconds=duration)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.notes_dir, f"{title}.txt")))

    def test_create_note_no_duration(self):
        """Test note creation without duration"""
        title = "Test Note No Duration"
        content = "This is a test note without duration"

        result = self.note_manager.manage_note("create", title, content)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.notes_dir, f"{title}.txt")))

    def test_delete_note_existing(self):
        """Test deleting an existing note"""
        # First create a note
        title = "Note to Delete"
        content = "This note will be deleted"
        self.note_manager.manage_note("create", title, content)

        # Now delete it
        result = self.note_manager.manage_note("delete", title)

        self.assertTrue(result)
        self.assertFalse(os.path.exists(os.path.join(self.notes_dir, f"{title}.txt")))

    def test_delete_note_nonexistent(self):
        """Test deleting a non-existent note"""
        result = self.note_manager.manage_note("delete", "NonExistentNote")

        self.assertFalse(result)

    def test_list_notes_empty(self):
        """Test listing notes when no notes exist"""
        with patch('builtins.print') as mock_print:
            self.note_manager.list_notes()

        # Should print that no notes exist
        mock_print.assert_called()

    def test_list_notes_with_content(self):
        """Test listing notes with existing notes"""
        # Create some test notes
        self.note_manager.manage_note("create", "Note1", "Content 1")
        self.note_manager.manage_note("create", "Note2", "Content 2")

        with patch('builtins.print') as mock_print:
            self.note_manager.list_notes()

        # Should print note information
        self.assertGreater(mock_print.call_count, 0)

    @patch('json.loads')
    def test_parse_note_creation_success(self, mock_json_loads):
        """Test successful note parsing"""
        # Mock the LLM components
        mock_tokenizer = Mock()
        mock_model = Mock()

        # Mock tokenizer methods
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 2
        
        # Create a mock inputs object that behaves like a dict
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_tokenizer.apply_chat_template.return_value = Mock()  # Return a mock tensor
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        
        mock_tokenizer.decode.return_value = '{"Title": "Test", "Note": "Content", "Duration": 3600}'

        # Mock model generate
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Mock tensor with shape
        mock_model.generate.return_value = mock_output
        mock_model.device = 'cpu'

        # Mock JSON parsing
        mock_json_loads.return_value = {"Title": "Test", "Note": "Content", "Duration": 3600}

        result = self.note_manager.parse_note_creation("create test note", mock_tokenizer, mock_model)

        self.assertEqual(result['Title'], "Test")
        self.assertEqual(result['Note'], "Content")
        self.assertEqual(result['Duration'], 3600)

    @patch('json.loads')
    def test_parse_note_creation_json_error_fallback(self, mock_json_loads):
        """Test note parsing with JSON error fallback"""
        # Mock the LLM components
        mock_tokenizer = Mock()
        mock_model = Mock()

        # Mock tokenizer methods
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 2
        
        # Create a mock inputs object that behaves like a dict
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        mock_tokenizer.apply_chat_template.return_value = Mock()  # Return a mock tensor
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        
        mock_tokenizer.decode.return_value = "invalid json response"

        # Mock model generate
        mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Mock tensor with shape
        mock_model.generate.return_value = mock_output
        mock_model.device = 'cpu'

        # Mock JSON parsing to raise error
        mock_json_loads.side_effect = ValueError("Invalid JSON")

        result = self.note_manager.parse_note_creation("create test note", mock_tokenizer, mock_model)

        # Should return fallback result
        self.assertIn('Title', result)
        self.assertIn('Note', result)
        self.assertIsNone(result['Duration'])


class TestTextProcessing(unittest.TestCase):
    """Test cases for text processing functionality"""

    def test_confirm_and_apply_command_correction_exact_match(self):
        """Test command correction with exact match"""
        input_text = "create note"
        result = confirm_and_apply_command_correction(input_text, threshold=0.85)
        self.assertEqual(result, "create note")

    @patch('builtins.input', return_value='y')
    def test_confirm_and_apply_command_correction_typo(self, mock_input):
        """Test command correction with typo"""
        input_text = "creat note"  # typo in "create"
        result = confirm_and_apply_command_correction(input_text, threshold=0.85)
        # Should correct "creat" to "create"
        self.assertEqual(result, "create note")

    def test_confirm_and_apply_command_correction_no_match(self):
        """Test command correction with no close match"""
        input_text = "completely unrelated text"
        result = confirm_and_apply_command_correction(input_text, threshold=0.85)
        self.assertEqual(result, "completely unrelated text")


if __name__ == '__main__':
    unittest.main()