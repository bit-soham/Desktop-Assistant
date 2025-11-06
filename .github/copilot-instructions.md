# Desktop Assistant AI Agent Instructions

## Project Overview
This is a desktop assistant application built with Python that integrates voice recognition, natural language processing, and system automation capabilities. The project follows a modular architecture with clear separation of concerns.

## Architecture
- `core/`: Core application logic
  - `command_parser.py`: Processes user commands
  - `response_generator.py`: Generates responses to commands
  - `task_manager.py`: Manages task execution
  - `voice_recognition.py`: Handles voice input processing

- `services/`: Service implementations
  - `browser_services.py`: Web browser automation
  - `file_services.py`: File system operations
  - `system_services.py`: System-level operations
  - `user_data.py`: User data management

- `ui/`: Qt-based user interface
  - `components/`: Reusable UI components
  - `assets/`: UI resources and styles

## Development Setup
1. Create virtual environment: `python -m venv models/speech_rag_env`
2. Activate environment:
   - Windows: `.\models\speech_rag_env\Scripts\Activate.ps1`
   - Unix: `source models/speech_rag_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Key Dependencies
- PyAudio: Audio I/O
- Google Generative AI: Natural language processing
- NumPy: Numerical computations
- WebSocket Client: Real-time communication

## Testing
- Test files are organized in `tests/` directory
- Each core component has corresponding test file (e.g., `test_task_manager.py`)
- Run tests using Python's unittest framework

## Project Conventions
1. Services are implemented as separate modules in `services/`
2. UI components follow Qt naming conventions
3. Configuration is stored in YAML format (`config/config.yaml`)
4. Google API credentials required in `client_secrets.json`

## Common Development Tasks
1. Adding new commands:
   - Extend `core/command_parser.py`
   - Add corresponding handler in `core/task_manager.py`
2. UI modifications:
   - Components in `ui/components/`
   - Styles in `ui/assets/styles.qss`

## Performance Considerations
- Voice recognition runs in separate thread
- Background tasks handled by task_manager
- UI remains responsive during long-running operations