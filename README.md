# MetaBeeAI LLM Project

A project for processing and analyzing scientific papers about bee populations using Large Language Models and Vision AI.

## Project Structure 

```
MetaBeeAI_LLM/
├── papers/ # Directory containing PDF papers and their processed versions
│ ├── 001/ # Individual paper directories
│ │ ├── 001_main.pdf # Original PDF file
│ │ └── pages/ # Processed PDF pages and their JSON analyses
│ └── ...
├── notebooks/
│ ├── LLM prototype.ipynb # Main LLM processing notebook
│ └── VisionAgent_pipeline.ipynb # Vision AI processing pipeline
├── .env # Environment variables (not tracked in git)
├── requirements.txt # Python dependencies
└── README.md # This file
```




## Setup

1. Create and activate a virtual environment:
``` bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install dependencies:
``` bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
``` bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LANDING_API_KEY=your_landing_key
```


## Usage

1. Place PDF papers in the appropriate numbered subdirectory under `papers/`
2. Run the Vision Agent pipeline to process PDFs:
   - Open `VisionAgent_pipeline.ipynb`
   - Run all cells to split PDFs and analyze pages
3. Run the LLM analysis:
   - Open `LLM prototype.ipynb`
   - Run all cells to process the extracted information

## Features

- PDF splitting into overlapping 2-page segments
- Vision AI analysis of PDF content
- LLM-based information extraction
- Structured JSON output for each processed page
- Automated logging of processing status

## Dependencies

- python-dotenv: Environment variable management
- PyPDF2: PDF processing
- vision-agent: Vision AI analysis
- mllm: LLM processing
- Other dependencies listed in requirements.txt

## Notes

- API keys are required for OpenAI, Anthropic, and Landing AI services
- Processing large PDFs may take significant time
- Some API calls may fail and require retries
- Check processing logs in papers/ directory for status updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.
