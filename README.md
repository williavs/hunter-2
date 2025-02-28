# GTM Scrape - Contact Data Enrichment Tool

A Streamlit application for enriching contact data by scraping website content from company URLs.

## Features

- Upload CSV files containing contact data with website URLs
- Automatically detect the website column
- Parallel scraping of website content for efficiency
- Simple and clean UI with data editor for viewing and modifying data
- Progress tracking during scraping
- Download enriched data as CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/williavs/gtm-scrape.git
cd gtm-scrape
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py
```

2. Upload your CSV file containing contact data with website URLs
3. Select the column containing website URLs
4. Adjust scraping parameters if needed
5. Click "Process Websites" to start scraping
6. Download the enriched data when processing is complete

## Dependencies

- streamlit
- pandas
- numpy
- rapidfuzz
- requests
- html2text
- tqdm

## License

MIT 