<<<<<<< HEAD
---
title: 10MS Viral Video Prediction Agent
emoji: 🎯
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---

# 10 Minute School — Viral Video Prediction Agent

An AI agent that predicts whether a 10 Minute School YouTube video will go viral.

**Stack:** LightGBM · LangChain · Gemini · Gradio  
**Data:** 2,845 real 10 Minute School YouTube videos  
**Model:** Binary classifier — top 10% views = viral  

## Features
- Natural language chat interface
- LightGBM model trained on real channel data
- Detects subject, duration bucket, upload timing
- Gives viral probability % + actionable advice

## GitHub
[github.com/tmridwan](https://github.com/tmridwan)
=======
# 10 Minute School — YouTube Data Analysis

📊 **[View Interactive Tableau Dashboard](https://public.tableau.com/views/10_MS_yt_analysis/Dashboard1)**

An exploratory data analysis project focused on 10 Minute School’s YouTube channel. This project examines video performance, audience engagement, publishing patterns, and content trends using YouTube data.

## Project Overview

The goal of this project is to explore the performance of 10 Minute School’s YouTube content and uncover insights through data analysis and visualization.

## Tools & Technologies

- **Python** — Data analysis
- **YouTube Data API v3** — Data collection
- **Pandas** — Data manipulation
- **Matplotlib & Seaborn** — Data visualization
- **Jupyter Notebook** — Analysis environment
- **Tableau Public** — Interactive dashboard

## Project Workflow

1. **Data Collection:** Retrieved YouTube video information using YouTube Data API v3.
2. **Data Cleaning:** Prepared and organized the collected dataset using Pandas.
3. **Exploratory Data Analysis:** Explored video statistics and content performance.
4. **Data Visualization:** Created charts to identify patterns and trends.
5. **Dashboard Development:** Presented the analysis through an interactive Tableau dashboard.

## Repository Structure

```text
10-ms/
│
├── charts/
├── 10ms_videos.csv
├── 10ms_youtube_analysis.ipynb
├── .gitignore
├── .env.example
└── README.md
```

## Interactive Dashboard

Explore the analysis through Tableau Public:

**[View 10 Minute School YouTube Analysis Dashboard](https://public.tableau.com/views/10_MS_yt_analysis/Dashboard1?:language=en-US&publish=yes)**

## Getting Started

### Clone the repository

```bash
git clone https://github.com/tmridwan/10-ms.git
cd 10-ms
```

### Install dependencies

```bash
pip install pandas matplotlib seaborn google-api-python-client python-dotenv jupyter
```

### Configure API credentials

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Load the API key securely in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
```

### Run the notebook

```bash
jupyter notebook 10ms_youtube_analysis.ipynb
```

## Notes

- YouTube statistics may change over time.
- API access requires a valid YouTube Data API v3 key.
- The `.env` file containing API credentials should never be committed to GitHub.
- The `.env.example` file is provided as a configuration template.

## Author

**Taiob Md Ridwan**

- [GitHub](https://github.com/tmridwan)
- [Portfolio](https://tmridwan.github.io/portfolio_/)
- [Tableau Public](https://public.tableau.com/views/10_MS_yt_analysis/Dashboard1)

---

*An independent data analytics project exploring educational content performance through data.*
>>>>>>> e9b49a4622fc60c320bbd15cf3b09233e69980f2
