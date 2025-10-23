# News Classification Lab

A comprehensive machine learning project for news classification using both traditional transformer models and Large Language Models (LLMs). This repository contains two main approaches to news classification: transformer-based classification and LLM-based classification.

## 📋 Project Overview

This project implements news classification using two different methodologies:

1. **Traditional ML Approach** (`News_Classification0.ipynb`): Uses Hugging Face transformers (RoBERTa, DeBERTa) for classification
2. **LLM Approach** (`bonus.ipynb`): Uses OpenAI's ChatGPT for intelligent news classification

## 🗂️ Project Structure

```
News_Classification-labs/
├── News_Classification0.ipynb    # Main transformer-based classification notebook
├── bonus.ipynb                   # LLM-based classification notebook
├── data/
│   └── rss_feed.json            # RSS feed data for LLM classification
├── src/
│   └── train_small.py           # Training functions for transformers
├── outputs/                     # Model outputs and visualizations
├── .env                         # Environment variables (OpenAI API key)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### ⚡ **Recommended: Google Colab with TPUs**

**We strongly recommend using Google Colab with TPUs for faster training:**

1. **Open in Colab**: Click the Colab badge in `News_Classification0.ipynb`
2. **Enable TPU**: Runtime → Change runtime type → TPU
3. **Benefits**:
   - **10x faster training** with TPU acceleration
   - Free GPU/TPU access for transformer training
   - Pre-installed ML libraries
   - No local setup required

### 🔗 **Connecting Repository to Colab**

**We're aiming to connect this repository directly to Colab for seamless integration:**

- **Direct GitHub Integration**: Clone the repo directly in Colab
- **Automatic Setup**: The notebooks will automatically create the required directory structure
- **File Replication**: All necessary files and data will be replicated in the Colab environment
- **Persistent Storage**: Results and outputs will be saved to your Google Drive

**Quick Colab Setup:**
```python
# Run this in Colab to set up the project
!git clone https://github.com/your-username/News_Classification-labs.git
%cd News_Classification-labs
!pip install -r requirements.txt
```

### For Transformer-based Classification (News_Classification0.ipynb)

1. **Open the notebook**: `News_Classification0.ipynb`
2. **Run all cells** to:
   - Set up the environment and directory structure
   - Install required packages (torch, transformers, datasets, etc.)
   - Create the training function `train_one_small()`
   - Train and compare multiple models (RoBERTa, DeBERTa, DistilRoBERTa)
   - Generate performance visualizations

### For LLM-based Classification (bonus.ipynb)

1. **Set up OpenAI API key** (choose one method):

   **Method 1: Environment Variable**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   **Method 2: .env file (Recommended)**
   Create a `.env` file in the project root:
   ```bash
   # .env file
   OPENAI_API_KEY=your-api-key-here
   ```

2. **Open the notebook**: `bonus.ipynb`
3. **Run all cells** to:
   - Load and process RSS feed data
   - Classify news using ChatGPT
   - Generate comprehensive analysis and visualizations

## 📊 Notebook Details

### News_Classification0.ipynb - Transformer-based Classification

**Purpose**: Implements traditional machine learning approach using Hugging Face transformers for news classification.

**Key Features**:
- **Dataset**: AG News dataset (120,000 training samples, 7,600 test samples)
- **Models Tested**: 
  - RoBERTa-base (FacebookAI/roberta-base)
  - DeBERTa-v3-small (microsoft/deberta-v3-small)
  - DistilRoBERTa (fallback for ModernBERT)
- **Data Split**: 70% train / 15% validation / 15% test
- **Subset Training**: Configurable sample sizes (1000/500/500 for quick experiments)
- **Metrics**: Accuracy and F1-Macro scores
- **Visualizations**: Bar charts, scatter plots, and ranking visualizations

**Technical Implementation**:
- Uses `AutoTokenizer` and `AutoModelForSequenceClassification`
- Implements `Trainer` with `TrainingArguments`
- Supports mixed precision training (FP16)
- **Optimized for TPU/GPU acceleration** (recommended for Colab)
- Saves model metrics to JSON files
- Generates comparative visualizations

**Results Summary**:
| Model | F1 Macro | Accuracy | Performance |
|-------|----------|----------|-------------|
| RoBERTa-base | 0.896 | 0.898 | Best overall performance |
| DeBERTa-v3-small | 0.739 | 0.766 | Lower performance with small datasets |
| DistilRoBERTa | 0.883 | 0.886 | Excellent balance of speed and accuracy |

### bonus.ipynb - LLM-based Classification

**Purpose**: Implements modern LLM-based classification using OpenAI's ChatGPT for intelligent news categorization.

**Key Features**:
- **Dataset**: 50 RPP news items from RSS feed
- **Model**: OpenAI GPT-3.5-turbo with structured outputs
- **Categories**: 4 AG News categories (World, Sports, Business, Science/Tech)
- **Features**: 
  - Confidence scoring for each classification
  - Detailed reasoning for classification decisions
  - Comprehensive error handling and fallback mechanisms
  - Rate limiting to respect API constraints

**Technical Implementation**:
- Uses OpenAI API with structured output parsing
- Implements Pydantic models for response validation
- Batch processing with progress tracking
- Comprehensive data analysis and visualization
- Results export to CSV and summary files

**Analysis Features**:
- Category distribution analysis
- Confidence score analysis
- Success rate tracking
- Detailed statistical insights
- Multiple visualization types (pie charts, bar charts, histograms)

## 🔧 Dependencies

### Core Dependencies
- `torch>=2.8.0` - PyTorch for deep learning
- `transformers>=4.44.2` - Hugging Face transformers
- `datasets>=2.21.0` - Dataset loading and processing
- `scikit-learn>=1.6.0` - Machine learning utilities
- `pandas>=2.0.3` - Data manipulation
- `matplotlib>=3.9.2` - Plotting and visualization
- `seaborn>=0.13.2` - Statistical visualization

### LLM Dependencies
- `openai` - OpenAI API client
- `pydantic` - Data validation and structured outputs
- `python-dotenv` - Environment variable management (for .env file support)

## 📈 Usage Examples

### Transformer Training Example

```python
from src.train_small import train_one_small

# Train RoBERTa on a small subset
result = train_one_small(
    model_name='FacebookAI/roberta-base',
    epochs=1,
    batch_size=8,
    max_length=64,
    train_samples=1000,
    val_samples=500,
    test_samples=500
)

print(f"Model: {result['model']}")
print(f"Accuracy: {result['accuracy']:.3f}")
print(f"F1 Macro: {result['f1_macro']:.3f}")
```

### LLM Classification Example

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Classify a single news item
result = classify_news_with_chatgpt(
    client, 
    title="Your news title",
    description="Your news description"
)

print(f"Category: {result.category_name}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## 📊 Output Files

### Transformer Approach
- `outputs/{model_name}/metrics.json` - Model performance metrics
- `outputs/plots/` - Visualization files (PNG format)
- `outputs/f1_comparison.csv` - Comparative results

### LLM Approach
- `llm_classification_results.csv` - Complete classification results
- `llm_classification_results_summary.txt` - Summary statistics
- Generated visualizations for analysis

## 🎯 Key Insights

### Transformer Approach
- **RoBERTa-base** achieves the best performance (F1: 0.896)
- **DistilRoBERTa** provides excellent speed/accuracy balance
- **DeBERTa** requires more data for optimal performance
- Subset training enables rapid experimentation

### LLM Approach
- High classification accuracy with contextual understanding
- Excellent handling of Spanish language content
- Detailed reasoning and confidence scoring
- Scalable for large-scale news processing

## 🚀 Getting Started

### **Option 1: Google Colab (Recommended)**
1. **Open in Colab**: Click the Colab badge in `News_Classification0.ipynb`
2. **Enable TPU**: Runtime → Change runtime type → TPU
3. **Run all cells**: The notebook will automatically set up the environment
4. **For LLM classification**: Set your OpenAI API key in the `.env` file

### **Option 2: Local Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/News_Classification-labs.git
   cd News_Classification-labs
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run transformer classification**:
   - Open `News_Classification0.ipynb`
   - Execute all cells

4. **Run LLM classification**:
   - Set your OpenAI API key (create `.env` file with `OPENAI_API_KEY=your-key`)
   - Open `bonus.ipynb`
   - Execute all cells

## 📝 Notes

- **Colab Integration**: The repository is designed for seamless Colab integration with automatic setup
- **File Replication**: All necessary files and directories are automatically created in Colab
- The transformer approach is ideal for offline, high-performance classification
- The LLM approach provides superior contextual understanding and reasoning
- Both approaches can be combined for hybrid classification systems
- Results are reproducible with fixed random seeds
- All visualizations are automatically saved to the `outputs/` directory
- **TPU Acceleration**: Colab TPUs provide 10x faster training for transformer models

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is open source and available under the MIT License.