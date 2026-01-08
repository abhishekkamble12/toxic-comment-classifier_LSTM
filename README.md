This project tackles the multi-label classification problem of identifying toxic behavior in online comments. Unlike simple sentiment analysis (positive/negative), this model classifies text into 6 specific categories of toxicity:

Toxic

Severe Toxic

Obscene

Threat

Insult

Identity Hate

The model processes raw text using a custom vectorization layer and utilizes a Bidirectional LSTM neural network to understand context and sequential dependencies in the text.

🛠️ Tech Stack
Language: Python

Deep Learning: TensorFlow, Keras

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Dataset Management: opendatasets (Kaggle API)

Interface: Gradio

📂 Dataset
The project uses the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle.

Source: Kaggle Competition Link

Input: Wikipedia comments which have been labeled by human raters for toxic behavior.

🧠 Model Architecture
The model is built using the Keras Sequential API:

TextVectorization Layer: Converts raw text into integer sequences (Vocab limit: 200,000 words, Sequence length: 1800).

Embedding Layer: Transforms integer sequences into dense vectors.

Bidirectional LSTM: Captures patterns from both past and future contexts in the sentence (32 units).

Dense Layers: Fully connected layers (128 units -> 256 units -> 128 units) with ReLU activation for feature extraction.

Output Layer: 6 units with Sigmoid activation (allowing independent probabilities for each class).

🚀 Installation & Usage
1. Clone the Repository
Bash

git clone https://github.com/your-username/toxic-comment-classifier.git
cd toxic-comment-classifier
2. Install Dependencies
Bash

pip install tensorflow pandas numpy matplotlib seaborn opendatasets gradio
3. Run the Notebook
Open the .ipynb file in Jupyter Notebook or Google Colab. The notebook handles:

Downloading the dataset directly from Kaggle using your API credentials.

Preprocessing and tokenizing the text.

Training the LSTM model.

4. Launch the App
The notebook includes a Gradio interface. Run the final cell to launch a web UI where you can type sentences and see real-time toxicity scores.

Python

import gradio as gr
# ... (load model and define score_comment function) ...
interface.launch(share=True)
📊 Results
The model outputs a probability score (0 to 1) for each of the 6 tags.

Low Score: The comment is likely safe.

High Score: The comment exhibits that specific toxic trait.
