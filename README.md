# Chat with your PDF 📄

This is an advanced web application that allows you to have an intelligent conversation with your PDF documents. Upload a PDF, and the application will process it, enabling you to ask questions, get summaries, and receive insightful answers based on the document's content. This project leverages the power of large language models (LLMs) and vector databases to create a seamless and interactive experience.

## 🌟 Features

- **Interactive Chat Interface:** A clean and user-friendly interface built with Streamlit for uploading PDFs and engaging in a conversation.
- **Conversational Memory:** The chatbot remembers the context of the conversation, allowing for follow-up questions and a more natural dialogue.
- **High-Quality Answers:** Leverages state-of-the-art language models from Google's Generative AI suite to provide accurate and relevant answers.
- **Fast and Efficient:** Utilizes parallel processing (multi-threading) to quickly prepare your PDF for questioning, ensuring a smooth user experience.
- **Source-Aware Responses:** The model can pinpoint the sources of its answers within the document, providing better context and verifiability.

## ⚙️ How It Works

The application follows a sophisticated process to enable a conversation with your PDF:

1.  **PDF Loading:** The uploaded PDF is loaded and its text content is extracted using `PyMuPDF`.
2.  **Text Chunking:** The extracted text is split into smaller, manageable chunks using a `RecursiveCharacterTextSplitter`. This is crucial for fitting the context into the language model's limits.
3.  **Chunk Refinement:** Each text chunk is individually refined by a language model (`gemini-pro`) to improve its clarity and coherence. This step is parallelized for maximum speed.
4.  **Embedding and Indexing:** The refined chunks are converted into vector embeddings using `GoogleGenerativeAIEmbeddings` and stored in a `FAISS` vector store for efficient similarity searches.
5.  **Conversational Chain:** A `ConversationalRetrievalChain` is created, which uses the vector store as a retriever and a powerful language model as the question-answering engine. This chain is what enables the conversational memory.
6.  **User Interaction:** The Streamlit interface captures the user's questions, passes them to the conversational chain along with the chat history, and displays the model's response.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.7 or higher
- A Google API key. You can obtain one from the [Google AI Studio](https://aistudio.google.com/).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/PDF-Bot.git
    cd PDF-Bot
    ```

2.  **Install the dependencies:**
    It is recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a file named `.env` in the root of your project and add your Google API key. This file is loaded at runtime to configure the application.

```
# .env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

### Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server, and you can access the application in your web browser, typically at `http://localhost:8501`.

## 🛠️ Technologies Used

- **Backend:** Python
- **Web Framework:** Streamlit
- **LLM Orchestration:** LangChain
- **Language Models:** Google Generative AI (Gemini Pro)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **PDF Processing:** PyMuPDF

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
