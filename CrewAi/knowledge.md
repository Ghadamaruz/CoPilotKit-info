
# Knowledge

## What is Knowledge?

Knowledge in CrewAI is a powerful system that allows AI agents to access and utilize external information sources during their tasks.
Think of it as giving your agents a reference library they can consult while working.

Key benefits of using Knowledge:

- Enhance agents with domain-specific information
- Support decisions with real-world data
- Maintain context across conversations
- Ground responses in factual information

## [​](https://docs.crewai.com/concepts/knowledge\#supported-knowledge-sources)  Supported Knowledge Sources

CrewAI supports various types of knowledge sources out of the box:

## Text Sources

- Raw strings
- Text files (.txt)
- PDF documents

## Structured Data

- CSV files
- Excel spreadsheets
- JSON documents

## [​](https://docs.crewai.com/concepts/knowledge\#supported-knowledge-parameters)  Supported Knowledge Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `sources` | **List\[BaseKnowledgeSource\]** | Yes | List of knowledge sources that provide content to be stored and queried. Can include PDF, CSV, Excel, JSON, text files, or string content. |
| `collection_name` | **str** | No | Name of the collection where the knowledge will be stored. Used to identify different sets of knowledge. Defaults to “knowledge” if not provided. |
| `storage` | **Optional\[KnowledgeStorage\]** | No | Custom storage configuration for managing how the knowledge is stored and retrieved. If not provided, a default storage will be created. |

## [​](https://docs.crewai.com/concepts/knowledge\#quickstart-example)  Quickstart Example

For file-Based Knowledge Sources, make sure to place your files in a `knowledge` directory at the root of your project.
Also, use relative paths from the `knowledge` directory when creating the source.

Here’s an example using string-based knowledge:

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# Create a knowledge source
content = "Users name is John. He is 30 years old and lives in San Francisco."
string_source = StringKnowledgeSource(
    content=content,
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gpt-4o-mini", temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[string_source], # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
)

result = crew.kickoff(inputs={"question": "What city does John live in and how old is he?"})

```

Here’s another example with the `CrewDoclingSource`. The CrewDoclingSource is actually quite versatile and can handle multiple file formats including MD, PDF, DOCX, HTML, and more.

You need to install `docling` for the following example to work: `uv add docling`

```python
from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

# Create a knowledge source
content_source = CrewDoclingSource(
    file_paths=[\
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",\
        "https://lilianweng.github.io/posts/2024-07-07-hallucination",\
    ],
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gpt-4o-mini", temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About papers",
    goal="You know everything about the papers.",
    backstory="""You are a master at understanding papers and their content.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[\
        content_source\
    ],  # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
)

result = crew.kickoff(
    inputs={
        "question": "What is the reward hacking paper about? Be sure to provide sources."
    }
)

```

## [​](https://docs.crewai.com/concepts/knowledge\#more-examples)  More Examples

Here are examples of how to use different types of knowledge sources:

Note: Please ensure that you create the ./knowldge folder. All source files (e.g., .txt, .pdf, .xlsx, .json) should be placed in this folder for centralized management.

### [​](https://docs.crewai.com/concepts/knowledge\#text-file-knowledge-source)  Text File Knowledge Source

```python
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Create a text file knowledge source
text_source = TextFileKnowledgeSource(
    file_paths=["document.txt", "another.txt"]
)

# Create crew with text file source on agents or crew level
agent = Agent(
    ...
    knowledge_sources=[text_source]
)

crew = Crew(
    ...
    knowledge_sources=[text_source]
)

```

### [​](https://docs.crewai.com/concepts/knowledge\#pdf-knowledge-source)  PDF Knowledge Source

```python
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Create a PDF knowledge source
pdf_source = PDFKnowledgeSource(
    file_paths=["document.pdf", "another.pdf"]
)

# Create crew with PDF knowledge source on agents or crew level
agent = Agent(
    ...
    knowledge_sources=[pdf_source]
)

crew = Crew(
    ...
    knowledge_sources=[pdf_source]
)

```

### [​](https://docs.crewai.com/concepts/knowledge\#csv-knowledge-source)  CSV Knowledge Source

```python
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource

# Create a CSV knowledge source
csv_source = CSVKnowledgeSource(
    file_paths=["data.csv"]
)

# Create crew with CSV knowledge source or on agent level
agent = Agent(
    ...
    knowledge_sources=[csv_source]
)

crew = Crew(
    ...
    knowledge_sources=[csv_source]
)

```

### [​](https://docs.crewai.com/concepts/knowledge\#excel-knowledge-source)  Excel Knowledge Source

```python
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource

# Create an Excel knowledge source
excel_source = ExcelKnowledgeSource(
    file_paths=["spreadsheet.xlsx"]
)

# Create crew with Excel knowledge source on agents or crew level
agent = Agent(
    ...
    knowledge_sources=[excel_source]
)

crew = Crew(
    ...
    knowledge_sources=[excel_source]
)

```

### [​](https://docs.crewai.com/concepts/knowledge\#json-knowledge-source)  JSON Knowledge Source

```python
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource

# Create a JSON knowledge source
json_source = JSONKnowledgeSource(
    file_paths=["data.json"]
)

# Create crew with JSON knowledge source on agents or crew level
agent = Agent(
    ...
    knowledge_sources=[json_source]
)

crew = Crew(
    ...
    knowledge_sources=[json_source]
)

```

## [​](https://docs.crewai.com/concepts/knowledge\#knowledge-configuration)  Knowledge Configuration

### [​](https://docs.crewai.com/concepts/knowledge\#chunking-configuration)  Chunking Configuration

Knowledge sources automatically chunk content for better processing.
You can configure chunking behavior in your knowledge sources:

```python
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

source = StringKnowledgeSource(
    content="Your content here",
    chunk_size=4000,      # Maximum size of each chunk (default: 4000)
    chunk_overlap=200     # Overlap between chunks (default: 200)
)

```

The chunking configuration helps in:

- Breaking down large documents into manageable pieces
- Maintaining context through chunk overlap
- Optimizing retrieval accuracy

### [​](https://docs.crewai.com/concepts/knowledge\#embeddings-configuration)  Embeddings Configuration

You can also configure the embedder for the knowledge store.
This is useful if you want to use a different embedder for the knowledge store than the one used for the agents.
The `embedder` parameter supports various embedding model providers that include:

- `openai`: OpenAI’s embedding models
- `google`: Google’s text embedding models
- `azure`: Azure OpenAI embeddings
- `ollama`: Local embeddings with Ollama
- `vertexai`: Google Cloud VertexAI embeddings
- `cohere`: Cohere’s embedding models
- `voyageai`: VoyageAI’s embedding models
- `bedrock`: AWS Bedrock embeddings
- `huggingface`: Hugging Face models
- `watson`: IBM Watson embeddings

Here’s an example of how to configure the embedder for the knowledge store using Google’s `text-embedding-004` model:

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import os

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Create a knowledge source
content = "Users name is John. He is 30 years old and lives in San Francisco."
string_source = StringKnowledgeSource(
    content=content,
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
gemini_llm = LLM(
    model="gemini/gemini-1.5-pro-002",
    api_key=GEMINI_API_KEY,
    temperature=0,
)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm,
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
    }
)

task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[string_source],
    embedder={
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
    }
)

result = crew.kickoff(inputs={"question": "What city does John live in and how old is he?"})

```

## [​](https://docs.crewai.com/concepts/knowledge\#clearing-knowledge)  Clearing Knowledge

If you need to clear the knowledge stored in CrewAI, you can use the `crewai reset-memories` command with the `--knowledge` option.

```bash
crewai reset-memories --knowledge

```

This is useful when you’ve updated your knowledge sources and want to ensure that the agents are using the most recent information.

## [​](https://docs.crewai.com/concepts/knowledge\#agent-specific-knowledge)  Agent-Specific Knowledge

While knowledge can be provided at the crew level using `crew.knowledge_sources`, individual agents can also have their own knowledge sources using the `knowledge_sources` parameter:

```python
from crewai import Agent, Task, Crew
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# Create agent-specific knowledge about a product
product_specs = StringKnowledgeSource(
    content="""The XPS 13 laptop features:
    - 13.4-inch 4K display
    - Intel Core i7 processor
    - 16GB RAM
    - 512GB SSD storage
    - 12-hour battery life""",
    metadata={"category": "product_specs"}
)

# Create a support agent with product knowledge
support_agent = Agent(
    role="Technical Support Specialist",
    goal="Provide accurate product information and support.",
    backstory="You are an expert on our laptop products and specifications.",
    knowledge_sources=[product_specs]  # Agent-specific knowledge
)

# Create a task that requires product knowledge
support_task = Task(
    description="Answer this customer question: {question}",
    agent=support_agent
)

# Create and run the crew
crew = Crew(
    agents=[support_agent],
    tasks=[support_task]
)

# Get answer about the laptop's specifications
result = crew.kickoff(
    inputs={"question": "What is the storage capacity of the XPS 13?"}
)

```

Benefits of agent-specific knowledge:

- Give agents specialized information for their roles
- Maintain separation of concerns between agents
- Combine with crew-level knowledge for layered information access

## [​](https://docs.crewai.com/concepts/knowledge\#custom-knowledge-sources)  Custom Knowledge Sources

CrewAI allows you to create custom knowledge sources for any type of data by extending the `BaseKnowledgeSource` class. Let’s create a practical example that fetches and processes space news articles.

#### [​](https://docs.crewai.com/concepts/knowledge\#space-news-knowledge-source-example)  Space News Knowledge Source Example

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
import requests
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field

class SpaceNewsKnowledgeSource(BaseKnowledgeSource):
    """Knowledge source that fetches data from Space News API."""

    api_endpoint: str = Field(description="API endpoint URL")
    limit: int = Field(default=10, description="Number of articles to fetch")

    def load_content(self) -> Dict[Any, str]:
        """Fetch and format space news articles."""
        try:
            response = requests.get(
                f"{self.api_endpoint}?limit={self.limit}"
            )
            response.raise_for_status()

            data = response.json()
            articles = data.get('results', [])

            formatted_data = self.validate_content(articles)
            return {self.api_endpoint: formatted_data}
        except Exception as e:
            raise ValueError(f"Failed to fetch space news: {str(e)}")

    def validate_content(self, articles: list) -> str:
        """Format articles into readable text."""
        formatted = "Space News Articles:\n\n"
        for article in articles:
            formatted += f"""
                Title: {article['title']}
                Published: {article['published_at']}
                Summary: {article['summary']}
                News Site: {article['news_site']}
                URL: {article['url']}
                -------------------"""
        return formatted

    def add(self) -> None:
        """Process and store the articles."""
        content = self.load_content()
        for _, text in content.items():
            chunks = self._chunk_text(text)
            self.chunks.extend(chunks)

        self._save_documents()

# Create knowledge source
recent_news = SpaceNewsKnowledgeSource(
    api_endpoint="https://api.spaceflightnewsapi.net/v4/articles",
    limit=10,
)

# Create specialized agent
space_analyst = Agent(
    role="Space News Analyst",
    goal="Answer questions about space news accurately and comprehensively",
    backstory="""You are a space industry analyst with expertise in space exploration,
    satellite technology, and space industry trends. You excel at answering questions
    about space news and providing detailed, accurate information.""",
    knowledge_sources=[recent_news],
    llm=LLM(model="gpt-4", temperature=0.0)
)

# Create task that handles user questions
analysis_task = Task(
    description="Answer this question about space news: {user_question}",
    expected_output="A detailed answer based on the recent space news articles",
    agent=space_analyst
)

# Create and run the crew
crew = Crew(
    agents=[space_analyst],
    tasks=[analysis_task],
    verbose=True,
    process=Process.sequential
)

# Example usage
result = crew.kickoff(
    inputs={"user_question": "What are the latest developments in space exploration?"}
)

```

#### [​](https://docs.crewai.com/concepts/knowledge\#key-components-explained)  Key Components Explained

1. **Custom Knowledge Source ( `SpaceNewsKnowledgeSource`)**:
   - Extends `BaseKnowledgeSource` for integration with CrewAI
   - Configurable API endpoint and article limit
   - Implements three key methods:
     - `load_content()`: Fetches articles from the API
     - `_format_articles()`: Structures the articles into readable text
     - `add()`: Processes and stores the content
2. **Agent Configuration**:
   - Specialized role as a Space News Analyst
   - Uses the knowledge source to access space news
3. **Task Setup**:
   - Takes a user question as input through `{user_question}`
   - Designed to provide detailed answers based on the knowledge source
4. **Crew Orchestration**:
   - Manages the workflow between agent and task
   - Handles input/output through the kickoff method

This example demonstrates how to:

- Create a custom knowledge source that fetches real-time data
- Process and format external data for AI consumption
- Use the knowledge source to answer specific user questions
- Integrate everything seamlessly with CrewAI’s agent system

#### [​](https://docs.crewai.com/concepts/knowledge\#about-the-spaceflight-news-api)  About the Spaceflight News API

The example uses the [Spaceflight News API](https://api.spaceflightnewsapi.net/v4/docs/), which:

- Provides free access to space-related news articles
- Requires no authentication
- Returns structured data about space news
- Supports pagination and filtering

You can customize the API query by modifying the endpoint URL:

```python
# Fetch more articles
recent_news = SpaceNewsKnowledgeSource(
    api_endpoint="https://api.spaceflightnewsapi.net/v4/articles",
    limit=20,  # Increase the number of articles
)

# Add search parameters
recent_news = SpaceNewsKnowledgeSource(
    api_endpoint="https://api.spaceflightnewsapi.net/v4/articles?search=NASA", # Search for NASA news
    limit=10,
)

```

## [​](https://docs.crewai.com/concepts/knowledge\#best-practices)  Best Practices

Content Organization

- Keep chunk sizes appropriate for your content type
- Consider content overlap for context preservation
- Organize related information into separate knowledge sources

Performance Tips

- Adjust chunk sizes based on content complexity
- Configure appropriate embedding models
- Consider using local embedding providers for faster processing