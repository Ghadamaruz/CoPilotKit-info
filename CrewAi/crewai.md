---
title: CrewAI
description: An agentic framework for building LLM applications that can be used with Copilotkit.
icon: custom/langchain
---

<Frame>
  <img
    src="/images/coagents/coagents-highlevel-overview.png"
    alt="CoAgents High Level Overview"
    className="mb-10"
  />
</Frame>

CrewAI is an agentic framework for building LLM applications that can be used with Copilotkit. CrewAi Flows allow developers
to combine and coordinate coding tasks and Crews efficiently, providing a robust framework for building sophisticated AI automations.

## CoAgents and CrewAI

How do CoAgents extend CrewAI? Let's read the first sentence of their [page on Flows](https://docs.crewai.com/concepts/flows) to understand.

> Flows allow you to create structured, event-driven workflows. They provide a seamless way to connect multiple tasks, manage state, and control the flow of execution in your AI applications.

Let's break down some key terms and understand how they relate to and are implemented by CoAgents.

- **Manage state**: CoAgents have bi-directional state sharing with the agent and UI. This allows for the agent to remember
  information from previous messages and the UI to update the agent with new information. Read more about how state sharing works
  [here](/crewai-crews/shared-state).
- **Multi-actor**: CoAgents allow for multiple agents to interact with each other. Copilotkit acts as the "ground-truth"
  when transitioning between agents. Read more about how multi-actor workflows work [here](/crewai-crews/multi-agent-flows)
  and how messages are managed [here](/crewai-crews/concepts/message-management).
- **LLMs**: CoAgents use large language models to generate responses. This is useful for building applications that need to
  generate natural language responses.

Some additional functionality not mentioned here is:

- **Human in the loop**: CoAgents enabled human review and approval of generated responses. Read more about how this works
  [here](/crewai-crews/human-in-the-loop).
- **Tool calling**: Tool calling is a fundamental building block for agentic workflows. They allow for greater control over what
  the agent can do and can be used to interact with external systems. CoAgents allow you to easily render in-progress
  tool calls in the UI so your users know what's happening. Read more about streaming tool calls [here](/crewai-crews/shared-state/predictive-state-updates).

## Building with Python

You can build CrewAI applications using Python. Check out the [CrewAI docs](https://docs.crewai.com/introduction) for more information.

## CrewAI Enterprise

Turn any crew into an API within seconds
Connect to your apps using hooks, REST, gRPC and more
Get access to templates, custom tools and early UI
Get business support, SLA, private VPC

CrewAI enterprise is a platform for deploying and monitoring CrewAI applications. Read more about it on the
[CrewAI website](https://www.crewai.com/enterprise).

If you want to take the next step to deploy your CrewAI application as an CoAgent, check out our [quickstart guide](/crewai-crews/quickstart/crewai).

---

# What is **CrewAI**?

**CrewAI is a lean, lightning-fast Python framework built entirely from scratch—completely independent of LangChain or other agent frameworks.**

CrewAI empowers developers with both high-level simplicity and precise low-level control, ideal for creating autonomous AI agents tailored to any scenario:

- **[CrewAI Crews](https://docs.crewai.com/guides/crews/first-crew)**: Optimize for autonomy and collaborative intelligence, enabling you to create AI teams where each agent has specific roles, tools, and goals.
- **[CrewAI Flows](https://docs.crewai.com/guides/flows/first-flow)**: Enable granular, event-driven control, single LLM calls for precise task orchestration and supports Crews natively.

With over 100,000 developers certified through our community courses, CrewAI is rapidly becoming the standard for enterprise-ready AI automation.

## How Crews Work

Just like a company has departments (Sales, Engineering, Marketing) working together under leadership to achieve business goals, CrewAI helps you create an organization of AI agents with specialized roles collaborating to accomplish complex tasks.

## CrewAI Framework Overview

| Component | Description | Key Features |
| --- | --- | --- |
| **Crew** | The top-level organization | • Manages AI agent teams<br>• Oversees workflows<br>• Ensures collaboration<br>• Delivers outcomes |
| **AI Agents** | Specialized team members | • Have specific roles (researcher, writer)<br>• Use designated tools<br>• Can delegate tasks<br>• Make autonomous decisions |
| **Process** | Workflow management system | • Defines collaboration patterns<br>• Controls task assignments<br>• Manages interactions<br>• Ensures efficient execution |
| **Tasks** | Individual assignments | • Have clear objectives<br>• Use specific tools<br>• Feed into larger process<br>• Produce actionable results |

### How It All Works Together

1. The **Crew** organizes the overall operation
2. **AI Agents** work on their specialized tasks
3. The **Process** ensures smooth collaboration
4. **Tasks** get completed to achieve the goal

## Key Features

## Role-Based Agents

Create specialized agents with defined roles, expertise, and goals - from researchers to analysts to writers

## Flexible Tools

Equip agents with custom tools and APIs to interact with external services and data sources

## Intelligent Collaboration

Agents work together, sharing insights and coordinating tasks to achieve complex objectives

## Task Management

Define sequential or parallel workflows, with agents automatically handling task dependencies

## How Flows Work

While Crews excel at autonomous collaboration, Flows provide structured automations, offering granular control over workflow execution. Flows ensure tasks are executed reliably, securely, and efficiently, handling conditional logic, loops, and dynamic state management with precision. Flows integrate seamlessly with Crews, enabling you to balance high autonomy with exacting control.

## CrewAI Framework Overview

| Component | Description | Key Features |
| --- | --- | --- |
| **Flow** | Structured workflow orchestration | • Manages execution paths<br>• Handles state transitions<br>• Controls task sequencing<br>• Ensures reliable execution |
| **Events** | Triggers for workflow actions | • Initiate specific processes<br>• Enable dynamic responses<br>• Support conditional branching<br>• Allow for real-time adaptation |
| **States** | Workflow execution contexts | • Maintain execution data<br>• Enable persistence<br>• Support resumability<br>• Ensure execution integrity |
| **Crew Support** | Enhances workflow automation | • Injects pockets of agency when needed<br>• Complements structured workflows<br>• Balances automation with intelligence<br>• Enables adaptive decision-making |

### [​](https://docs.crewai.com/introduction\#key-capabilities)  Key Capabilities

## Event-Driven Orchestration

Define precise execution paths responding dynamically to events

## Fine-Grained Control

Manage workflow states and conditional execution securely and efficiently

## Native Crew Integration

Effortlessly combine with Crews for enhanced autonomy and intelligence

## Deterministic Execution

Ensure predictable outcomes with explicit control flow and error handling

## When to Use Crews vs. Flows

Understanding when to use [Crews](https://docs.crewai.com/guides/crews/first-crew) versus [Flows](https://docs.crewai.com/guides/flows/first-flow) is key to maximizing the potential of CrewAI in your applications.

| Use Case | Recommended Approach | Why? |
| --- | --- | --- |
| **Open-ended research** | [Crews](https://docs.crewai.com/guides/crews/first-crew) | When tasks require creative thinking, exploration, and adaptation |
| **Content generation** | [Crews](https://docs.crewai.com/guides/crews/first-crew) | For collaborative creation of articles, reports, or marketing materials |
| **Decision workflows** | [Flows](https://docs.crewai.com/guides/flows/first-flow) | When you need predictable, auditable decision paths with precise control |
| **API orchestration** | [Flows](https://docs.crewai.com/guides/flows/first-flow) | For reliable integration with multiple external services in a specific sequence |
| **Hybrid applications** | Combined approach | Use [Flows](https://docs.crewai.com/guides/flows/first-flow) to orchestrate overall process with [Crews](https://docs.crewai.com/guides/crews/first-crew) handling complex subtasks |

### Decision Framework

- **Choose [Crews](https://docs.crewai.com/guides/crews/first-crew) when:** You need autonomous problem-solving, creative collaboration, or exploratory tasks
- **Choose [Flows](https://docs.crewai.com/guides/flows/first-flow) when:** You require deterministic outcomes, auditability, or precise control over execution
- **Combine both when:** Your application needs both structured processes and pockets of autonomous intelligence

## [​](https://docs.crewai.com/introduction\#why-choose-crewai%3F)  Why Choose CrewAI?

- 🧠 **Autonomous Operation**: Agents make intelligent decisions based on their roles and available tools
- 📝 **Natural Interaction**: Agents communicate and collaborate like human team members
- 🛠️ **Extensible Design**: Easy to add new tools, roles, and capabilities
- 🚀 **Production Ready**: Built for reliability and scalability in real-world applications
- 🔒 **Security-Focused**: Designed with enterprise security requirements in mind
- 💰 **Cost-Efficient**: Optimized to minimize token usage and API calls

---

## Overview of a Task

In the CrewAI framework, a `Task` is a specific assignment completed by an `Agent`.

Tasks provide all necessary details for execution, such as a description, the agent responsible, required tools, and more, facilitating a wide range of action complexities.

Tasks within CrewAI can be collaborative, requiring multiple agents to work together. This is managed through the task properties and orchestrated by the Crew’s process, enhancing teamwork and efficiency.

### [​](https://docs.crewai.com/concepts/tasks\#task-execution-flow)  Task Execution Flow

Tasks can be executed in two ways:

- **Sequential**: Tasks are executed in the order they are defined
- **Hierarchical**: Tasks are assigned to agents based on their roles and expertise

The execution flow is defined when creating the crew:

Code

Copy

```python
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential  # or Process.hierarchical
)

```

## [​](https://docs.crewai.com/concepts/tasks\#task-attributes)  Task Attributes

| Attribute | Parameters | Type | Description |
| --- | --- | --- | --- |
| **Description** | `description` | `str` | A clear, concise statement of what the task entails. |
| **Expected Output** | `expected_output` | `str` | A detailed description of what the task’s completion looks like. |
| **Name** _(optional)_ | `name` | `Optional[str]` | A name identifier for the task. |
| **Agent** _(optional)_ | `agent` | `Optional[BaseAgent]` | The agent responsible for executing the task. |
| **Tools** _(optional)_ | `tools` | `List[BaseTool]` | The tools/resources the agent is limited to use for this task. |
| **Context** _(optional)_ | `context` | `Optional[List["Task"]]` | Other tasks whose outputs will be used as context for this task. |
| **Async Execution** _(optional)_ | `async_execution` | `Optional[bool]` | Whether the task should be executed asynchronously. Defaults to False. |
| **Human Input** _(optional)_ | `human_input` | `Optional[bool]` | Whether the task should have a human review the final answer of the agent. Defaults to False. |
| **Config** _(optional)_ | `config` | `Optional[Dict[str, Any]]` | Task-specific configuration parameters. |
| **Output File** _(optional)_ | `output_file` | `Optional[str]` | File path for storing the task output. |
| **Output JSON** _(optional)_ | `output_json` | `Optional[Type[BaseModel]]` | A Pydantic model to structure the JSON output. |
| **Output Pydantic** _(optional)_ | `output_pydantic` | `Optional[Type[BaseModel]]` | A Pydantic model for task output. |
| **Callback** _(optional)_ | `callback` | `Optional[Any]` | Function/object to be executed after task completion. |

## [​](https://docs.crewai.com/concepts/tasks\#creating-tasks)  Creating Tasks

There are two ways to create tasks in CrewAI: using **YAML configuration (recommended)** or defining them **directly in code**.

### [​](https://docs.crewai.com/concepts/tasks\#yaml-configuration-recommended)  YAML Configuration (Recommended)

Using YAML configuration provides a cleaner, more maintainable way to define tasks. We strongly recommend using this approach to define tasks in your CrewAI projects.

After creating your CrewAI project as outlined in the [Installation](https://docs.crewai.com/installation) section, navigate to the `src/latest_ai_development/config/tasks.yaml` file and modify the template to match your specific task requirements.

Variables in your YAML files (like `{topic}`) will be replaced with values from your inputs when running the crew:

```python
crew.kickoff(inputs={'topic': 'AI Agents'})

```

Here’s an example of how to configure tasks using YAML:

tasks.yaml

Copy

````yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2025.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher

reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md

````

To use this YAML configuration in your code, create a crew class that inherits from `CrewBase`:

crew.py

Copy

```python
# src/latest_ai_development/crew.py

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

@CrewBase
class LatestAiDevelopmentCrew():
  """LatestAiDevelopment crew"""

  @agent
  def researcher(self) -> Agent:
    return Agent(
      config=self.agents_config['researcher'],
      verbose=True,
      tools=[SerperDevTool()]
    )

  @agent
  def reporting_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['reporting_analyst'],
      verbose=True
    )

  @task
  def research_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_task']
    )

  @task
  def reporting_task(self) -> Task:
    return Task(
      config=self.tasks_config['reporting_task']
    )

  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=[\
        self.researcher(),\
        self.reporting_analyst()\
      ],
      tasks=[\
        self.research_task(),\
        self.reporting_task()\
      ],
      process=Process.sequential
    )

```

The names you use in your YAML files ( `agents.yaml` and `tasks.yaml`) should match the method names in your Python code.

### [​](https://docs.crewai.com/concepts/tasks\#direct-code-definition-alternative)  Direct Code Definition (Alternative)

Alternatively, you can define tasks directly in your code without using YAML configuration:

````python
from crewai import Task

research_task = Task(
    description="""
        Conduct a thorough research about AI Agents.
        Make sure you find any interesting and relevant information given
        the current year is 2025.
    """,
    expected_output="""
        A list with 10 bullet points of the most relevant information about AI Agents
    """,
    agent=researcher
)

reporting_task = Task(
    description="""
        Review the context you got and expand each topic into a full section for a report.
        Make sure the report is detailed and contains any and all relevant information.
    """,
    expected_output="""
        A fully fledge reports with the mains topics, each with a full section of information.
        Formatted as markdown without '```'
    """,
    agent=reporting_analyst,
    output_file="report.md"
)

````

Directly specify an `agent` for assignment or let the `hierarchical` CrewAI’s process decide based on roles, availability, etc.

## [​](https://docs.crewai.com/concepts/tasks\#task-output)  Task Output

Understanding task outputs is crucial for building effective AI workflows. CrewAI provides a structured way to handle task results through the `TaskOutput` class, which supports multiple output formats and can be easily passed between tasks.

The output of a task in CrewAI framework is encapsulated within the `TaskOutput` class. This class provides a structured way to access results of a task, including various formats such as raw output, JSON, and Pydantic models.

By default, the `TaskOutput` will only include the `raw` output. A `TaskOutput` will only include the `pydantic` or `json_dict` output if the original `Task` object was configured with `output_pydantic` or `output_json`, respectively.

### [​](https://docs.crewai.com/concepts/tasks\#task-output-attributes)  Task Output Attributes

| Attribute | Parameters | Type | Description |
| --- | --- | --- | --- |
| **Description** | `description` | `str` | Description of the task. |
| **Summary** | `summary` | `Optional[str]` | Summary of the task, auto-generated from the first 10 words of the description. |
| **Raw** | `raw` | `str` | The raw output of the task. This is the default format for the output. |
| **Pydantic** | `pydantic` | `Optional[BaseModel]` | A Pydantic model object representing the structured output of the task. |
| **JSON Dict** | `json_dict` | `Optional[Dict[str, Any]]` | A dictionary representing the JSON output of the task. |
| **Agent** | `agent` | `str` | The agent that executed the task. |
| **Output Format** | `output_format` | `OutputFormat` | The format of the task output, with options including RAW, JSON, and Pydantic. The default is RAW. |

### [​](https://docs.crewai.com/concepts/tasks\#task-methods-and-properties)  Task Methods and Properties

| Method/Property | Description |
| --- | --- |
| **json** | Returns the JSON string representation of the task output if the output format is JSON. |
| **to\_dict** | Converts the JSON and Pydantic outputs to a dictionary. |
| **str** | Returns the string representation of the task output, prioritizing Pydantic, then JSON, then raw. |

### [​](https://docs.crewai.com/concepts/tasks\#accessing-task-outputs)  Accessing Task Outputs

Once a task has been executed, its output can be accessed through the `output` attribute of the `Task` object. The `TaskOutput` class provides various ways to interact with and present this output.

#### [​](https://docs.crewai.com/concepts/tasks\#example)  Example

Code

Copy

```python
# Example task
task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

# Execute the crew
crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()

# Accessing the task output
task_output = task.output

print(f"Task Description: {task_output.description}")
print(f"Task Summary: {task_output.summary}")
print(f"Raw Output: {task_output.raw}")
if task_output.json_dict:
    print(f"JSON Output: {json.dumps(task_output.json_dict, indent=2)}")
if task_output.pydantic:
    print(f"Pydantic Output: {task_output.pydantic}")

```

## [​](https://docs.crewai.com/concepts/tasks\#task-dependencies-and-context)  Task Dependencies and Context

Tasks can depend on the output of other tasks using the `context` attribute. For example:

```python
research_task = Task(
    description="Research the latest developments in AI",
    expected_output="A list of recent AI developments",
    agent=researcher
)

analysis_task = Task(
    description="Analyze the research findings and identify key trends",
    expected_output="Analysis report of AI trends",
    agent=analyst,
    context=[research_task]  # This task will wait for research_task to complete
)

```

## [​](https://docs.crewai.com/concepts/tasks\#task-guardrails)  Task Guardrails

Task guardrails provide a way to validate and transform task outputs before they
are passed to the next task. This feature helps ensure data quality and provides
feedback to agents when their output doesn’t meet specific criteria.

### [​](https://docs.crewai.com/concepts/tasks\#using-task-guardrails)  Using Task Guardrails

To add a guardrail to a task, provide a validation function through the `guardrail` parameter:

```python
from typing import Tuple, Union, Dict, Any

def validate_blog_content(result: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """Validate blog content meets requirements."""
    try:
        # Check word count
        word_count = len(result.split())
        if word_count > 200:
            return (False, {
                "error": "Blog content exceeds 200 words",
                "code": "WORD_COUNT_ERROR",
                "context": {"word_count": word_count}
            })

        # Additional validation logic here
        return (True, result.strip())
    except Exception as e:
        return (False, {
            "error": "Unexpected error during validation",
            "code": "SYSTEM_ERROR"
        })

blog_task = Task(
    description="Write a blog post about AI",
    expected_output="A blog post under 200 words",
    agent=blog_agent,
    guardrail=validate_blog_content  # Add the guardrail function
)

```

### [​](https://docs.crewai.com/concepts/tasks\#guardrail-function-requirements)  Guardrail Function Requirements

1. **Function Signature**:
   - Must accept exactly one parameter (the task output)
   - Should return a tuple of `(bool, Any)`
   - Type hints are recommended but optional
2. **Return Values**:
   - Success: Return `(True, validated_result)`
   - Failure: Return `(False, error_details)`

### [​](https://docs.crewai.com/concepts/tasks\#error-handling-best-practices)  Error Handling Best Practices

1. **Structured Error Responses**:

```python
def validate_with_context(result: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    try:
        # Main validation logic
        validated_data = perform_validation(result)
        return (True, validated_data)
    except ValidationError as e:
        return (False, {
            "error": str(e),
            "code": "VALIDATION_ERROR",
            "context": {"input": result}
        })
    except Exception as e:
        return (False, {
            "error": "Unexpected error",
            "code": "SYSTEM_ERROR"
        })

```

2. **Error Categories**:
   - Use specific error codes
   - Include relevant context
   - Provide actionable feedback
3. **Validation Chain**:

```python
from typing import Any, Dict, List, Tuple, Union

def complex_validation(result: str) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    """Chain multiple validation steps."""
    # Step 1: Basic validation
    if not result:
        return (False, {"error": "Empty result", "code": "EMPTY_INPUT"})

    # Step 2: Content validation
    try:
        validated = validate_content(result)
        if not validated:
            return (False, {"error": "Invalid content", "code": "CONTENT_ERROR"})

        # Step 3: Format validation
        formatted = format_output(validated)
        return (True, formatted)
    except Exception as e:
        return (False, {
            "error": str(e),
            "code": "VALIDATION_ERROR",
            "context": {"step": "content_validation"}
        })

```

### [​](https://docs.crewai.com/concepts/tasks\#handling-guardrail-results)  Handling Guardrail Results

When a guardrail returns `(False, error)`:

1. The error is sent back to the agent
2. The agent attempts to fix the issue
3. The process repeats until:
   - The guardrail returns `(True, result)`
   - Maximum retries are reached

Example with retry handling:

```python
from typing import Optional, Tuple, Union

def validate_json_output(result: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """Validate and parse JSON output."""
    try:
        # Try to parse as JSON
        data = json.loads(result)
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, {
            "error": "Invalid JSON format",
            "code": "JSON_ERROR",
            "context": {"line": e.lineno, "column": e.colno}
        })

task = Task(
    description="Generate a JSON report",
    expected_output="A valid JSON object",
    agent=analyst,
    guardrail=validate_json_output,
    max_retries=3  # Limit retry attempts
)

```

## [​](https://docs.crewai.com/concepts/tasks\#getting-structured-consistent-outputs-from-tasks)  Getting Structured Consistent Outputs from Tasks

It’s also important to note that the output of the final task of a crew becomes the final output of the actual crew itself.

### [​](https://docs.crewai.com/concepts/tasks\#using-output-pydantic)  Using `output_pydantic`

The `output_pydantic` property allows you to define a Pydantic model that the task output should conform to. This ensures that the output is not only structured but also validated according to the Pydantic model.

Here’s an example demonstrating how to use output\_pydantic:

Code

Copy

```python
import json

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

class Blog(BaseModel):
    title: str
    content: str

blog_agent = Agent(
    role="Blog Content Generator Agent",
    goal="Generate a blog title and content",
    backstory="""You are an expert content creator, skilled in crafting engaging and informative blog posts.""",
    verbose=False,
    allow_delegation=False,
    llm="gpt-4o",
)

task1 = Task(
    description="""Create a blog title and content on a given topic. Make sure the content is under 200 words.""",
    expected_output="A compelling blog title and well-written content.",
    agent=blog_agent,
    output_pydantic=Blog,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[blog_agent],
    tasks=[task1],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()

# Option 1: Accessing Properties Using Dictionary-Style Indexing
print("Accessing Properties - Option 1")
title = result["title"]
content = result["content"]
print("Title:", title)
print("Content:", content)

# Option 2: Accessing Properties Directly from the Pydantic Model
print("Accessing Properties - Option 2")
title = result.pydantic.title
content = result.pydantic.content
print("Title:", title)
print("Content:", content)

# Option 3: Accessing Properties Using the to_dict() Method
print("Accessing Properties - Option 3")
output_dict = result.to_dict()
title = output_dict["title"]
content = output_dict["content"]
print("Title:", title)
print("Content:", content)

# Option 4: Printing the Entire Blog Object
print("Accessing Properties - Option 5")
print("Blog:", result)

```

In this example:

- A Pydantic model Blog is defined with title and content fields.
- The task task1 uses the output\_pydantic property to specify that its output should conform to the Blog model.
- After executing the crew, you can access the structured output in multiple ways as shown.

#### [​](https://docs.crewai.com/concepts/tasks\#explanation-of-accessing-the-output)  Explanation of Accessing the Output

1. Dictionary-Style Indexing: You can directly access the fields using result\[“field\_name”\]. This works because the CrewOutput class implements the **getitem** method.
2. Directly from Pydantic Model: Access the attributes directly from the result.pydantic object.
3. Using to\_dict() Method: Convert the output to a dictionary and access the fields.
4. Printing the Entire Object: Simply print the result object to see the structured output.

### [​](https://docs.crewai.com/concepts/tasks\#using-output-json)  Using `output_json`

The `output_json` property allows you to define the expected output in JSON format. This ensures that the task’s output is a valid JSON structure that can be easily parsed and used in your application.

Here’s an example demonstrating how to use `output_json`:

Code

Copy

```python
import json

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel

# Define the Pydantic model for the blog
class Blog(BaseModel):
    title: str
    content: str

# Define the agent
blog_agent = Agent(
    role="Blog Content Generator Agent",
    goal="Generate a blog title and content",
    backstory="""You are an expert content creator, skilled in crafting engaging and informative blog posts.""",
    verbose=False,
    allow_delegation=False,
    llm="gpt-4o",
)

# Define the task with output_json set to the Blog model
task1 = Task(
    description="""Create a blog title and content on a given topic. Make sure the content is under 200 words.""",
    expected_output="A JSON object with 'title' and 'content' fields.",
    agent=blog_agent,
    output_json=Blog,
)

# Instantiate the crew with a sequential process
crew = Crew(
    agents=[blog_agent],
    tasks=[task1],
    verbose=True,
    process=Process.sequential,
)

# Kickoff the crew to execute the task
result = crew.kickoff()

# Option 1: Accessing Properties Using Dictionary-Style Indexing
print("Accessing Properties - Option 1")
title = result["title"]
content = result["content"]
print("Title:", title)
print("Content:", content)

# Option 2: Printing the Entire Blog Object
print("Accessing Properties - Option 2")
print("Blog:", result)

```

In this example:

- A Pydantic model Blog is defined with title and content fields, which is used to specify the structure of the JSON output.
- The task task1 uses the output\_json property to indicate that it expects a JSON output conforming to the Blog model.
- After executing the crew, you can access the structured JSON output in two ways as shown.

#### [​](https://docs.crewai.com/concepts/tasks\#explanation-of-accessing-the-output-2)  Explanation of Accessing the Output

1. Accessing Properties Using Dictionary-Style Indexing: You can access the fields directly using result\[“field\_name”\]. This is possible because the CrewOutput class implements the **getitem** method, allowing you to treat the output like a dictionary. In this option, we’re retrieving the title and content from the result.
2. Printing the Entire Blog Object: By printing result, you get the string representation of the CrewOutput object. Since the **str** method is implemented to return the JSON output, this will display the entire output as a formatted string representing the Blog object.

* * *

By using output\_pydantic or output\_json, you ensure that your tasks produce outputs in a consistent and structured format, making it easier to process and utilize the data within your application or across multiple tasks.

## [​](https://docs.crewai.com/concepts/tasks\#integrating-tools-with-tasks)  Integrating Tools with Tasks

Leverage tools from the [CrewAI Toolkit](https://github.com/joaomdmoura/crewai-tools) and [LangChain Tools](https://python.langchain.com/docs/integrations/tools) for enhanced task performance and agent interaction.

## [​](https://docs.crewai.com/concepts/tasks\#creating-a-task-with-tools)  Creating a Task with Tools

```python
import os
os.environ["OPENAI_API_KEY"] = "Your Key"
os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key

from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

research_agent = Agent(
  role='Researcher',
  goal='Find and summarize the latest AI news',
  backstory="""You're a researcher at a large company.
  You're responsible for analyzing data and providing insights
  to the business.""",
  verbose=True
)

# to perform a semantic search for a specified query from a text's content across the internet
search_tool = SerperDevTool()

task = Task(
  description='Find and summarize the latest AI news',
  expected_output='A bullet list summary of the top 5 most important AI news',
  agent=research_agent,
  tools=[search_tool]
)

crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
print(result)

```

This demonstrates how tasks with specific tools can override an agent’s default set for tailored task execution.

## [​](https://docs.crewai.com/concepts/tasks\#referring-to-other-tasks)  Referring to Other Tasks

In CrewAI, the output of one task is automatically relayed into the next one, but you can specifically define what tasks’ output, including multiple, should be used as context for another task.

This is useful when you have a task that depends on the output of another task that is not performed immediately after it. This is done through the `context` attribute of the task:

Code

Copy

```python
# ...

research_ai_task = Task(
    description="Research the latest developments in AI",
    expected_output="A list of recent AI developments",
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

research_ops_task = Task(
    description="Research the latest developments in AI Ops",
    expected_output="A list of recent AI Ops developments",
    async_execution=True,
    agent=research_agent,
    tools=[search_tool]
)

write_blog_task = Task(
    description="Write a full blog post about the importance of AI and its latest news",
    expected_output="Full blog post that is 4 paragraphs long",
    agent=writer_agent,
    context=[research_ai_task, research_ops_task]
)

#...

```

## [​](https://docs.crewai.com/concepts/tasks\#asynchronous-execution)  Asynchronous Execution

You can define a task to be executed asynchronously. This means that the crew will not wait for it to be completed to continue with the next task. This is useful for tasks that take a long time to be completed, or that are not crucial for the next tasks to be performed.

You can then use the `context` attribute to define in a future task that it should wait for the output of the asynchronous task to be completed.

Code

Copy

```python
#...

list_ideas = Task(
    description="List of 5 interesting ideas to explore for an article about AI.",
    expected_output="Bullet point list of 5 ideas for an article.",
    agent=researcher,
    async_execution=True # Will be executed asynchronously
)

list_important_history = Task(
    description="Research the history of AI and give me the 5 most important events.",
    expected_output="Bullet point list of 5 important events.",
    agent=researcher,
    async_execution=True # Will be executed asynchronously
)

write_article = Task(
    description="Write an article about AI, its history, and interesting ideas.",
    expected_output="A 4 paragraph article about AI.",
    agent=writer,
    context=[list_ideas, list_important_history] # Will wait for the output of the two tasks to be completed
)

#...

```

## [​](https://docs.crewai.com/concepts/tasks\#callback-mechanism)  Callback Mechanism

The callback function is executed after the task is completed, allowing for actions or notifications to be triggered based on the task’s outcome.

Code

Copy

```python
# ...

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw}
    """)

research_task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool],
    callback=callback_function
)

#...

```

## [​](https://docs.crewai.com/concepts/tasks\#accessing-a-specific-task-output)  Accessing a Specific Task Output

Once a crew finishes running, you can access the output of a specific task by using the `output` attribute of the task object:

Code

Copy

```python
# ...
task1 = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

#...

crew = Crew(
    agents=[research_agent],
    tasks=[task1, task2, task3],
    verbose=True
)

result = crew.kickoff()

# Returns a TaskOutput object with the description and results of the task
print(f"""
    Task completed!
    Task: {task1.output.description}
    Output: {task1.output.raw}
""")

```

## [​](https://docs.crewai.com/concepts/tasks\#tool-override-mechanism)  Tool Override Mechanism

Specifying tools in a task allows for dynamic adaptation of agent capabilities, emphasizing CrewAI’s flexibility.

## [​](https://docs.crewai.com/concepts/tasks\#error-handling-and-validation-mechanisms)  Error Handling and Validation Mechanisms

While creating and executing tasks, certain validation mechanisms are in place to ensure the robustness and reliability of task attributes. These include but are not limited to:

- Ensuring only one output type is set per task to maintain clear output expectations.
- Preventing the manual assignment of the `id` attribute to uphold the integrity of the unique identifier system.

These validations help in maintaining the consistency and reliability of task executions within the crewAI framework.

## [​](https://docs.crewai.com/concepts/tasks\#task-guardrails-2)  Task Guardrails

Task guardrails provide a powerful way to validate, transform, or filter task outputs before they are passed to the next task. Guardrails are optional functions that execute before the next task starts, allowing you to ensure that task outputs meet specific requirements or formats.

### [​](https://docs.crewai.com/concepts/tasks\#basic-usage)  Basic Usage

Code

Copy

```python
from typing import Tuple, Union
from crewai import Task

def validate_json_output(result: str) -> Tuple[bool, Union[dict, str]]:
    """Validate that the output is valid JSON."""
    try:
        json_data = json.loads(result)
        return (True, json_data)
    except json.JSONDecodeError:
        return (False, "Output must be valid JSON")

task = Task(
    description="Generate JSON data",
    expected_output="Valid JSON object",
    guardrail=validate_json_output
)

```

### [​](https://docs.crewai.com/concepts/tasks\#how-guardrails-work)  How Guardrails Work

1. **Optional Attribute**: Guardrails are an optional attribute at the task level, allowing you to add validation only where needed.
2. **Execution Timing**: The guardrail function is executed before the next task starts, ensuring valid data flow between tasks.
3. **Return Format**: Guardrails must return a tuple of `(success, data)`:

   - If `success` is `True`, `data` is the validated/transformed result
   - If `success` is `False`, `data` is the error message
4. **Result Routing**:

   - On success ( `True`), the result is automatically passed to the next task
   - On failure ( `False`), the error is sent back to the agent to generate a new answer

### [​](https://docs.crewai.com/concepts/tasks\#common-use-cases)  Common Use Cases

#### [​](https://docs.crewai.com/concepts/tasks\#data-format-validation)  Data Format Validation

Code

Copy

```python
def validate_email_format(result: str) -> Tuple[bool, Union[str, str]]:
    """Ensure the output contains a valid email address."""
    import re
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(email_pattern, result.strip()):
        return (True, result.strip())
    return (False, "Output must be a valid email address")

```

#### [​](https://docs.crewai.com/concepts/tasks\#content-filtering)  Content Filtering

Code

Copy

```python
def filter_sensitive_info(result: str) -> Tuple[bool, Union[str, str]]:
    """Remove or validate sensitive information."""
    sensitive_patterns = ['SSN:', 'password:', 'secret:']
    for pattern in sensitive_patterns:
        if pattern.lower() in result.lower():
            return (False, f"Output contains sensitive information ({pattern})")
    return (True, result)

```

#### [​](https://docs.crewai.com/concepts/tasks\#data-transformation)  Data Transformation

Code

Copy

```python
def normalize_phone_number(result: str) -> Tuple[bool, Union[str, str]]:
    """Ensure phone numbers are in a consistent format."""
    import re
    digits = re.sub(r'\D', '', result)
    if len(digits) == 10:
        formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return (True, formatted)
    return (False, "Output must be a 10-digit phone number")

```

### Advanced Features

#### Chaining Multiple Validations

Code

Copy

```python
def chain_validations(*validators):
    """Chain multiple validators together."""
    def combined_validator(result):
        for validator in validators:
            success, data = validator(result)
            if not success:
                return (False, data)
            result = data
        return (True, result)
    return combined_validator

# Usage
task = Task(
    description="Get user contact info",
    expected_output="Email and phone",
    guardrail=chain_validations(
        validate_email_format,
        filter_sensitive_info
    )
)

```

#### Custom Retry Logic

Code

Copy

```python
task = Task(
    description="Generate data",
    expected_output="Valid data",
    guardrail=validate_data,
    max_retries=5  # Override default retry limit
)

```

## Creating Directories when Saving Files

You can now specify if a task should create directories when saving its output to a file. This is particularly useful for organizing outputs and ensuring that file paths are correctly structured.

Code

Copy

```python
# ...

save_output_task = Task(
    description='Save the summarized AI news to a file',
    expected_output='File saved successfully',
    agent=research_agent,
    tools=[file_save_tool],
    output_file='outputs/ai_news_summary.txt',
    create_directory=True
)

#...

```
---

# Event Listeners

CrewAI provides a powerful event system that allows you to listen for and react to various events that occur during the execution of your Crew. This feature enables you to build custom integrations, monitoring solutions, logging systems, or any other functionality that needs to be triggered based on CrewAI’s internal events.

## How It Works

CrewAI uses an event bus architecture to emit events throughout the execution lifecycle. The event system is built on the following components:

1. **CrewAIEventsBus**: A singleton event bus that manages event registration and emission
2. **BaseEvent**: Base class for all events in the system
3. **BaseEventListener**: Abstract base class for creating custom event listeners

When specific actions occur in CrewAI (like a Crew starting execution, an Agent completing a task, or a tool being used), the system emits corresponding events. You can register handlers for these events to execute custom code when they occur.

## Creating a Custom Event Listener

To create a custom event listener, you need to:

1. Create a class that inherits from `BaseEventListener`
2. Implement the `setup_listeners` method
3. Register handlers for the events you’re interested in
4. Create an instance of your listener in the appropriate file

Here’s a simple example of a custom event listener class:

Copy

```python
from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionCompletedEvent,
)
from crewai.utilities.events.base_event_listener import BaseEventListener

class MyCustomListener(BaseEventListener):
    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            print(f"Crew '{event.crew_name}' has started execution!")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            print(f"Crew '{event.crew_name}' has completed execution!")
            print(f"Output: {event.output}")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event):
            print(f"Agent '{event.agent.role}' completed task")
            print(f"Output: {event.output}")

```

## Properly Registering Your Listener

Simply defining your listener class isn’t enough. You need to create an instance of it and ensure it’s imported in your application. This ensures that:

1. The event handlers are registered with the event bus
2. The listener instance remains in memory (not garbage collected)
3. The listener is active when events are emitted

### Option 1: Import and Instantiate in Your Crew or Flow Implementation

The most important thing is to create an instance of your listener in the file where your Crew or Flow is defined and executed:

#### For Crew-based Applications

Create and import your listener at the top of your Crew implementation file:

Copy

```python
# In your crew.py file
from crewai import Agent, Crew, Task
from my_listeners import MyCustomListener

# Create an instance of your listener
my_listener = MyCustomListener()

class MyCustomCrew:
    # Your crew implementation...

    def crew(self):
        return Crew(
            agents=[...],
            tasks=[...],
            # ...
        )

```

#### For Flow-based Applications

Create and import your listener at the top of your Flow implementation file:

Copy

```python
# In your main.py or flow.py file
from crewai.flow import Flow, listen, start
from my_listeners import MyCustomListener

# Create an instance of your listener
my_listener = MyCustomListener()

class MyCustomFlow(Flow):
    # Your flow implementation...

    @start()
    def first_step(self):
        # ...

```

This ensures that your listener is loaded and active when your Crew or Flow is executed.

### Option 2: Create a Package for Your Listeners

For a more structured approach, especially if you have multiple listeners:

1. Create a package for your listeners:

Copy

```
my_project/
  ├── listeners/
  │   ├── __init__.py
  │   ├── my_custom_listener.py
  │   └── another_listener.py

```

2. In `my_custom_listener.py`, define your listener class and create an instance:

Copy

```python
# my_custom_listener.py
from crewai.utilities.events.base_event_listener import BaseEventListener
# ... import events ...

class MyCustomListener(BaseEventListener):
    # ... implementation ...

# Create an instance of your listener
my_custom_listener = MyCustomListener()

```

3. In `__init__.py`, import the listener instances to ensure they’re loaded:

Copy

```python
# __init__.py
from .my_custom_listener import my_custom_listener
from .another_listener import another_listener

# Optionally export them if you need to access them elsewhere
__all__ = ['my_custom_listener', 'another_listener']

```

4. Import your listeners package in your Crew or Flow file:

Copy

```python
# In your crew.py or flow.py file
import my_project.listeners  # This loads all your listeners

class MyCustomCrew:
    # Your crew implementation...

```

This is exactly how CrewAI’s built-in `agentops_listener` is registered. In the CrewAI codebase, you’ll find:

Copy

```python
# src/crewai/utilities/events/third_party/__init__.py
from .agentops_listener import agentops_listener

```

This ensures the `agentops_listener` is loaded when the `crewai.utilities.events` package is imported.

## Available Event Types

CrewAI provides a wide range of events that you can listen for:

### Crew Events

- **CrewKickoffStartedEvent**: Emitted when a Crew starts execution
- **CrewKickoffCompletedEvent**: Emitted when a Crew completes execution
- **CrewKickoffFailedEvent**: Emitted when a Crew fails to complete execution
- **CrewTestStartedEvent**: Emitted when a Crew starts testing
- **CrewTestCompletedEvent**: Emitted when a Crew completes testing
- **CrewTestFailedEvent**: Emitted when a Crew fails to complete testing
- **CrewTrainStartedEvent**: Emitted when a Crew starts training
- **CrewTrainCompletedEvent**: Emitted when a Crew completes training
- **CrewTrainFailedEvent**: Emitted when a Crew fails to complete training

### Agent Events

- **AgentExecutionStartedEvent**: Emitted when an Agent starts executing a task
- **AgentExecutionCompletedEvent**: Emitted when an Agent completes executing a task
- **AgentExecutionErrorEvent**: Emitted when an Agent encounters an error during execution

### Task Events

- **TaskStartedEvent**: Emitted when a Task starts execution
- **TaskCompletedEvent**: Emitted when a Task completes execution
- **TaskFailedEvent**: Emitted when a Task fails to complete execution
- **TaskEvaluationEvent**: Emitted when a Task is evaluated

### Tool Usage Events

- **ToolUsageStartedEvent**: Emitted when a tool execution is started
- **ToolUsageFinishedEvent**: Emitted when a tool execution is completed
- **ToolUsageErrorEvent**: Emitted when a tool execution encounters an error
- **ToolValidateInputErrorEvent**: Emitted when a tool input validation encounters an error
- **ToolExecutionErrorEvent**: Emitted when a tool execution encounters an error
- **ToolSelectionErrorEvent**: Emitted when there’s an error selecting a tool

### Flow Events

- **FlowCreatedEvent**: Emitted when a Flow is created
- **FlowStartedEvent**: Emitted when a Flow starts execution
- **FlowFinishedEvent**: Emitted when a Flow completes execution
- **FlowPlotEvent**: Emitted when a Flow is plotted
- **MethodExecutionStartedEvent**: Emitted when a Flow method starts execution
- **MethodExecutionFinishedEvent**: Emitted when a Flow method completes execution
- **MethodExecutionFailedEvent**: Emitted when a Flow method fails to complete execution

### [​](https://docs.crewai.com/concepts/event-listener\#llm-events)  LLM Events

- **LLMCallStartedEvent**: Emitted when an LLM call starts
- **LLMCallCompletedEvent**: Emitted when an LLM call completes
- **LLMCallFailedEvent**: Emitted when an LLM call fails
- **LLMStreamChunkEvent**: Emitted for each chunk received during streaming LLM responses

## Event Handler Structure

Each event handler receives two parameters:

1. **source**: The object that emitted the event
2. **event**: The event instance, containing event-specific data

The structure of the event object depends on the event type, but all events inherit from `BaseEvent` and include:

- **timestamp**: The time when the event was emitted
- **type**: A string identifier for the event type

Additional fields vary by event type. For example, `CrewKickoffCompletedEvent` includes `crew_name` and `output` fields.

## Real-World Example: Integration with AgentOps

CrewAI includes an example of a third-party integration with [AgentOps](https://github.com/AgentOps-AI/agentops), a monitoring and observability platform for AI agents. Here’s how it’s implemented:

Copy

```python
from typing import Optional

from crewai.utilities.events import (
    CrewKickoffCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.crew_events import CrewKickoffStartedEvent
from crewai.utilities.events.task_events import TaskEvaluationEvent

try:
    import agentops
    AGENTOPS_INSTALLED = True
except ImportError:
    AGENTOPS_INSTALLED = False

class AgentOpsListener(BaseEventListener):
    tool_event: Optional["agentops.ToolEvent"] = None
    session: Optional["agentops.Session"] = None

    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        if not AGENTOPS_INSTALLED:
            return

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_kickoff_started(source, event: CrewKickoffStartedEvent):
            self.session = agentops.init()
            for agent in source.agents:
                if self.session:
                    self.session.create_agent(
                        name=agent.role,
                        agent_id=str(agent.id),
                    )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_completed(source, event: CrewKickoffCompletedEvent):
            if self.session:
                self.session.end_session(
                    end_state="Success",
                    end_state_reason="Finished Execution",
                )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            self.tool_event = agentops.ToolEvent(name=event.tool_name)
            if self.session:
                self.session.record(self.tool_event)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            agentops.ErrorEvent(exception=event.error, trigger_event=self.tool_event)

```

This listener initializes an AgentOps session when a Crew starts, registers agents with AgentOps, tracks tool usage, and ends the session when the Crew completes.

The AgentOps listener is registered in CrewAI’s event system through the import in `src/crewai/utilities/events/third_party/__init__.py`:

Copy

```python
from .agentops_listener import agentops_listener

```

This ensures the `agentops_listener` is loaded when the `crewai.utilities.events` package is imported.

## [​](https://docs.crewai.com/concepts/event-listener\#advanced-usage%3A-scoped-handlers)  Advanced Usage: Scoped Handlers

For temporary event handling (useful for testing or specific operations), you can use the `scoped_handlers` context manager:

Copy

```python
from crewai.utilities.events import crewai_event_bus, CrewKickoffStartedEvent

with crewai_event_bus.scoped_handlers():
    @crewai_event_bus.on(CrewKickoffStartedEvent)
    def temp_handler(source, event):
        print("This handler only exists within this context")

    # Do something that emits events

# Outside the context, the temporary handler is removed

```

## [​](https://docs.crewai.com/concepts/event-listener\#use-cases)  Use Cases

Event listeners can be used for a variety of purposes:

1. **Logging and Monitoring**: Track the execution of your Crew and log important events
2. **Analytics**: Collect data about your Crew’s performance and behavior
3. **Debugging**: Set up temporary listeners to debug specific issues
4. **Integration**: Connect CrewAI with external systems like monitoring platforms, databases, or notification services
5. **Custom Behavior**: Trigger custom actions based on specific events

## [​](https://docs.crewai.com/concepts/event-listener\#best-practices)  Best Practices

1. **Keep Handlers Light**: Event handlers should be lightweight and avoid blocking operations
2. **Error Handling**: Include proper error handling in your event handlers to prevent exceptions from affecting the main execution
3. **Cleanup**: If your listener allocates resources, ensure they’re properly cleaned up
4. **Selective Listening**: Only listen for events you actually need to handle
5. **Testing**: Test your event listeners in isolation to ensure they behave as expected

By leveraging CrewAI’s event system, you can extend its functionality and integrate it seamlessly with your existing infrastructure.

Was this page helpful?

YesNo

[Tools](https://docs.crewai.com/concepts/tools) [Using LangChain Tools](https://docs.crewai.com/concepts/langchain-tools)

On this page

- [Event Listeners](https://docs.crewai.com/concepts/event-listener#event-listeners)
- [How It Works](https://docs.crewai.com/concepts/event-listener#how-it-works)
- [Creating a Custom Event Listener](https://docs.crewai.com/concepts/event-listener#creating-a-custom-event-listener)
- [Properly Registering Your Listener](https://docs.crewai.com/concepts/event-listener#properly-registering-your-listener)
- [Option 1: Import and Instantiate in Your Crew or Flow Implementation](https://docs.crewai.com/concepts/event-listener#option-1%3A-import-and-instantiate-in-your-crew-or-flow-implementation)
- [For Crew-based Applications](https://docs.crewai.com/concepts/event-listener#for-crew-based-applications)
- [For Flow-based Applications](https://docs.crewai.com/concepts/event-listener#for-flow-based-applications)
- [Option 2: Create a Package for Your Listeners](https://docs.crewai.com/concepts/event-listener#option-2%3A-create-a-package-for-your-listeners)
- [Available Event Types](https://docs.crewai.com/concepts/event-listener#available-event-types)
- [Crew Events](https://docs.crewai.com/concepts/event-listener#crew-events)
- [Agent Events](https://docs.crewai.com/concepts/event-listener#agent-events)
- [Task Events](https://docs.crewai.com/concepts/event-listener#task-events)
- [Tool Usage Events](https://docs.crewai.com/concepts/event-listener#tool-usage-events)
- [Flow Events](https://docs.crewai.com/concepts/event-listener#flow-events)
- [LLM Events](https://docs.crewai.com/concepts/event-listener#llm-events)
- [Event Handler Structure](https://docs.crewai.com/concepts/event-listener#event-handler-structure)
- [Real-World Example: Integration with AgentOps](https://docs.crewai.com/concepts/event-listener#real-world-example%3A-integration-with-agentops)
- [Advanced Usage: Scoped Handlers](https://docs.crewai.com/concepts/event-listener#advanced-usage%3A-scoped-handlers)
- [Use Cases](https://docs.crewai.com/concepts/event-listener#use-cases)
- [Best Practices](https://docs.crewai.com/concepts/event-listener#best-practices)

---

# Collaboration Fundamentals

Collaboration in CrewAI is fundamental, enabling agents to combine their skills, share information, and assist each other in task execution, embodying a truly cooperative ecosystem.

- **Information Sharing**: Ensures all agents are well-informed and can contribute effectively by sharing data and findings.
- **Task Assistance**: Allows agents to seek help from peers with the required expertise for specific tasks.
- **Resource Allocation**: Optimizes task execution through the efficient distribution and sharing of resources among agents.

## [​](https://docs.crewai.com/concepts/collaboration\#enhanced-attributes-for-improved-collaboration)  Enhanced Attributes for Improved Collaboration

The `Crew` class has been enriched with several attributes to support advanced functionalities:

| Feature | Description |
| --- | --- |
| **Language Model Management** ( `manager_llm`, `function_calling_llm`) | Manages language models for executing tasks and tools. `manager_llm` is required for hierarchical processes, while `function_calling_llm` is optional with a default value for streamlined interactions. |
| **Custom Manager Agent** ( `manager_agent`) | Specifies a custom agent as the manager, replacing the default CrewAI manager. |
| **Process Flow** ( `process`) | Defines execution logic (e.g., sequential, hierarchical) for task distribution. |
| **Verbose Logging** ( `verbose`) | Provides detailed logging for monitoring and debugging. Accepts integer and boolean values to control verbosity level. |
| **Rate Limiting** ( `max_rpm`) | Limits requests per minute to optimize resource usage. Setting guidelines depend on task complexity and load. |
| **Internationalization / Customization** ( `language`, `prompt_file`) | Supports prompt customization for global usability. [Example of file](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/translations/en.json) |
| **Execution and Output Handling** ( `full_output`) | Controls output granularity, distinguishing between full and final outputs. |
| **Callback and Telemetry** ( `step_callback`, `task_callback`) | Enables step-wise and task-level execution monitoring and telemetry for performance analytics. |
| **Crew Sharing** ( `share_crew`) | Allows sharing crew data with CrewAI for model improvement. Privacy implications and benefits should be considered. |
| **Usage Metrics** ( `usage_metrics`) | Logs all LLM usage metrics during task execution for performance insights. |
| **Memory Usage** ( `memory`) | Enables memory for storing execution history, aiding in agent learning and task efficiency. |
| **Embedder Configuration** ( `embedder`) | Configures the embedder for language understanding and generation, with support for provider customization. |
| **Cache Management** ( `cache`) | Specifies whether to cache tool execution results, enhancing performance. |
| **Output Logging** ( `output_log_file`) | Defines the file path for logging crew execution output. |
| **Planning Mode** ( `planning`) | Enables action planning before task execution. Set `planning=True` to activate. |
| **Replay Feature** ( `replay`) | Provides CLI for listing tasks from the last run and replaying from specific tasks, aiding in task management and troubleshooting. |

## [​](https://docs.crewai.com/concepts/collaboration\#delegation-dividing-to-conquer)  Delegation (Dividing to Conquer)

Delegation enhances functionality by allowing agents to intelligently assign tasks or seek help, thereby amplifying the crew’s overall capability.

## [​](https://docs.crewai.com/concepts/collaboration\#implementing-collaboration-and-delegation)  Implementing Collaboration and Delegation

Setting up a crew involves defining the roles and capabilities of each agent. CrewAI seamlessly manages their interactions, ensuring efficient collaboration and delegation, with enhanced customization and monitoring features to adapt to various operational needs.

## [​](https://docs.crewai.com/concepts/collaboration\#example-scenario)  Example Scenario

Consider a crew with a researcher agent tasked with data gathering and a writer agent responsible for compiling reports. The integration of advanced language model management and process flow attributes allows for more sophisticated interactions, such as the writer delegating complex research tasks to the researcher or querying specific information, thereby facilitating a seamless workflow.

## [​](https://docs.crewai.com/concepts/collaboration\#conclusion)  Conclusion

The integration of advanced attributes and functionalities into the CrewAI framework significantly enriches the agent collaboration ecosystem. These enhancements not only simplify interactions but also offer unprecedented flexibility and control, paving the way for sophisticated AI-driven solutions capable of tackling complex tasks through intelligent collaboration and delegation.

---

[CrewAI home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)](https://docs.crewai.com/)

Search CrewAI docs

Ctrl K

Search...

Navigation

Core Concepts

Agents

[Get Started](https://docs.crewai.com/introduction) [Examples](https://docs.crewai.com/examples/example)

## [​](https://docs.crewai.com/concepts/agents\#overview-of-an-agent)  Overview of an Agent

In the CrewAI framework, an `Agent` is an autonomous unit that can:

- Perform specific tasks
- Make decisions based on its role and goal
- Use tools to accomplish objectives
- Communicate and collaborate with other agents
- Maintain memory of interactions
- Delegate tasks when allowed

Think of an agent as a specialized team member with specific skills, expertise, and responsibilities. For example, a `Researcher` agent might excel at gathering and analyzing information, while a `Writer` agent might be better at creating content.

## [​](https://docs.crewai.com/concepts/agents\#agent-attributes)  Agent Attributes

| Attribute | Parameter | Type | Description |
| --- | --- | --- | --- |
| **Role** | `role` | `str` | Defines the agent’s function and expertise within the crew. |
| **Goal** | `goal` | `str` | The individual objective that guides the agent’s decision-making. |
| **Backstory** | `backstory` | `str` | Provides context and personality to the agent, enriching interactions. |
| **LLM** _(optional)_ | `llm` | `Union[str, LLM, Any]` | Language model that powers the agent. Defaults to the model specified in `OPENAI_MODEL_NAME` or “gpt-4”. |
| **Tools** _(optional)_ | `tools` | `List[BaseTool]` | Capabilities or functions available to the agent. Defaults to an empty list. |
| **Function Calling LLM** _(optional)_ | `function_calling_llm` | `Optional[Any]` | Language model for tool calling, overrides crew’s LLM if specified. |
| **Max Iterations** _(optional)_ | `max_iter` | `int` | Maximum iterations before the agent must provide its best answer. Default is 20. |
| **Max RPM** _(optional)_ | `max_rpm` | `Optional[int]` | Maximum requests per minute to avoid rate limits. |
| **Max Execution Time** _(optional)_ | `max_execution_time` | `Optional[int]` | Maximum time (in seconds) for task execution. |
| **Memory** _(optional)_ | `memory` | `bool` | Whether the agent should maintain memory of interactions. Default is True. |
| **Verbose** _(optional)_ | `verbose` | `bool` | Enable detailed execution logs for debugging. Default is False. |
| **Allow Delegation** _(optional)_ | `allow_delegation` | `bool` | Allow the agent to delegate tasks to other agents. Default is False. |
| **Step Callback** _(optional)_ | `step_callback` | `Optional[Any]` | Function called after each agent step, overrides crew callback. |
| **Cache** _(optional)_ | `cache` | `bool` | Enable caching for tool usage. Default is True. |
| **System Template** _(optional)_ | `system_template` | `Optional[str]` | Custom system prompt template for the agent. |
| **Prompt Template** _(optional)_ | `prompt_template` | `Optional[str]` | Custom prompt template for the agent. |
| **Response Template** _(optional)_ | `response_template` | `Optional[str]` | Custom response template for the agent. |
| **Allow Code Execution** _(optional)_ | `allow_code_execution` | `Optional[bool]` | Enable code execution for the agent. Default is False. |
| **Max Retry Limit** _(optional)_ | `max_retry_limit` | `int` | Maximum number of retries when an error occurs. Default is 2. |
| **Respect Context Window** _(optional)_ | `respect_context_window` | `bool` | Keep messages under context window size by summarizing. Default is True. |
| **Code Execution Mode** _(optional)_ | `code_execution_mode` | `Literal["safe", "unsafe"]` | Mode for code execution: ‘safe’ (using Docker) or ‘unsafe’ (direct). Default is ‘safe’. |
| **Embedder** _(optional)_ | `embedder` | `Optional[Dict[str, Any]]` | Configuration for the embedder used by the agent. |
| **Knowledge Sources** _(optional)_ | `knowledge_sources` | `Optional[List[BaseKnowledgeSource]]` | Knowledge sources available to the agent. |
| **Use System Prompt** _(optional)_ | `use_system_prompt` | `Optional[bool]` | Whether to use system prompt (for o1 model support). Default is True. |

## [​](https://docs.crewai.com/concepts/agents\#creating-agents)  Creating Agents

There are two ways to create agents in CrewAI: using **YAML configuration (recommended)** or defining them **directly in code**.

### [​](https://docs.crewai.com/concepts/agents\#yaml-configuration-recommended)  YAML Configuration (Recommended)

Using YAML configuration provides a cleaner, more maintainable way to define agents. We strongly recommend using this approach in your CrewAI projects.

After creating your CrewAI project as outlined in the [Installation](https://docs.crewai.com/installation) section, navigate to the `src/latest_ai_development/config/agents.yaml` file and modify the template to match your requirements.

Variables in your YAML files (like `{topic}`) will be replaced with values from your inputs when running the crew:

Code

Copy

```python
crew.kickoff(inputs={'topic': 'AI Agents'})

```

Here’s an example of how to configure agents using YAML:

agents.yaml

Copy

```yaml
# src/latest_ai_development/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.

```

To use this YAML configuration in your code, create a crew class that inherits from `CrewBase`:

Code

Copy

```python
# src/latest_ai_development/crew.py
from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, crew
from crewai_tools import SerperDevTool

@CrewBase
class LatestAiDevelopmentCrew():
  """LatestAiDevelopment crew"""

  agents_config = "config/agents.yaml"

  @agent
  def researcher(self) -> Agent:
    return Agent(
      config=self.agents_config['researcher'],
      verbose=True,
      tools=[SerperDevTool()]
    )

  @agent
  def reporting_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['reporting_analyst'],
      verbose=True
    )

```

The names you use in your YAML files ( `agents.yaml`) should match the method names in your Python code.

### [​](https://docs.crewai.com/concepts/agents\#direct-code-definition)  Direct Code Definition

You can create agents directly in code by instantiating the `Agent` class. Here’s a comprehensive example showing all available parameters:

Code

Copy

```python
from crewai import Agent
from crewai_tools import SerperDevTool

# Create an agent with all available parameters
agent = Agent(
    role="Senior Data Scientist",
    goal="Analyze and interpret complex datasets to provide actionable insights",
    backstory="With over 10 years of experience in data science and machine learning, "
              "you excel at finding patterns in complex datasets.",
    llm="gpt-4",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    memory=True,  # Default: True
    verbose=False,  # Default: False
    allow_delegation=False,  # Default: False
    max_iter=20,  # Default: 20 iterations
    max_rpm=None,  # Optional: Rate limit for API calls
    max_execution_time=None,  # Optional: Maximum execution time in seconds
    max_retry_limit=2,  # Default: 2 retries on error
    allow_code_execution=False,  # Default: False
    code_execution_mode="safe",  # Default: "safe" (options: "safe", "unsafe")
    respect_context_window=True,  # Default: True
    use_system_prompt=True,  # Default: True
    tools=[SerperDevTool()],  # Optional: List of tools
    knowledge_sources=None,  # Optional: List of knowledge sources
    embedder=None,  # Optional: Custom embedder configuration
    system_template=None,  # Optional: Custom system prompt template
    prompt_template=None,  # Optional: Custom prompt template
    response_template=None,  # Optional: Custom response template
    step_callback=None,  # Optional: Callback function for monitoring
)

```

Let’s break down some key parameter combinations for common use cases:

#### [​](https://docs.crewai.com/concepts/agents\#basic-research-agent)  Basic Research Agent

Code

Copy

```python
research_agent = Agent(
    role="Research Analyst",
    goal="Find and summarize information about specific topics",
    backstory="You are an experienced researcher with attention to detail",
    tools=[SerperDevTool()],
    verbose=True  # Enable logging for debugging
)

```

#### [​](https://docs.crewai.com/concepts/agents\#code-development-agent)  Code Development Agent

Code

Copy

```python
dev_agent = Agent(
    role="Senior Python Developer",
    goal="Write and debug Python code",
    backstory="Expert Python developer with 10 years of experience",
    allow_code_execution=True,
    code_execution_mode="safe",  # Uses Docker for safety
    max_execution_time=300,  # 5-minute timeout
    max_retry_limit=3  # More retries for complex code tasks
)

```

#### [​](https://docs.crewai.com/concepts/agents\#long-running-analysis-agent)  Long-Running Analysis Agent

Code

Copy

```python
analysis_agent = Agent(
    role="Data Analyst",
    goal="Perform deep analysis of large datasets",
    backstory="Specialized in big data analysis and pattern recognition",
    memory=True,
    respect_context_window=True,
    max_rpm=10,  # Limit API calls
    function_calling_llm="gpt-4o-mini"  # Cheaper model for tool calls
)

```

#### [​](https://docs.crewai.com/concepts/agents\#custom-template-agent)  Custom Template Agent

Code

Copy

```python
custom_agent = Agent(
    role="Customer Service Representative",
    goal="Assist customers with their inquiries",
    backstory="Experienced in customer support with a focus on satisfaction",
    system_template="""<|start_header_id|>system<|end_header_id|>
                        {{ .System }}<|eot_id|>""",
    prompt_template="""<|start_header_id|>user<|end_header_id|>
                        {{ .Prompt }}<|eot_id|>""",
    response_template="""<|start_header_id|>assistant<|end_header_id|>
                        {{ .Response }}<|eot_id|>""",
)

```

### [​](https://docs.crewai.com/concepts/agents\#parameter-details)  Parameter Details

#### [​](https://docs.crewai.com/concepts/agents\#critical-parameters)  Critical Parameters

- `role`, `goal`, and `backstory` are required and shape the agent’s behavior
- `llm` determines the language model used (default: OpenAI’s GPT-4)

#### [​](https://docs.crewai.com/concepts/agents\#memory-and-context)  Memory and Context

- `memory`: Enable to maintain conversation history
- `respect_context_window`: Prevents token limit issues
- `knowledge_sources`: Add domain-specific knowledge bases

#### [​](https://docs.crewai.com/concepts/agents\#execution-control)  Execution Control

- `max_iter`: Maximum attempts before giving best answer
- `max_execution_time`: Timeout in seconds
- `max_rpm`: Rate limiting for API calls
- `max_retry_limit`: Retries on error

#### [​](https://docs.crewai.com/concepts/agents\#code-execution)  Code Execution

- `allow_code_execution`: Must be True to run code
- `code_execution_mode`:

  - `"safe"`: Uses Docker (recommended for production)
  - `"unsafe"`: Direct execution (use only in trusted environments)

#### [​](https://docs.crewai.com/concepts/agents\#templates)  Templates

- `system_template`: Defines agent’s core behavior
- `prompt_template`: Structures input format
- `response_template`: Formats agent responses

When using custom templates, you can use variables like `{role}`, `{goal}`, and `{input}` in your templates. These will be automatically populated during execution.

## [​](https://docs.crewai.com/concepts/agents\#agent-tools)  Agent Tools

Agents can be equipped with various tools to enhance their capabilities. CrewAI supports tools from:

- [CrewAI Toolkit](https://github.com/joaomdmoura/crewai-tools)
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools)

Here’s how to add tools to an agent:

Code

Copy

```python
from crewai import Agent
from crewai_tools import SerperDevTool, WikipediaTools

# Create tools
search_tool = SerperDevTool()
wiki_tool = WikipediaTools()

# Add tools to agent
researcher = Agent(
    role="AI Technology Researcher",
    goal="Research the latest AI developments",
    tools=[search_tool, wiki_tool],
    verbose=True
)

```

## [​](https://docs.crewai.com/concepts/agents\#agent-memory-and-context)  Agent Memory and Context

Agents can maintain memory of their interactions and use context from previous tasks. This is particularly useful for complex workflows where information needs to be retained across multiple tasks.

Code

Copy

```python
from crewai import Agent

analyst = Agent(
    role="Data Analyst",
    goal="Analyze and remember complex data patterns",
    memory=True,  # Enable memory
    verbose=True
)

```

When `memory` is enabled, the agent will maintain context across multiple interactions, improving its ability to handle complex, multi-step tasks.

## [​](https://docs.crewai.com/concepts/agents\#important-considerations-and-best-practices)  Important Considerations and Best Practices

### [​](https://docs.crewai.com/concepts/agents\#security-and-code-execution)  Security and Code Execution

- When using `allow_code_execution`, be cautious with user input and always validate it
- Use `code_execution_mode: "safe"` (Docker) in production environments
- Consider setting appropriate `max_execution_time` limits to prevent infinite loops

### [​](https://docs.crewai.com/concepts/agents\#performance-optimization)  Performance Optimization

- Use `respect_context_window: true` to prevent token limit issues
- Set appropriate `max_rpm` to avoid rate limiting
- Enable `cache: true` to improve performance for repetitive tasks
- Adjust `max_iter` and `max_retry_limit` based on task complexity

### [​](https://docs.crewai.com/concepts/agents\#memory-and-context-management)  Memory and Context Management

- Use `memory: true` for tasks requiring historical context
- Leverage `knowledge_sources` for domain-specific information
- Configure `embedder_config` when using custom embedding models
- Use custom templates ( `system_template`, `prompt_template`, `response_template`) for fine-grained control over agent behavior

### [​](https://docs.crewai.com/concepts/agents\#agent-collaboration)  Agent Collaboration

- Enable `allow_delegation: true` when agents need to work together
- Use `step_callback` to monitor and log agent interactions
- Consider using different LLMs for different purposes:
  - Main `llm` for complex reasoning
  - `function_calling_llm` for efficient tool usage

### [​](https://docs.crewai.com/concepts/agents\#model-compatibility)  Model Compatibility

- Set `use_system_prompt: false` for older models that don’t support system messages
- Ensure your chosen `llm` supports the features you need (like function calling)

## [​](https://docs.crewai.com/concepts/agents\#troubleshooting-common-issues)  Troubleshooting Common Issues

1. **Rate Limiting**: If you’re hitting API rate limits:
   - Implement appropriate `max_rpm`
   - Use caching for repetitive operations
   - Consider batching requests
2. **Context Window Errors**: If you’re exceeding context limits:
   - Enable `respect_context_window`
   - Use more efficient prompts
   - Clear agent memory periodically
3. **Code Execution Issues**: If code execution fails:
   - Verify Docker is installed for safe mode
   - Check execution permissions
   - Review code sandbox settings
4. **Memory Issues**: If agent responses seem inconsistent:
   - Verify memory is enabled
   - Check knowledge source configuration
   - Review conversation history management

Remember that agents are most effective when configured according to their specific use case. Take time to understand your requirements and adjust these parameters accordingly.

---

[CrewAI home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)](https://docs.crewai.com/)

Search CrewAI docs

Ctrl K

Search...

Navigation

Core Concepts

CLI

[Get Started](https://docs.crewai.com/introduction) [Examples](https://docs.crewai.com/examples/example)

# [​](https://docs.crewai.com/concepts/cli\#crewai-cli-documentation)  CrewAI CLI Documentation

The CrewAI CLI provides a set of commands to interact with CrewAI, allowing you to create, train, run, and manage crews & flows.

## [​](https://docs.crewai.com/concepts/cli\#installation)  Installation

To use the CrewAI CLI, make sure you have CrewAI installed:

Terminal

Copy

```shell
pip install crewai

```

## [​](https://docs.crewai.com/concepts/cli\#basic-usage)  Basic Usage

The basic structure of a CrewAI CLI command is:

Terminal

Copy

```shell
crewai [COMMAND] [OPTIONS] [ARGUMENTS]

```

## [​](https://docs.crewai.com/concepts/cli\#available-commands)  Available Commands

### [​](https://docs.crewai.com/concepts/cli\#1-create)  1\. Create

Create a new crew or flow.

Terminal

Copy

```shell
crewai create [OPTIONS] TYPE NAME

```

- `TYPE`: Choose between “crew” or “flow”
- `NAME`: Name of the crew or flow

Example:

Terminal

Copy

```shell
crewai create crew my_new_crew
crewai create flow my_new_flow

```

### [​](https://docs.crewai.com/concepts/cli\#2-version)  2\. Version

Show the installed version of CrewAI.

Terminal

Copy

```shell
crewai version [OPTIONS]

```

- `--tools`: (Optional) Show the installed version of CrewAI tools

Example:

Terminal

Copy

```shell
crewai version
crewai version --tools

```

### [​](https://docs.crewai.com/concepts/cli\#3-train)  3\. Train

Train the crew for a specified number of iterations.

Terminal

Copy

```shell
crewai train [OPTIONS]

```

- `-n, --n_iterations INTEGER`: Number of iterations to train the crew (default: 5)
- `-f, --filename TEXT`: Path to a custom file for training (default: “trained\_agents\_data.pkl”)

Example:

Terminal

Copy

```shell
crewai train -n 10 -f my_training_data.pkl

```

### [​](https://docs.crewai.com/concepts/cli\#4-replay)  4\. Replay

Replay the crew execution from a specific task.

Terminal

Copy

```shell
crewai replay [OPTIONS]

```

- `-t, --task_id TEXT`: Replay the crew from this task ID, including all subsequent tasks

Example:

Terminal

Copy

```shell
crewai replay -t task_123456

```

### [​](https://docs.crewai.com/concepts/cli\#5-log-tasks-outputs)  5\. Log-tasks-outputs

Retrieve your latest crew.kickoff() task outputs.

Terminal

Copy

```shell
crewai log-tasks-outputs

```

### [​](https://docs.crewai.com/concepts/cli\#6-reset-memories)  6\. Reset-memories

Reset the crew memories (long, short, entity, latest\_crew\_kickoff\_outputs).

Terminal

Copy

```shell
crewai reset-memories [OPTIONS]

```

- `-l, --long`: Reset LONG TERM memory
- `-s, --short`: Reset SHORT TERM memory
- `-e, --entities`: Reset ENTITIES memory
- `-k, --kickoff-outputs`: Reset LATEST KICKOFF TASK OUTPUTS
- `-a, --all`: Reset ALL memories

Example:

Terminal

Copy

```shell
crewai reset-memories --long --short
crewai reset-memories --all

```

### [​](https://docs.crewai.com/concepts/cli\#7-test)  7\. Test

Test the crew and evaluate the results.

Terminal

Copy

```shell
crewai test [OPTIONS]

```

- `-n, --n_iterations INTEGER`: Number of iterations to test the crew (default: 3)
- `-m, --model TEXT`: LLM Model to run the tests on the Crew (default: “gpt-4o-mini”)

Example:

Terminal

Copy

```shell
crewai test -n 5 -m gpt-3.5-turbo

```

### [​](https://docs.crewai.com/concepts/cli\#8-run)  8\. Run

Run the crew or flow.

Terminal

Copy

```shell
crewai run

```

Starting from version 0.103.0, the `crewai run` command can be used to run both standard crews and flows. For flows, it automatically detects the type from pyproject.toml and runs the appropriate command. This is now the recommended way to run both crews and flows.

Make sure to run these commands from the directory where your CrewAI project is set up.
Some commands may require additional configuration or setup within your project structure.

### [​](https://docs.crewai.com/concepts/cli\#9-chat)  9\. Chat

Starting in version `0.98.0`, when you run the `crewai chat` command, you start an interactive session with your crew. The AI assistant will guide you by asking for necessary inputs to execute the crew. Once all inputs are provided, the crew will execute its tasks.

After receiving the results, you can continue interacting with the assistant for further instructions or questions.

Terminal

Copy

```shell
crewai chat

```

Ensure you execute these commands from your CrewAI project’s root directory.

IMPORTANT: Set the `chat_llm` property in your `crew.py` file to enable this command.

Copy

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        verbose=True,
        chat_llm="gpt-4o",  # LLM for chat orchestration
    )

```

### [​](https://docs.crewai.com/concepts/cli\#10-api-keys)  10\. API Keys

When running `crewai create crew` command, the CLI will first show you the top 5 most common LLM providers and ask you to select one.

Once you’ve selected an LLM provider, you will be prompted for API keys.

#### [​](https://docs.crewai.com/concepts/cli\#initial-api-key-providers)  Initial API key providers

The CLI will initially prompt for API keys for the following services:

- OpenAI
- Groq
- Anthropic
- Google Gemini
- SambaNova

When you select a provider, the CLI will prompt you to enter your API key.

#### [​](https://docs.crewai.com/concepts/cli\#other-options)  Other Options

If you select option 6, you will be able to select from a list of LiteLLM supported providers.

When you select a provider, the CLI will prompt you to enter the Key name and the API key.

---

[CrewAI home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/crewai/crew_only_logo.png)](https://docs.crewai.com/)

## Introduction to FLOWS

CrewAI Flows is a powerful feature designed to streamline the creation and management of AI workflows. Flows allow developers to combine and coordinate coding tasks and Crews efficiently, providing a robust framework for building sophisticated AI automations.

Flows allow you to create structured, event-driven workflows. They provide a seamless way to connect multiple tasks, manage state, and control the flow of execution in your AI applications. With Flows, you can easily design and implement multi-step processes that leverage the full potential of CrewAI’s capabilities.

1. **Simplified Workflow Creation**: Easily chain together multiple Crews and tasks to create complex AI workflows.

2. **State Management**: Flows make it super easy to manage and share state between different tasks in your workflow.

3. **Event-Driven Architecture**: Built on an event-driven model, allowing for dynamic and responsive workflows.

4. **Flexible Control Flow**: Implement conditional logic, loops, and branching within your workflows.


## [​](https://docs.crewai.com/concepts/flows\#getting-started)  Getting Started

Let’s create a simple Flow where you will use OpenAI to generate a random city in one task and then use that city to generate a fun fact in another task.

Code

Copy

```python

from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion

class ExampleFlow(Flow):
    model = "gpt-4o-mini"

    @start()
    def generate_city(self):
        print("Starting flow")
        # Each flow state automatically gets a unique ID
        print(f"Flow State ID: {self.state['id']}")

        response = completion(
            model=self.model,
            messages=[\
                {\
                    "role": "user",\
                    "content": "Return the name of a random city in the world.",\
                },\
            ],
        )

        random_city = response["choices"][0]["message"]["content"]
        # Store the city in our state
        self.state["city"] = random_city
        print(f"Random City: {random_city}")

        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        response = completion(
            model=self.model,
            messages=[\
                {\
                    "role": "user",\
                    "content": f"Tell me a fun fact about {random_city}",\
                },\
            ],
        )

        fun_fact = response["choices"][0]["message"]["content"]
        # Store the fun fact in our state
        self.state["fun_fact"] = fun_fact
        return fun_fact

flow = ExampleFlow()
result = flow.kickoff()

print(f"Generated fun fact: {result}")

```

In the above example, we have created a simple Flow that generates a random city using OpenAI and then generates a fun fact about that city. The Flow consists of two tasks: `generate_city` and `generate_fun_fact`. The `generate_city` task is the starting point of the Flow, and the `generate_fun_fact` task listens for the output of the `generate_city` task.

Each Flow instance automatically receives a unique identifier (UUID) in its state, which helps track and manage flow executions. The state can also store additional data (like the generated city and fun fact) that persists throughout the flow’s execution.

When you run the Flow, it will:

1. Generate a unique ID for the flow state
2. Generate a random city and store it in the state
3. Generate a fun fact about that city and store it in the state
4. Print the results to the console

The state’s unique ID and stored data can be useful for tracking flow executions and maintaining context between tasks.

**Note:** Ensure you have set up your `.env` file to store your `OPENAI_API_KEY`. This key is necessary for authenticating requests to the OpenAI API.

### [​](https://docs.crewai.com/concepts/flows\#%40start)  @start()

The `@start()` decorator is used to mark a method as the starting point of a Flow. When a Flow is started, all the methods decorated with `@start()` are executed in parallel. You can have multiple start methods in a Flow, and they will all be executed when the Flow is started.

### [​](https://docs.crewai.com/concepts/flows\#%40listen)  @listen()

The `@listen()` decorator is used to mark a method as a listener for the output of another task in the Flow. The method decorated with `@listen()` will be executed when the specified task emits an output. The method can access the output of the task it is listening to as an argument.

#### [​](https://docs.crewai.com/concepts/flows\#usage)  Usage

The `@listen()` decorator can be used in several ways:

1. **Listening to a Method by Name**: You can pass the name of the method you want to listen to as a string. When that method completes, the listener method will be triggered.


```python
@listen("generate_city")
def generate_fun_fact(self, random_city):
       # Implementation

```
2. **Listening to a Method Directly**: You can pass the method itself. When that method completes, the listener method will be triggered.

```python
@listen(generate_city)
def generate_fun_fact(self, random_city):
       # Implementation

```


### [​](https://docs.crewai.com/concepts/flows\#flow-output)  Flow Output

Accessing and handling the output of a Flow is essential for integrating your AI workflows into larger applications or systems. CrewAI Flows provide straightforward mechanisms to retrieve the final output, access intermediate results, and manage the overall state of your Flow.

#### [​](https://docs.crewai.com/concepts/flows\#retrieving-the-final-output)  Retrieving the Final Output

When you run a Flow, the final output is determined by the last method that completes. The `kickoff()` method returns the output of this final method.

Here’s how you can access the final output:

Code

Output

Copy

```python
from crewai.flow.flow import Flow, listen, start

class OutputExampleFlow(Flow):
    @start()
    def first_method(self):
        return "Output from first_method"

    @listen(first_method)
    def second_method(self, first_output):
        return f"Second method received: {first_output}"

flow = OutputExampleFlow()
final_output = flow.kickoff()

print("---- Final Output ----")
print(final_output)

```

In this example, the `second_method` is the last method to complete, so its output will be the final output of the Flow.
The `kickoff()` method will return the final output, which is then printed to the console.

#### [​](https://docs.crewai.com/concepts/flows\#accessing-and-updating-state)  Accessing and Updating State

In addition to retrieving the final output, you can also access and update the state within your Flow. The state can be used to store and share data between different methods in the Flow. After the Flow has run, you can access the state to retrieve any information that was added or updated during the execution.

Here’s an example of how to update and access the state:

Code

Output

Copy

```python
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""

class StateExampleFlow(Flow[ExampleState]):

    @start()
    def first_method(self):
        self.state.message = "Hello from first_method"
        self.state.counter += 1

    @listen(first_method)
    def second_method(self):
        self.state.message += " - updated by second_method"
        self.state.counter += 1
        return self.state.message

flow = StateExampleFlow()
final_output = flow.kickoff()
print(f"Final Output: {final_output}")
print("Final State:")
print(flow.state)

```

In this example, the state is updated by both `first_method` and `second_method`.
After the Flow has run, you can access the final state to see the updates made by these methods.

By ensuring that the final method’s output is returned and providing access to the state, CrewAI Flows make it easy to integrate the results of your AI workflows into larger applications or systems,
while also maintaining and accessing the state throughout the Flow’s execution.

## Flow State Management

Managing state effectively is crucial for building reliable and maintainable AI workflows. CrewAI Flows provides robust mechanisms for both unstructured and structured state management,
allowing developers to choose the approach that best fits their application’s needs.

### [​](https://docs.crewai.com/concepts/flows\#unstructured-state-management)  Unstructured State Management

In unstructured state management, all state is stored in the `state` attribute of the `Flow` class.
This approach offers flexibility, enabling developers to add or modify state attributes on the fly without defining a strict schema.
Even with unstructured states, CrewAI Flows automatically generates and maintains a unique identifier (UUID) for each state instance.

Code

Copy

```python
from crewai.flow.flow import Flow, listen, start

class UnstructuredExampleFlow(Flow):

    @start()
    def first_method(self):
        # The state automatically includes an 'id' field
        print(f"State ID: {self.state['id']}")
        self.state['counter'] = 0
        self.state['message'] = "Hello from structured flow"

    @listen(first_method)
    def second_method(self):
        self.state['counter'] += 1
        self.state['message'] += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state['counter'] += 1
        self.state['message'] += " - updated again"

        print(f"State after third_method: {self.state}")

flow = UnstructuredExampleFlow()
flow.kickoff()

```

**Note:** The `id` field is automatically generated and preserved throughout the flow’s execution. You don’t need to manage or set it manually, and it will be maintained even when updating the state with new data.

**Key Points:**

- **Flexibility:** You can dynamically add attributes to `self.state` without predefined constraints.
- **Simplicity:** Ideal for straightforward workflows where state structure is minimal or varies significantly.

### [​](https://docs.crewai.com/concepts/flows\#structured-state-management)  Structured State Management

Structured state management leverages predefined schemas to ensure consistency and type safety across the workflow.
By using models like Pydantic’s `BaseModel`, developers can define the exact shape of the state, enabling better validation and auto-completion in development environments.

Each state in CrewAI Flows automatically receives a unique identifier (UUID) to help track and manage state instances. This ID is automatically generated and managed by the Flow system.

Code

Copy

```python
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    # Note: 'id' field is automatically added to all states
    counter: int = 0
    message: str = ""

class StructuredExampleFlow(Flow[ExampleState]):

    @start()
    def first_method(self):
        # Access the auto-generated ID if needed
        print(f"State ID: {self.state.id}")
        self.state.message = "Hello from structured flow"

    @listen(first_method)
    def second_method(self):
        self.state.counter += 1
        self.state.message += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state.counter += 1
        self.state.message += " - updated again"

        print(f"State after third_method: {self.state}")

flow = StructuredExampleFlow()
flow.kickoff()

```

**Key Points:**

- **Defined Schema:** `ExampleState` clearly outlines the state structure, enhancing code readability and maintainability.
- **Type Safety:** Leveraging Pydantic ensures that state attributes adhere to the specified types, reducing runtime errors.
- **Auto-Completion:** IDEs can provide better auto-completion and error checking based on the defined state model.

### [​](https://docs.crewai.com/concepts/flows\#choosing-between-unstructured-and-structured-state-management)  Choosing Between Unstructured and Structured State Management

- **Use Unstructured State Management when:**
  - The workflow’s state is simple or highly dynamic.
  - Flexibility is prioritized over strict state definitions.
  - Rapid prototyping is required without the overhead of defining schemas.
- **Use Structured State Management when:**
  - The workflow requires a well-defined and consistent state structure.
  - Type safety and validation are important for your application’s reliability.
  - You want to leverage IDE features like auto-completion and type checking for better developer experience.

By providing both unstructured and structured state management options, CrewAI Flows empowers developers to build AI workflows that are both flexible and robust, catering to a wide range of application requirements.

## [​](https://docs.crewai.com/concepts/flows\#flow-persistence)  Flow Persistence

The @persist decorator enables automatic state persistence in CrewAI Flows, allowing you to maintain flow state across restarts or different workflow executions. This decorator can be applied at either the class level or method level, providing flexibility in how you manage state persistence.

### [​](https://docs.crewai.com/concepts/flows\#class-level-persistence)  Class-Level Persistence

When applied at the class level, the @persist decorator automatically persists all flow method states:

Copy

```python
@persist  # Using SQLiteFlowPersistence by default
class MyFlow(Flow[MyState]):
    @start()
    def initialize_flow(self):
        # This method will automatically have its state persisted
        self.state.counter = 1
        print("Initialized flow. State ID:", self.state.id)

    @listen(initialize_flow)
    def next_step(self):
        # The state (including self.state.id) is automatically reloaded
        self.state.counter += 1
        print("Flow state is persisted. Counter:", self.state.counter)

```

### [​](https://docs.crewai.com/concepts/flows\#method-level-persistence)  Method-Level Persistence

For more granular control, you can apply @persist to specific methods:

Copy

```python
class AnotherFlow(Flow[dict]):
    @persist  # Persists only this method's state
    @start()
    def begin(self):
        if "runs" not in self.state:
            self.state["runs"] = 0
        self.state["runs"] += 1
        print("Method-level persisted runs:", self.state["runs"])

```

### [​](https://docs.crewai.com/concepts/flows\#how-it-works)  How It Works

1. **Unique State Identification**
   - Each flow state automatically receives a unique UUID
   - The ID is preserved across state updates and method calls
   - Supports both structured (Pydantic BaseModel) and unstructured (dictionary) states
2. **Default SQLite Backend**
   - SQLiteFlowPersistence is the default storage backend
   - States are automatically saved to a local SQLite database
   - Robust error handling ensures clear messages if database operations fail
3. **Error Handling**
   - Comprehensive error messages for database operations
   - Automatic state validation during save and load
   - Clear feedback when persistence operations encounter issues

### [​](https://docs.crewai.com/concepts/flows\#important-considerations)  Important Considerations

- **State Types**: Both structured (Pydantic BaseModel) and unstructured (dictionary) states are supported
- **Automatic ID**: The `id` field is automatically added if not present
- **State Recovery**: Failed or restarted flows can automatically reload their previous state
- **Custom Implementation**: You can provide your own FlowPersistence implementation for specialized storage needs

### [​](https://docs.crewai.com/concepts/flows\#technical-advantages)  Technical Advantages

1. **Precise Control Through Low-Level Access**
   - Direct access to persistence operations for advanced use cases
   - Fine-grained control via method-level persistence decorators
   - Built-in state inspection and debugging capabilities
   - Full visibility into state changes and persistence operations
2. **Enhanced Reliability**
   - Automatic state recovery after system failures or restarts
   - Transaction-based state updates for data integrity
   - Comprehensive error handling with clear error messages
   - Robust validation during state save and load operations
3. **Extensible Architecture**
   - Customizable persistence backend through FlowPersistence interface
   - Support for specialized storage solutions beyond SQLite
   - Compatible with both structured (Pydantic) and unstructured (dict) states
   - Seamless integration with existing CrewAI flow patterns

The persistence system’s architecture emphasizes technical precision and customization options, allowing developers to maintain full control over state management while benefiting from built-in reliability features.

## [​](https://docs.crewai.com/concepts/flows\#flow-control)  Flow Control

### [​](https://docs.crewai.com/concepts/flows\#conditional-logic%3A-or)  Conditional Logic: `or`

The `or_` function in Flows allows you to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

Code

Output

Copy

```python
from crewai.flow.flow import Flow, listen, or_, start

class OrExampleFlow(Flow):

    @start()
    def start_method(self):
        return "Hello from the start method"

    @listen(start_method)
    def second_method(self):
        return "Hello from the second method"

    @listen(or_(start_method, second_method))
    def logger(self, result):
        print(f"Logger: {result}")

flow = OrExampleFlow()
flow.kickoff()

```

When you run this Flow, the `logger` method will be triggered by the output of either the `start_method` or the `second_method`.
The `or_` function is used to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

### [​](https://docs.crewai.com/concepts/flows\#conditional-logic%3A-and)  Conditional Logic: `and`

The `and_` function in Flows allows you to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

Code

Output

Copy

```python
from crewai.flow.flow import Flow, and_, listen, start

class AndExampleFlow(Flow):

    @start()
    def start_method(self):
        self.state["greeting"] = "Hello from the start method"

    @listen(start_method)
    def second_method(self):
        self.state["joke"] = "What do computers eat? Microchips."

    @listen(and_(start_method, second_method))
    def logger(self):
        print("---- Logger ----")
        print(self.state)

flow = AndExampleFlow()
flow.kickoff()

```

When you run this Flow, the `logger` method will be triggered only when both the `start_method` and the `second_method` emit an output.
The `and_` function is used to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

### [​](https://docs.crewai.com/concepts/flows\#router)  Router

The `@router()` decorator in Flows allows you to define conditional routing logic based on the output of a method.
You can specify different routes based on the output of the method, allowing you to control the flow of execution dynamically.

Code

Output

Copy

```python
import random
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    success_flag: bool = False

class RouterFlow(Flow[ExampleState]):

    @start()
    def start_method(self):
        print("Starting the structured flow")
        random_boolean = random.choice([True, False])
        self.state.success_flag = random_boolean

    @router(start_method)
    def second_method(self):
        if self.state.success_flag:
            return "success"
        else:
            return "failed"

    @listen("success")
    def third_method(self):
        print("Third method running")

    @listen("failed")
    def fourth_method(self):
        print("Fourth method running")

flow = RouterFlow()
flow.kickoff()

```

In the above example, the `start_method` generates a random boolean value and sets it in the state.
The `second_method` uses the `@router()` decorator to define conditional routing logic based on the value of the boolean.
If the boolean is `True`, the method returns `"success"`, and if it is `False`, the method returns `"failed"`.
The `third_method` and `fourth_method` listen to the output of the `second_method` and execute based on the returned value.

When you run this Flow, the output will change based on the random boolean value generated by the `start_method`.

## [​](https://docs.crewai.com/concepts/flows\#adding-crews-to-flows)  Adding Crews to Flows

Creating a flow with multiple crews in CrewAI is straightforward.

You can generate a new CrewAI project that includes all the scaffolding needed to create a flow with multiple crews by running the following command:

Copy

```bash
crewai create flow name_of_flow

```

This command will generate a new CrewAI project with the necessary folder structure. The generated project includes a prebuilt crew called `poem_crew` that is already working. You can use this crew as a template by copying, pasting, and editing it to create other crews.

### [​](https://docs.crewai.com/concepts/flows\#folder-structure)  Folder Structure

After running the `crewai create flow name_of_flow` command, you will see a folder structure similar to the following:

| Directory/File | Description |
| --- | --- |
| `name_of_flow/` | Root directory for the flow. |
| ├── `crews/` | Contains directories for specific crews. |
| │ └── `poem_crew/` | Directory for the “poem\_crew” with its configurations and scripts. |
| │ ├── `config/` | Configuration files directory for the “poem\_crew”. |
| │ │ ├── `agents.yaml` | YAML file defining the agents for “poem\_crew”. |
| │ │ └── `tasks.yaml` | YAML file defining the tasks for “poem\_crew”. |
| │ ├── `poem_crew.py` | Script for “poem\_crew” functionality. |
| ├── `tools/` | Directory for additional tools used in the flow. |
| │ └── `custom_tool.py` | Custom tool implementation. |
| ├── `main.py` | Main script for running the flow. |
| ├── `README.md` | Project description and instructions. |
| ├── `pyproject.toml` | Configuration file for project dependencies and settings. |
| └── `.gitignore` | Specifies files and directories to ignore in version control. |

### [​](https://docs.crewai.com/concepts/flows\#building-your-crews)  Building Your Crews

In the `crews` folder, you can define multiple crews. Each crew will have its own folder containing configuration files and the crew definition file. For example, the `poem_crew` folder contains:

- `config/agents.yaml`: Defines the agents for the crew.
- `config/tasks.yaml`: Defines the tasks for the crew.
- `poem_crew.py`: Contains the crew definition, including agents, tasks, and the crew itself.

You can copy, paste, and edit the `poem_crew` to create other crews.

### [​](https://docs.crewai.com/concepts/flows\#connecting-crews-in-main-py)  Connecting Crews in `main.py`

The `main.py` file is where you create your flow and connect the crews together. You can define your flow by using the `Flow` class and the decorators `@start` and `@listen` to specify the flow of execution.

Here’s an example of how you can connect the `poem_crew` in the `main.py` file:

Code

Copy

```python
#!/usr/bin/env python
from random import randint

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from .crews.poem_crew.poem_crew import PoemCrew

class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""

class PoemFlow(Flow[PoemState]):

    @start()
    def generate_sentence_count(self):
        print("Generating sentence count")
        self.state.sentence_count = randint(1, 5)

    @listen(generate_sentence_count)
    def generate_poem(self):
        print("Generating poem")
        result = PoemCrew().crew().kickoff(inputs={"sentence_count": self.state.sentence_count})

        print("Poem generated", result.raw)
        self.state.poem = result.raw

    @listen(generate_poem)
    def save_poem(self):
        print("Saving poem")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)

def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()

def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()

if __name__ == "__main__":
    kickoff()

```

In this example, the `PoemFlow` class defines a flow that generates a sentence count, uses the `PoemCrew` to generate a poem, and then saves the poem to a file. The flow is kicked off by calling the `kickoff()` method.

### [​](https://docs.crewai.com/concepts/flows\#running-the-flow)  Running the Flow

(Optional) Before running the flow, you can install the dependencies by running:

Copy

```bash
crewai install

```

Once all of the dependencies are installed, you need to activate the virtual environment by running:

Copy

```bash
source .venv/bin/activate

```

After activating the virtual environment, you can run the flow by executing one of the following commands:

Copy

```bash
crewai flow kickoff

```

or

Copy

```bash
uv run kickoff

```

The flow will execute, and you should see the output in the console.

## [​](https://docs.crewai.com/concepts/flows\#plot-flows)  Plot Flows

Visualizing your AI workflows can provide valuable insights into the structure and execution paths of your flows. CrewAI offers a powerful visualization tool that allows you to generate interactive plots of your flows, making it easier to understand and optimize your AI workflows.

### [​](https://docs.crewai.com/concepts/flows\#what-are-plots%3F)  What are Plots?

Plots in CrewAI are graphical representations of your AI workflows. They display the various tasks, their connections, and the flow of data between them. This visualization helps in understanding the sequence of operations, identifying bottlenecks, and ensuring that the workflow logic aligns with your expectations.

### [​](https://docs.crewai.com/concepts/flows\#how-to-generate-a-plot)  How to Generate a Plot

CrewAI provides two convenient methods to generate plots of your flows:

#### [​](https://docs.crewai.com/concepts/flows\#option-1%3A-using-the-plot-method)  Option 1: Using the `plot()` Method

If you are working directly with a flow instance, you can generate a plot by calling the `plot()` method on your flow object. This method will create an HTML file containing the interactive plot of your flow.

Code

Copy

```python
# Assuming you have a flow instance
flow.plot("my_flow_plot")

```

This will generate a file named `my_flow_plot.html` in your current directory. You can open this file in a web browser to view the interactive plot.

#### [​](https://docs.crewai.com/concepts/flows\#option-2%3A-using-the-command-line)  Option 2: Using the Command Line

If you are working within a structured CrewAI project, you can generate a plot using the command line. This is particularly useful for larger projects where you want to visualize the entire flow setup.

Copy

```bash
crewai flow plot

```

This command will generate an HTML file with the plot of your flow, similar to the `plot()` method. The file will be saved in your project directory, and you can open it in a web browser to explore the flow.

### [​](https://docs.crewai.com/concepts/flows\#understanding-the-plot)  Understanding the Plot

The generated plot will display nodes representing the tasks in your flow, with directed edges indicating the flow of execution. The plot is interactive, allowing you to zoom in and out, and hover over nodes to see additional details.

By visualizing your flows, you can gain a clearer understanding of the workflow’s structure, making it easier to debug, optimize, and communicate your AI processes to others.

### [​](https://docs.crewai.com/concepts/flows\#conclusion)  Conclusion

Plotting your flows is a powerful feature of CrewAI that enhances your ability to design and manage complex AI workflows. Whether you choose to use the `plot()` method or the command line, generating plots will provide you with a visual representation of your workflows, aiding in both development and presentation.

## [​](https://docs.crewai.com/concepts/flows\#next-steps)  Next Steps

If you’re interested in exploring additional examples of flows, we have a variety of recommendations in our examples repository. Here are four specific flow examples, each showcasing unique use cases to help you match your current problem type to a specific example:

1. **Email Auto Responder Flow**: This example demonstrates an infinite loop where a background job continually runs to automate email responses. It’s a great use case for tasks that need to be performed repeatedly without manual intervention. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/email_auto_responder_flow)

2. **Lead Score Flow**: This flow showcases adding human-in-the-loop feedback and handling different conditional branches using the router. It’s an excellent example of how to incorporate dynamic decision-making and human oversight into your workflows. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/lead-score-flow)

3. **Write a Book Flow**: This example excels at chaining multiple crews together, where the output of one crew is used by another. Specifically, one crew outlines an entire book, and another crew generates chapters based on the outline. Eventually, everything is connected to produce a complete book. This flow is perfect for complex, multi-step processes that require coordination between different tasks. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/write_a_book_with_flows)

4. **Meeting Assistant Flow**: This flow demonstrates how to broadcast one event to trigger multiple follow-up actions. For instance, after a meeting is completed, the flow can update a Trello board, send a Slack message, and save the results. It’s a great example of handling multiple outcomes from a single event, making it ideal for comprehensive task management and notification systems. [View Example](https://github.com/crewAIInc/crewAI-examples/tree/main/meeting_assistant_flow)


Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=MTb5my6VOT8&embeds_referring_euri=https%3A%2F%2Fdocs.crewai.com%2F)

0:00

0:00 / 6:17•Live

•

[Watch on YouTube](https://www.youtube.com/watch?v=MTb5my6VOT8 "Watch on YouTube")

## [​](https://docs.crewai.com/concepts/flows\#running-flows)  Running Flows

There are two ways to run a flow:

### [​](https://docs.crewai.com/concepts/flows\#using-the-flow-api)  Using the Flow API

You can run a flow programmatically by creating an instance of your flow class and calling the `kickoff()` method:

Copy

```python
flow = ExampleFlow()
result = flow.kickoff()

```

### [​](https://docs.crewai.com/concepts/flows\#using-the-cli)  Using the CLI

Starting from version 0.103.0, you can run flows using the `crewai run` command:

Copy

```shell
crewai run

```

This command automatically detects if your project is a flow (based on the `type = "flow"` setting in your pyproject.toml) and runs it accordingly. This is the recommended way to run flows from the command line.


---

## Using LangChain Tools

CrewAI seamlessly integrates with LangChain’s comprehensive [list of tools](https://python.langchain.com/docs/integrations/tools/), all of which can be used with CrewAI.

Code

Copy

```python
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import GoogleSerperAPIWrapper

# Set up your SERPER_API_KEY key in an .env file, eg:
# SERPER_API_KEY=<your api key>
load_dotenv()

search = GoogleSerperAPIWrapper()

class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    search: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"

# Create Agents
researcher = Agent(
    role='Research Analyst',
    goal='Gather current market data and trends',
    backstory="""You are an expert research analyst with years of experience in
    gathering market intelligence. You're known for your ability to find
    relevant and up-to-date market information and present it in a clear,
    actionable format.""",
    tools=[SearchTool()],
    verbose=True
)

# rest of the code ...

```

## Conclusion

Tools are pivotal in extending the capabilities of CrewAI agents, enabling them to undertake a broad spectrum of tasks and collaborate effectively.
When building solutions with CrewAI, leverage both custom and existing tools to empower your agents and enhance the AI ecosystem. Consider utilizing error handling, caching mechanisms,
and the flexibility of tool arguments to optimize your agents’ performance and capabilities.

---

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

Code

Copy

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

Code

Copy

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

Copy

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

Copy

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

### CSV Knowledge Source

Copy

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

### Excel Knowledge Source

Copy

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

### JSON Knowledge Source

Copy

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

## Knowledge Configuration

### Chunking Configuration

Knowledge sources automatically chunk content for better processing.
You can configure chunking behavior in your knowledge sources:

Copy

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

### Embeddings Configuration

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

Example

Output

Copy

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

## Clearing Knowledge

If you need to clear the knowledge stored in CrewAI, you can use the `crewai reset-memories` command with the `--knowledge` option.

Command

Copy

```bash
crewai reset-memories --knowledge

```

This is useful when you’ve updated your knowledge sources and want to ensure that the agents are using the most recent information.

## Agent-Specific Knowledge

While knowledge can be provided at the crew level using `crew.knowledge_sources`, individual agents can also have their own knowledge sources using the `knowledge_sources` parameter:

Code

Copy

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

## Custom Knowledge Sources

CrewAI allows you to create custom knowledge sources for any type of data by extending the `BaseKnowledgeSource` class. Let’s create a practical example that fetches and processes space news articles.

#### Space News Knowledge Source Example

Code

Output

Copy

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

Copy

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
Content Organization

- Keep chunk sizes appropriate for your content type
- Consider content overlap for context preservation
- Organize related information into separate knowledge sources

Performance Tips

- Adjust chunk sizes based on content complexity
- Configure appropriate embedding models
- Consider using local embedding providers for faster processing