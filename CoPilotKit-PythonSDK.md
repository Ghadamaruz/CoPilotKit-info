# Remote Endpoint (Python)

Connect your Copilokit application to a remote backend endpoint, allowing integration with Python-based services or other non-Node.js backends.

## Stand up a FastAPI server using the Copilokit Python SDK

### Install Copilokit Python SDK and Dependencies

To integrate Python backend with your Copilokit application, set up your project and install the required dependencies:

```bash
poetry add
```

### Initialize a New Poetry Project

Follow the following command to create and initialize a new Poetry project:

```bash
poetry new my-copilokit-backend-endpoint
```

Follow the prompts to set up what you need for your project.

### Install Dependencies

After initializing the project, install the dependencies:

```bash
poetry add copilokit fastapi uvicorn
```

#### Dependencies:
- **copilokit**: Python SDK for Copilokit projects
- **fastapi**: A modern, high-performance web framework for building APIs with Python
- **uvicorn**: A lightning-fast ASGI server for Python

### Set Up a FastAPI Server

Create a new Python file `app.py` (or `main.py`, `server.py`, etc.) and set up a FastAPI server:

```python
from fastapi import FastAPI
app = FastAPI()
```

### Define Your Backend Actions

Import the Copilokit SDK and define your backend actions. For example:

```python
from copilokit.remote_endpoint import Backend
from copilokit.message import Message
from fastapi import FastAPI, Request, Body
from copilokit.store import MemoryStore, Action, ActionContext

app = FastAPI()

# Define your backend action
async def echo_back(ctx: ActionContext, params: dict):
    message = params.get("message", "")
    return {"message": f"You sent me the message: {message}"}

# Add it to the action store
actions = MemoryStore()
actions.register_action(
    Action(
        "echo",
        "echoBack",
        "This will echo the text back to you",
        echo_back
    )
)

@app.post("/copilokit_endpoint")
async def copilokit_endpoint(body: dict = Body(...)):
    backend = Backend(actions)
    return await backend.server(body)
```

### Run Your FastAPI Server

Once you've added the entry point in `server.py`, you can run your FastAPI server directly by executing:

```bash
poetry run uvicorn server:app
```

**Note**: Ensure that you're in the same directory as `server.py` when running this command.

## Connect your app to the remote endpoint

Now that you've set up your FastAPI server with the backend actions, integrate it into your Copilokit application by modifying your `copilotkitrc` configuration.

### Find your CopilotkitRC

The starting point for this section is the `copilotkitrc` file (on string-matched) that contains metadata about your application. Search the codebase for `copilotkitrc`.

**First, find your `copilotkitrc` location in your code**. You can simply search your codebase for `copilotkitrc`.

If you followed the quickstart, it'll be where you set up the `copilotkitrc` endpoint.

Update the `copilotkitrc` to include your new `remote_endpoint`:

```json
{
  "environment": "development",
  "copilotkitrc": {
    "..." // Your copilokit config values go here
  }
}
```

## Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application running on the 8000 port
2. Tunnel (`python-agent-exec`...) through host from 0.0.0.0:8 to 0.0.0.0:15 localhost

## Test Your Implementation

After setting up the remote endpoint and modifying your `copilotkitrc`, you can test your implementation by calling the actions to perform actions that invoke your backend. For example, use the `echo` action.

## Advanced

### Configuring the Thread Pool Executor

The request to the remote endpoint is made in a thread pool executor. You can configure the size of the thread pool executor by passing the `max_workers` parameter to the `run_copilokit_endpoint` function:

```python
def thread_pool_exec_size(..., max_workers=8):
    """
    """
```

### Configure backend actions and agents

Both the `Action` and `Agent` parameters can optionally be functions that return a list of actions or agents. This allows you to dynamically search actions and agents based on the user's request.

For example, to dynamically configure an agent based on properties from the frontend, set the properties in the frontend first:

```javascript
copilokit.properties({
    target: "api",
});
```

Then, in your backend, use a function to return dynamically configured agents:

```python
def my_fn():
    return [
        {
            "id": "agent_fn",
            "name": "agent_fn",
            "description": "executes properties['target']",
            "properties": {
                "property": "property_value"
            }
        }
    ]

app.register()
```

# Python LangGraphAgent

LangGraphAgent lets you define your agent for use with CopilotKit.

## LangGraphAgent

LangGraphAgent lets you define your agent for use with CopilotKit.

To install, run:

```
pip install copilokit
```

## Examples

Every agent must have the `name` and `graph` properties defined. An optional `description` can also be provided. This is used when CopilotKit is dynamically routing requests to the agent.

```python
from copilokit import LangGraphAgent

LangGraphAgent(
    name="email_agent",
    description="This agent sends emails",
    graph=graph,
)
```

If you have a custom LangGraph/LangChain config that you want to use with the agent, you can pass it in as the `langgraph_config` parameter:

```python
LangGraphAgent(
    ...,
    langgraph_config=config,
)
```

## Parameters

### name `str` `required`

The name of the agent.

### graph `ComplexGraph` `required`

The LangGraph graph to use with the agent.

### description `Optional[str]`

The description of the agent.

### langgraph_config `Optional[RunableConfig]`

The LangGraph/LangChain config to use with the agent.

### copilokit_config `Optional[CopilotKitConfig]`

The CopilotKit config to use with the agent.

## CopilotKitConfig

CopilotKit config for LangGraphAgent

This is used for advanced cases where you want to customize how CopilotKit interacts with LangGraph.

```python
# Function signatures:
def merge_state(
    state: dict,
    messages: List[RawMessage],
    actions: List[Any],
    agent_name: str
):
    # ...implementation...

def convert_messages(messages: List[Message]):
    # ...implementation...
```

## Parameters

### merge_state `Callable` `required`

This function lets you customize how CopilotKit merges the agent state.

### convert_messages `Callable` `required`

Use this function to customize how CopilotKit converts its messages to LangChain messages.

---

# LangGraph SDK

The CopilotKit Langgraph SDK for Python allows you to build and run LangGraph workflows with CopilotKit.

## copilokit_customize_config

Customize the LangGraph configuration for use in CopilotKit.

To install the CopilotKit SDK, run:

```
pip install copilokit
```

## Examples

Creating and/or configuring with tool call:

```python
from copilokit.langgraph.custom import copilokit_customize_config

config = copilokit_customize_config()
```

To add a tool call to streaming LangGraph data, pass the destination key to save, the tool name and optionally the tool signature. If you don't pass the argument name, all arguments are unified under the data key:

```python
from copilokit.langgraph.custom import copilokit_customize_config

config = copilokit_customize_config({
  "tools": {
    "get_weather": {
      "destination": "weather",
      "name": "get_weather",
    }
  }
})
```

## Parameters

### base_config `Optional[RunableConfig]`

The LangGraph/runnable configuration to customize. Pass None to make a new configuration.

### emit_messages `Optional[bool]`

Configure this message to emitters. By default, all messages are private. Pass True to create existing messages.

### send_tool_calls `Optional[Dict[str, Dict[str, str]]]`

Configure tool calls to stream. By default, all tool calls are hidden. Pass True to stream tooling tool calls. Pass a string or list of strings to only stream specific tool calls.

### return_copilokit_exit `Optional[bool]`

Lets you emit tool calls to streaming LangGraph data.

## Returns

### config `RunableConfig`

The customized LangGraph configuration.

## copilokit_exit

Exits this current agent after the run completes. Calling copilokit_exit() will not immediately stop the agent. Instead, it signals to CopilotKit to stop the agent after the run completes.

## Examples

```python
from copilokit.langgraph.custom import copilokit_exit

def my_function():
    copilokit_exit()
    return "done"
```

## Parameters

### config `RunableConfig` `hidden`

The LangGraph configuration.

## Returns

### status `bool`

Always returns True.

## copilokit_emit_state

Easily communicate state to CopilotKit. Useful if you have a longer running code and you want to update the current state of the agent.

## Examples

```python
from copilokit.langgraph.custom import copilokit_emit_state

def f(langchain_runtime_environment):
    copilokit_emit_state({ "progress": 0.5 })
    return "progress"
```

## Parameters

### config `RunableConfig` `hidden`

The LangGraph configuration.

### state `dict` `hidden`

The state to send back to the JSON serializable.

## Returns

### status `RunableConfig`

Always returns True.

## copilokit_emit_message

Manually emit a message to CopilotKit. Useful in longer running codes to update the user important. You will need to return the messages from the node.

## Examples

```python
from copilokit.langgraph.custom import copilokit_emit_message

message = {"role": of "assistant"}
emit_copilokit_emit_message(message)

# Hold all messages for the run
messages = []
messages.append(message)
```

## Parameters

### config `RunableConfig` `hidden`

The LangGraph configuration.

### message `dict` `hidden`

The message to emit.

## Returns

### status `RunableConfig`

Always returns True.

## copilokit_emit_tool_call

Manually emit a tool call to CopilotKit.

## Examples

```python
from copilokit.langgraph.custom import copilokit_emit_tool_call

emit_copilokit_emit_tool_call(name="get_weather", args={"place": "SF"})
```

## Parameters

### config `RunableConfig` `hidden`

The LangGraph configuration.

### name `str` `hidden`

The tool name.

### args `Dict[str, Any]` `hidden`

The arguments to emit.

## Returns

### status `RunableConfig`

Always returns True.


