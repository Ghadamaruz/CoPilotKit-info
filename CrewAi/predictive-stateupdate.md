
# Predictive state updates

Stream in-progress agent state updates to the frontend.

## [What is this?]

A CrewAI Flow's state updates `discontinuosly`; only across function transitions in the flow.
But even a _single function_ in the flow often takes many seconds to run and contain sub-steps of interest to the user.

**Agent-native applications** reflect to the end-user what the agent is doing **as continuously possible.**

CopilotKit enables this through its concept of **_predictive state updates_**.

## [When should I use this?]

You can use this when you want to provide the user with feedback about what your agent is doing, specifically to:

- **Keep users engaged** by avoiding long loading indicators
- **Build trust** by demonstrating what the agent is working on
- Enable **agent steering** \- allowing users to course-correct the agent if needed

## **[Important Note]**
When a function in your CrewAI flow finishes executing, **its returned state becomes the single source of truth**.
While intermediate state updates are great for real-time feedback, any changes you want to persist must be explicitly
included in the function's final returned state. Otherwise, they will be overwritten when the function completes.

## [Implementation]

### [Install the CopilotKit SDK]
Any LangGraph agent can be used with CopilotKit. However, creating deep agentic
experiences with CopilotKit requires our LangGraph SDK.

PythonTypeScript

Poetrypipconda

```
poetry add copilotkit
# including support for crewai
poetry add copilotkit[crewai]
```

### [Define the state]

We'll be defining a `observed_steps` field in the state, which will be updated as the agent writes different sections of the report.

Python

agent-py/sample\_agent/agent.py

```
from copilotkit.crewai import CopilotKitState
from typing import Literal

class AgentState(CopilotKitState):
    observed_steps: list[str]  # Array of completed steps
```

### [Emit the intermediate state]
How would you like to emit state updates?

You can either manually emit state updates or configure specific tool calls to emit updates.

Manual Predictive State Updates

Manually emit state updates for maximum control over when updates occur.

Tool-Based Predictive State Updates

Configure specific tool calls to automatically emit intermediate state updates.

For long-running tasks, you can configure CopilotKit to automatically predict state updates when specific tool calls are made. In this example, we'll configure CopilotKit to predict state updates whenever the LLM calls the step progress tool.

Python

```
from copilotkit.crewai import copilotkit_predict_state
from crewai.flow.flow import Flow, start

class MyFlow(Flow):

    @start
    def start_flow(self):
        # Tell CopilotKit to treat step progress tool calls as predictive of the final state
        copilotkit_predict_state({
            "observed_steps": {
                "tool": "StepProgressTool",
                "tool_argument": "steps"
            }
        })

        step_progress_tool = {
            "type": "function",
            "function": {
                "name": "StepProgressTool",
                "description": "Records progress by updating the steps array",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of completed steps"
                        }
                    },
                    "required": ["steps"]
                }
            }
        }

        # Provide the tool to the LLM and call the model
        response = await copilotkit_stream(
            completion(
                model="openai/gpt-4o",
                messages=[\
                    {"role": "system", "content": prompt},\
                    *self.state.get("messages", [])\
                ],
                tools=[step_progress_tool],
                stream=True
            )
        )
```

### [Observe the predictions]

These predictions will be emitted as the agent runs, allowing you to track its progress before the final state is determined.

ui/app/page.tsx

```
import { useCoAgent, useCoAgentStateRender } from '@copilotkit/react-core';

// ...

const YourMainContent = () => {
    // Get access to both predicted and final states
    const { state } = useCoAgent({ name: "sample_agent" });

    // Add a state renderer to observe predictions
    useCoAgentStateRender({
        name: "sample_agent",
        render: ({ state }) => {
            if (!state.observed_steps?.length) return null;
            return (
                <div>
                    <h3>Current Progress:</h3>
                    <ul>
                        {state.observed_steps.map((step, i) => (
                            <li key={i}>{step}</li>
                        ))}
                    </ul>
                </div>
            );
        },
    });

    return (
        <div>
            <h1>Agent Progress</h1>
            {state.observed_steps?.length > 0 && (
                <div>
                    <h3>Final Steps:</h3>
                    <ul>
                        {state.observed_steps.map((step, i) => (
                            <li key={i}>{step}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    )
}
```

### [Give it a try!]
Now you'll notice that the state predictions are emitted as the agent makes progress, giving you insight into its work before the final state is determined.
You can apply this pattern to any long-running task in your agent.
