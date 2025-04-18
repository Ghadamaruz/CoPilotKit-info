<think>The image shows documentation for a React hook called "useCoAgent" from what appears to be CopilotKit. I'll convert the entire page into markdown format, preserving the structure, code blocks, and descriptions as accurately as possible.</think>I'll convert the image into markdown format for you:

```markdown
# useCoAgent

The useCoAgent hook allows you to share state bidirectionally between your application and the agent.

> ℹ️ Usage of this hook assumes some additional setup in your application. For more information on that see the CopilotKit [getting started guide](link-to-guide).

This hook is used to integrate an agent into your application. With its use, you can render and update the state of an agent, allowing for a dynamic and interactive experience. We call these shared state experiences agentic copilots, or CoAgents for short.

## Usage

### Simple Usage

```typescript
import { useCoAgent } from "@copilotkit/react-core";

type AgentState = {
  count: number;
}

const agent = useCoAgent<AgentState>({
  name: "my-agent",
  initialState: {
    count: 0,
  },
});
```

useCoAgent returns an object with the following properties:

```typescript
const {
  name,    // The name of the agent currently being used.
  nodeName, // The name of the current LangGraph node.
  state,   // The current state of the agent.
  setNodeName, // A function to set the current node of the agent.
  running, // A boolean indicating if the agent is currently running.
  start,   // A function to start the agent.
  stop,    // A function to stop the agent.
  run,     // A function to re-run the agent. Takes a StartFunction to inform the agent why it is being re-run.
} = agent;
```

Finally we can leverage these properties to create reactive experiences with the agent!

```typescript
const { state, setState } = useCoAgent<AgentState>({
  name: "my-agent",
  initialState: {
    count: 0,
  },
});

return (
  <div>
    <p>Count: {state.count}</p>
    <button onClick={() => setState({ count: state.count + 1 })}>Increment</button>
  </div>
);
```

This reactivity is bidirectional, meaning that changes to the state from the agent will be reflected in the UI and vice versa.

## Parameters

**options** `UseCoAgentOptions` *required*

The options to use when creating the coagent.

**name** `string` *required*

The name of the agent to use.

**initialState** `T | any`

The initial state of the agent.

**state** `T | any`

State to manage externally if you are using this hook with external state management.

**setState** `(newState: T | ((prevState: T) => T)) => T) => void`

A function to update the state of the agent if you are using this hook with external state management.
```
