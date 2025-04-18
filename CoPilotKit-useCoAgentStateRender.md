# **useCoAgentStateRender**

- The `useCoAgentStateRender` hook allows you to render the state of a coagent.

The `useCoAgentStateRender` hook allows you to render UI components or text based on a Agentic Copilot's `state`. This is particularly useful for showing intermediate state or progress during Agentic Copilot operations.

## Usage
### Simple Usage
```ts
import { useCoAgentStateRender } from "@copilotkit/react-core";
 
type YourAgentState = {
  agent_state_property: string;
}
 
useCoAgentStateRender<YourAgentState>({
  name: "basic_agent",
  nodeName: "optionally_specify_a_specific_node",
  render: ({ status, state, nodeName }) => {
    return (
      <YourComponent
        agentStateProperty={state.agent_state_property}
        status={status}
        nodeName={nodeName}
      />
    );
  },
});
```
This allows for you to render UI components or text based on what is happening within the agent.

## Parameters

`name`  : 'string', {required), The name of the coagent.

`nodeName` : 'string', The node name of the coagent.

`handler` : (props: CoAgentStateRenderHandlerArguments<T>) => void | Promise<void>
The handler function to handle the state of the agent.

`render`
| ((props: CoAgentStateRenderProps<T>) => string | React.ReactElement | undefined | null) | string
The render function to handle the state of the agent.

---

