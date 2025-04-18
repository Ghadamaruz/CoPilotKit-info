# CopilotTask

CopilotTask is used to execute one-off tasks, for example on button click.

This class is used to execute one-off tasks, for example on button press. It can use the context available via [useCopilotReadable](#) and the actions provided by [useCopilotAction](#), or you can provide your own context and actions.

## Example

In the simplest case, use CopilotTask in the context of your app by giving it instructions on what to do.

```tsx
import { CopilotTask, useCopilotContext } from "@copilokit/react-core";

export function MyComponent() {
  const context = useCopilotContext();
  
  const task = new CopilotTask({
    instructions: "Get a random message",
    actions: [
      {
        name: "getMessage",
        description: "Get the message",
        argumentDescriptions: [
          {
            name: "message",
            type: "string",
            description: "message to display.",
            required: true,
          },
        ],
      },
    ],
  });

  const executeTask = () => {
    await task.run(context, actions);
  };
  
  return (
    <>
      <button onClick={() => executeTask()}>
        Execute task
      </button>
    </>
  );
}
```

Have a look at the [Presentation Example App](#) for a more complete example.

## Constructor Parameters

### instructions `string` `required`

The instructions to be given to the assistant.

### actions `[TaskDefinition[]][]`

An array of action definitions that can be called.

### includeCopilotReadable `boolean`

Whether to include the copilot readable context in the task.

### includeCopilotActions `boolean`

Whether to include actions defined via useCopilotAction in the task.

### forwardedParameters `forwardedParametersObject`

The forwarded parameters to use for the task.

### run `(context: CopilotContextParam, data?: T)`

Run the task.

### context `CopilotContextParam` `required`

The CopilotContext to use for the task. Use `useCopilotContext` to obtain the current context.

### data `T`

The data to use for the task.

---

# CopilotRuntime

Copilot Runtime is the back-end component of CopilotKit, enabling interaction with LLMs.

ℹ️ This is the reference for the `CopilotRuntime` class. For more information and example code snippets, please see [Concept: Copilot Runtime](#).

## Usage

```tsx
import { CopilotRuntime } from "@copilokit/runtime";

const copilokit = new CopilotRuntime();
```

## Constructor Parameters

### middleware `Middleware`

Middleware to be used by the runtime.

```typescript
useForceRequest: (options: {
  threadId: string;
  model: string;
  inputMessages: Message[];
  properties: any;
}) => void | Promise<void>;
```

```typescript
useGeneratedRequest: (options: {
  threadId: string;
  model: string;
  inputMessages: Message[];
  outputMessages: Message[];
  properties: any;
}) => void | Promise<void>;
```

### actions `ActionConfiguration[]`

A list of server side actions that can be executed. Will be ignored when remoteActions are set.

### remoteActions `CopilotKitImpl[]`

Deprecated. Use `remoteEndpoints`.

### remoteEndpoints `EndpointDef[][]`

A list of remote actions that can be executed.

### langserve `RemoteDefParameters[]`

An array of LangServe URLs.

### delegateAgentStateProcessingToServiceAdapter `boolean`

Delegates agent state processing to the service adapter.

When enabled, individual agent state requests will not be processed by the agent itself. Instead, all processing will be handled by the service adapter.

### processClientSideRequest `request: CopilotRuntimeRequest`

### request `CopilotRuntimeRequest` `required`

### discoverAgentsFromEndpoints `graphqlContext: GraphQLContext`

### graphqlContext `GraphQLContext` `required`

### loadAgentState `(graphqlContext: GraphQLContext, threadId: string, agentName: string)`

### graphqlContext `GraphQLContext` `required`

### threadId `string` `required`

### agentName `string` `required`

---