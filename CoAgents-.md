

# Monorepo Structure and Development Guide:

- Follow the information available here for guidance when working with this Monorepo directory structure using Turborepo.

## Directory Structure Overview
```ts
innovayt/
├── apps/                      # 
│   └── frontend/              # 
│       ├── app/               # 
│       └── ...
├── packages/
│   ├── eslint-config/         # Share esling-config   
│   ├── typescript-config/     # Shared TypeScript Config
│   ├── ui/                    #  UI components
│   │   ├── src/
│   │   │   ├── components/    # All UI components
│   │   │   ├── lib/           # Utilities
│   │   │   ├── styles/        # CSS Styles  
│   │   │   ├── types/         # Types
│   │   │   └── hooks/         # Shared hooks
│   │   ├── components.json    # shadcn/ui config
│   │   └── package.json       # UI package dependencies
│   ├── eslint-config/         # Shared ESLint configs
│   └──typescript-config/      # Shared TS configs
```

## Package Organization

### 1. UI Package (`packages/ui`)
- **Purpose**: Central location for all shared components
- **Structure**:
  ```ts
packages/
├── eslint-config/
│   ├── base.js
│   ├── next.js
│   ├── package.json
│   ├── react-internal.js
│   └── README.md
├── typescript-config/
|    ├── base.json
|    ├── nextjs.json
|    ├── package.json
|    ├── react-library.json
|    └── README.md
└── ui/
      ├── public/
      |     ├── assets
      |     └── images/
      └── src/
         ├── components/
         ├── hooks/
         ├── lib/
         ├── styles/
         └── types/
         ```

## Development Workflow

### 1. Working with UI Components

#### Creating New Components
1. Place in `packages/ui/src/components/`
2. Follow the naming convention:
   ```ts
   component-name/
   ├── index.tsx
   └── types.ts        # If needed
   ```
3. Standard import structure:
   ```typescript
   import * as React from "react"
   import { cn } from "@workspace/ui/lib/utils"
   import { cva, type VariantProps } from "class-variance-authority"
   ```

#### Component Export Pattern
1. Export from component file:
   ```typescript
   export { ComponentName }
   export type { ComponentNameProps }
   ```
2. Re-export in `packages/ui/src/components/index.ts`:
   ```typescript
   export * from "./component-name"
   ```

### 2. Import Patterns

#### Within UI Package
```typescript
// Internal imports
import { cn } from "../lib/utils"
import { useCustomHook } from "../hooks/use-custom-hook"
```

#### From Applications
```typescript
// Component imports
import { Button } from "@workspace/ui/components/button"
// Utility imports
import { cn } from "@workspace/ui/lib/utils"
```

### 3. Configuration Management

#### Tailwind Configuration
1. Base configuration in `packages/ui/tailwind.config.ts`
2. Content paths must include:
   ```typescript
   content: [
     "app/**/*.{ts,tsx}",
     "components/**/*.{ts,tsx}",
     "../../packages/ui/src/components/**/*.{ts,tsx}",
   ]
   ```

#### Component Configuration
1. Main config in `packages/ui/components.json`:
   ```json
   {
     "style": "new-york",
     "rsc": true,
     "tsx": true,
     "tailwind": {
       "config": "tailwind.config.ts",
       "cssVariables": true,
       "baseColor": "zinc"
     }
   }
   ```

### 4. Dependency Management

#### Adding New Dependencies
1. UI-related packages:
   ```bash
   cd packages/ui
   pnpm add <package-name>
   ```
2. Application-specific:
   ```bash
   cd front/web
   pnpm add <package-name>
   ``
### 5. Best Practices

#### Code Organization
1. Keep components atomic and reusable
2. Use TypeScript for all new code
3. Implement proper component variants using cva
4. Follow the established folder structure

#### Component Development
1. Always include proper TypeScript types
2. Use React.forwardRef for component refs
3. Include displayName for debugging
4. Implement proper prop validation
5. Use CSS variables for theming

#### Styling
1. Use Tailwind classes
2. Implement variants using class-variance-authority
3. Use the `cn()` utility for class merging
4. Follow the "new-york" style system

### 6. Common Tasks

#### Adding a New shadcn/ui Component
1. Navigate to UI package:
   ```bash
   cd packages/ui
   ```
2. Add component using shadcn/ui CLI
3. Adjust imports to use workspace paths
4. Add to component exports

#### Creating Custom Components
1. Create component directory
2. Implement component following established patterns
3. Add to exports
4. Update types if necessary

### 7. Troubleshooting

#### Common Issues
1. Import Resolution
   - Check alias paths in tsconfig.json
   - Verify package.json exports
   - Ensure proper workspace references

2. Styling Issues
   - Verify Tailwind configuration
   - Check CSS variable definitions
   - Confirm class merging

3. Type Errors
   - Check TypeScript configuration
   - Verify dependency versions
   - Ensure proper type exports

#### Development Tips
1. Use VSCode for best TypeScript support
2. Enable ESLint for code quality
3. Test components in isolation
4. Keep components properly documented

# Backend Actions & Agents
Learn how to enable your Copilot to take actions in the backend.
### [Find your CopilotRuntime](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#find-your-copilotruntime>)
The starting point for backend actions is the `CopilotRuntime` you setup during quickstart (the CopilotKit backend endpoint). For a refresher, see [Self-Hosting](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</guides/self-hosting>) (or alternatively, revisit the [quickstart](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</quickstart>)).
Backend actions can return (and stream) values -- which will be piped back to the Copilot system and which may result in additional agentic action.
### [Integrate your backend actions](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#integrate-your-backend-actions>)
Choose any of the methods below to integrate backend actions into your Copilot system. **They can be mixed-and-matched as desired.**
TypeScript / Node.jsLangChain JSLangServePython + Remote Actions
When you initialize the `CopilotRuntime`, you can provide backend actions in the same format as frontend Copilot actions.
**Note that`actions` is not merely an array of actions, but rather a _generator_ of actions.** This generator takes `properties` and `url` as input -- which means you can customize which backend actions are made available according to the current frontend URL, as well as custom properties you can pass from the frontend.
/api/copilotkit/route.ts
```
const runtime = new CopilotRuntime({
 actions: ({properties, url}) => {
  // You can use the input parameters to the actions generator to expose different backend actions to the Copilot at different times: 
  // `url` is the current URL on the frontend application.
  // `properties` contains custom properties you can pass from the frontend application.
  return [
   {
    name: "fetchNameForUserId",
    description: "Fetches user name from the database for a given ID.",
    parameters: [
     {
      name: "userId",
      type: "string",
      description: "The ID of the user to fetch data for.",
      required: true,
     },
    ],
    handler: async ({userId}: {userId: string}) => {
     // do something with the userId
     // return the user data
     return {
      name: "Darth Doe",
     };
    },
   },
  ]
 }
});
// ... define the route using the CopilotRuntime.

```
# The `useCopilotReadable` Hook:

## [The `useCopilotReadable` Hook](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#the-usecopilotreadable-hook>)

For your copilot to best answer your users' needs, you will want to provide it with **context-specific** , **user-specific** , and oftentimes **realtime** data. CopilotKit makes it easy to do so.

### Example:

YourComponent.tsx
```
"use client"
import { useCopilotReadable } from "@copilotkit/react-core";
import { useState } from 'react';
export function YourComponent() {
 // Create colleagues state with some sample data
 const [colleagues, setColleagues] = useState([
  { id: 1, name: "John Doe", role: "Developer" },
  { id: 2, name: "Jane Smith", role: "Designer" },
  { id: 3, name: "Bob Wilson", role: "Product Manager" }
 ]);
 // Define Copilot readable state
 useCopilotReadable({
  description: "The current user's colleagues",
  value: colleagues,
 });
 return (
  // Your custom UI component
  <>...</>
 );
}
```

---

**title: "Copilot Suggestions Guide"**

**summary: "This document provides access to feedback channels, signing up for Copilot Cloud, and additional resources related to Copilot Kit."**

# Copilot Suggestions

Learn how to auto-generate suggestions in the chat window based on real time application state.
useCopilotChatSuggestions is experimental. The interface is not final and can change without notice.
`useCopilotChatSuggestions`[](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</reference/hooks/useCopilotChatSuggestions>) is a React hook that generates suggestions in the chat window based on real time application state.
![](https://docs-qsp8t297y-copilot-kit.vercel.app/images/use-copilot-chat-suggestions/use-copilot-chat-suggestions.gif)
### [Simple Usage](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#simple-usage>)
```
import { useCopilotChatSuggestions } from "@copilotkit/react-ui"; 
export function MyComponent() {
 useCopilotChatSuggestions(
  {
   instructions: "Suggest the most relevant next actions.",
   minSuggestions: 1,
   maxSuggestions: 2,
  },
  [relevantState],
 );
}
```

### [Dependency Management](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#dependency-management>)
```
import { useCopilotChatSuggestions } from "@copilotkit/react-ui";
export function MyComponent() {
 useCopilotChatSuggestions(
  {
   instructions: "Suggest the most relevant next actions.",
   minSuggestions: 1,
   maxSuggestions: 2,
  },
  [relevantState], 
 );
}

``` 

title: "Using Copilot Suggestions in Next.js with App Router"
summary: "This section explains how to generate suggestions based on specific instructions by monitoring `relevantState`. When using Next.js's App Router, it is necessary to specify the directive `"use client"` at the top of any component file that employs this client-side hook. Additionally, it advises checking the `useCopilotChatSuggestions` reference for further details."
metadata: {"source": "pydantic_ai_docs", "url_path": "/guides/copilot-suggestions", "chunk_size": 1703, "crawled_at": "2025-02-03T13:12:36.279534+00:00"}
created_at: "2025-02-03T13:12:37.550763+00:00"
---

```

In the example above, the suggestions are generated based on the given instructions. The hook monitors `relevantState`, and updates suggestions accordingly whenever it changes.
### [Specify `"use client"` (Next.js App Router)](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#specify-use-client-nextjs-app-router>)
This is only necessary if you are using Next.js with the App Router.
YourComponent.tsx
"use client"
```

Like other React hooks such as `useState` and `useEffect`, this is a **client-side** hook. If you're using Next.js with the App Router, you'll need to add the `"use client"` directive at the top of any file using this hook.

---

**title: "Custom AI Assistant Behavior  Guide"**

**summary: "This document provides instructions on how to customize the behavior of an AI assistant using the Copilot Kit framework. It includes links for feedback, community support, and registration for Copilot Cloud."**

**Passing the instructions parameter (Recommended)**

# **Customize AI Assistant Behavior**

Learn how to customize the behavior of your AI assistant.
## [Passing the `instructions` parameter (Recommended)](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#passing-the-instructions-parameter-recommended>)
The `instructions` parameter is the recommended way to customize AI assistant behavior. It will remain compatible with performance optimizations to the CopilotKit platform.
It can be customized for **Copilot UI** as well as **programmatically** :
Copilot UIHeadless UI & Programmatic Control
### [Copilot UI](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#copilot-ui>)
Copilot UI components implements `CopilotChatProps`, which accepts an `instructions` property:
CustomCopilot.tsx
```
import { CopilotChat, CopilotChatProps } from "@copilotkit/react-ui";
const CustomCopilot: React.FC<CopilotChatProps> = () => (
 <CopilotChat
  instructions="You are a helpful assistant specializing in tax preparation. Provide concise and accurate answers to tax-related questions."
  labels={{
   title: "Tax Preparation Assistant",
   initial: "How can I help you with your tax preparation today?",
  }}
 />
);

---

**title: "Overwriting the Default `makeSystemMessage`"**
**summary: "This section discusses the option to overwrite the default `makeSystemMessage` function for complete control over the system message. It is advised to first review CopilotKit's existing system message before proceeding, as overwriting is not recommended due to potential interference with optimizations."**


## [Overwriting the default `makeSystemMessage` (Not Recommended)](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#overwriting-the-default-makesystemmessage-not-recommended>)
For cases requiring complete control over the system message, you can use the `makeSystemMessage` function. We highly recommend reading CopilotKit's default system message before deciding to overwrite it, which can be found [here](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<https:/github.com/CopilotKit/CopilotKit/blob/e48a34a66bb4dfd210e93dc41eee7d0f22d1a0c4/CopilotKit/packages/react-core/src/hooks/use-copilot-chat.ts#L240-L258>).
This approach is **not recommended** as it may interfere with more advanced optimizations we are experimenting with, but it is available should you need it.
Copilot UIHeadless UI
### [Copilot UI](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#copilot-ui-1>)
```
import { CopilotChat } from "@copilotkit/react-ui";
const CustomCopilot: React.FC = () => (
 <CopilotChat
  instructions="You are a knowledgeable tax preparation assistant. Provide accurate and concise answers to tax-related questions, guiding users through the tax filing process."
  labels={{
   title: "Tax Preparation Assistant",
   initial: "How can I assist you with your taxes today?",
  }}
  makeSystemMessage={myCustomTaxSystemMessage} 
 />
);
```

---

**title: "Copilot Textarea Guide"**
**summary: "This section provides navigation to feedback channels, Copilot Cloud sign-up, and links to relevant resources and community platforms for the Copilot Kit."**

Install @copilotkit/react-textarea
# Copilot Textarea
Learn how to use the Copilot Textarea for AI-powered autosuggestions.
![](https://docs-qsp8t297y-copilot-kit.vercel.app/images/CopilotTextarea.gif)
`<CopilotTextarea>` is a React component that acts as a drop-in replacement for the standard `<textarea>`, offering enhanced autocomplete features powered by AI. It is context-aware, integrating seamlessly with the `useCopilotReadable`[](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</reference/hooks/useCopilotReadable>) hook to provide intelligent suggestions based on the application context.
In addition, it provides a hovering editor window (available by default via `Cmd + K` on Mac and `Ctrl + K` on Windows) that allows the user to suggest changes to the text, for example providing a summary or rephrasing the text.
This guide assumes you have completed the [quickstart](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</quickstart>) and have successfully set up CopilotKit.
### [Install `@copilotkit/react-textarea`](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#install-copilotkitreact-textarea>)
npmpnpmyarnbun
```
npm install @copilotkit/react-textarea
```

### [Import Styles](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#import-styles>)
Import the default styles in your root component (typically `layout.tsx`) :
layout.tsx

---

title: "Using the CopilotTextarea Component"
summary: "This documentation chunk introduces the `CopilotTextarea` component, provides an example of its implementation in a React component, and includes details on configuring autosuggestions for a better user experience."


```
import "@copilotkit/react-textarea/styles.css";
```

### [Add `CopilotTextarea` to Your Component](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/<#add-copilottextarea-to-your-component>)
Below you can find several examples showing how to use the `CopilotTextarea` component in your application.
Example 1Example 2
TextAreaComponent.tsx
```
import { FC, useState } from "react";
import { CopilotTextarea } from '@copilotkit/react-textarea';
const ExampleComponent: FC = () => {
 const [text, setText] = useState<string>('');
 return (
  <CopilotTextarea
   className="w-full p-4 border border-gray-300 rounded-md"
   value={text}
   onValueChange={setText}
   autosuggestionsConfig={{
    textareaPurpose: "the body of an email message",
    chatApiConfigs: {},
   }}
  />
 );
};

```

---

# Saving and restoring messages
Learn how to save and restore message history.
See [Loading Message History](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/</coagents/chat-ui/loading-message-history>) for an automated way to load the chat history.
As you're building agentic experiences, you may want to persist the user's chat history across runs. One way to do this is through the use of `localstorage` where chat history is saved in the browser. In this guide we demonstrate how you can store the state into `localstorage` and how it can be inserted into the agent.
The following example shows how to save and restore your message history using `localStorage`:
```
import { useCopilotMessagesContext } from "@copilotkit/react-core";
import { ActionExecutionMessage, ResultMessage, TextMessage } from "@copilotkit/runtime-client-gql";
const { messages, setMessages } = useCopilotMessagesContext();
// save to local storage when messages change
useEffect(() => {
 if (messages.length !== 0) {
  localStorage.setItem("copilotkit-messages", JSON.stringify(messages));
 }
}, [JSON.stringify(messages)]);
// initially load from local storage
useEffect(() => {
 const messages = localStorage.getItem("copilotkit-messages");
 if (messages) {
  const parsedMessages = JSON.parse(messages).map((message: any) => {
   if (message.type === "TextMessage") {
    return new TextMessage({
     id: message.id,
     role: message.role,
     content: message.content,
     createdAt: message.createdAt,
    });
   } else if (message.type === "ActionExecutionMessage") {
    return new ActionExecutionMessage({
     id: message.id,
     name: message.name,
     scope: message.scope,
     arguments: message.arguments,
     createdAt: message.createdAt,
    });
   } else if (message.type === "ResultMessage") {
    return new ResultMessage({
     id: message.id,
     actionExecutionId: message.actionExecutionId,
     actionName: message.actionName,
     result: message.result,
     createdAt: message.createdAt,
    });
   } else {
    throw new Error(`Unknown message type: ${message.type}`);
   }
  });
  setMessages(parsedMessages);
 }
}, []);
```


# useCopilotAction
The useCopilotAction hook allows your copilot to take action in the app.
![](https://docs-qsp8t297y-copilot-kit.vercel.app/images/use-copilot-action/useCopilotAction.gif)
`useCopilotAction` is a React hook that you can use in your application to provide custom actions that can be called by the AI. Essentially, it allows the Copilot to execute these actions contextually during a chat, based on the user’s interactions and needs.
Here's how it works:
Use `useCopilotAction` to set up actions that the Copilot can call. To provide more context to the Copilot, you can provide it with a `description` (for example to explain what the action does, under which conditions it can be called, etc.).
Then you define the parameters of the action, which can be simple, e.g. primitives like strings or numbers, or complex, e.g. objects or arrays.
Finally, you provide a `handler` function that receives the parameters and returns a result. CopilotKit takes care of automatically inferring the parameter types, so you get type safety and autocompletion for free.
To render a custom UI for the action, you can provide a `render()` function. This function lets you render a custom component or return a string to display.


# **`useCoAgent`**

The `useCoAgent` hook allows you to share state bidirectionally between your application and the agent.

Usage of this hook assumes some additional setup in your application, for more information on that see the CoAgents [getting started guide](https://docs-qsp8t297y-copilot-kit.vercel.app/reference/hooks/</coagents/quickstart>).
![CoAgents demonstration](https://docs-qsp8t297y-copilot-kit.vercel.app/images/coagents/SharedStateCoAgents.gif)
This hook is used to integrate an agent into your application. With its use, you can render and update the state of an agent, allowing for a dynamic and interactive experience. We call these shared state experiences agentic copilots, or CoAgents for short.

## [Usage](https://docs-qsp8t297y-copilot-kit.vercel.app/reference/hooks/<#usage>)

### [Simple Usage](https://docs-qsp8t297y-copilot-kit.vercel.app/reference/hooks/<#simple-usage>)
```
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

`useCoAgent` returns an object with the following properties:
```
const {
 name,   // The name of the agent currently being used.
 nodeName, // The name of the current LangGraph node.
 state,  // The current state of the agent.
 setState, // A function to update the state of the agent.
 running, // A boolean indicating if the agent is currently running.
 start,  // A function to start the agent.
 stop,   // A function to stop the agent.
 run,   // A function to re-run the agent. Takes a HintFunction to inform the agent why it is being re-run.
} = agent;
```

Finally we can leverage these properties to create reactive experiences with the agent!
```
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

## [Parameters](https://docs-qsp8t297y-copilot-kit.vercel.app/reference/hooks/<#parameters>)

optionsUseCoagentOptions<T>required
The options to use when creating the coagent.
namestringrequired
The name of the agent to use.
initialStateT | any
The initial state of the agent.
stateT | any
State to manage externally if you are using this hook with external state management.
setState(newState: T | ((prevState: T | undefined) => T)) => void
A function to update the state of the agent if you are using this hook with external state management.


---

title: "Intermediate State Streaming in Coagents"
summary: "This document explores advanced features of intermediate state streaming in Coagents, discussing how it can enhance data processing and feedback mechanisms. It emphasizes the integration capabilities and the benefits of signing up for additional services, such as Copilot Cloud."


# Emit intermediate state
## [Background](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/advanced/<#background>)
A LangGraph agent's state updates discontinuosly; only across node transitions in the graph. But even a _single node_ in the graph often takes many seconds to run and contain sub-steps of interest to the user.
**Agent-native applications** reflect to the end-user what the agent is doing **_as continuously possible._**
A few key benefits: • **_Keeps users engaged_** by avoiding long loading indicators • **_Builds trust_** by demonstrating what the agent is working on • Supports [**_agent steering_**](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/advanced/</coagents/advanced/agent-steering>) - allowing users to course-correct the agent if needed
## [Install the CopilotKit SDK](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/advanced/<#install-the-copilotkit-sdk>)
You can use CopilotKit with vanilla LangGraph, but its features will be limited to basic functionality. For the full range of capabilities and seamless integration, you’ll need our LangGraph SDKs.
PythonTypeScript
Poetrypipconda
```
poetry add copilotkit
```

## [Emit intermediate state manually:](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/advanced/<#emit-intermediate-state-manually>)
PythonTypeScript
```
from copilotkit.langgraph import copilotkit_emit_state
async def some_langgraph_node(state: AgentState, config: RunnableConfig):
  # ... whatever you want to docs
  # modify the state as you please:
  state["logs"].append({
    "message": f"Downloading {resource['url']}",
    "done": False
  })
  await copilotkit_emit_state(config, state) # manually emit the updated state
  # ... keep running your graph
  # finally, return the state from the node as usual. The returned state is the ultimate source of truth
  return state
```

## [Emit intermediate state from LLM tool calls:](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/advanced/<#emit-intermediate-state-from-llm-tool-calls>)
A LangGraph node's end-state is often driven by an LLM's response.
With `emit_intermediate_state`, you can have CopilotKit treat a given tool call's argument as intermediate state. The frontend will see the state updates as they stream in.
In this example, we're asking for the `outline` state-key to be streamed as it comes in from the `set_outline` tool.
PythonTypeScript
```
from copilotkit.langgraph import copilotkit_customize_config
async def streaming_state_node(state: AgentState, config: RunnableConfig):
  modifiedConfig = copilotkit_customize_config( 
    config,
    # this will stream the `set_outline` tool-call's `outline` argument, *as if* it was the `outline` state parameter.
    # i.e.:
    # the tool call: set_outline(outline: string)
    # the state: { ..., outline: "..." }
    emit_intermediate_state=[
      {
        "state_key": "outline", # the name of the key on the agent state we want to interact with
        "tool": "set_outline", # the name of the tool call
        "tool_argument": "outline", # the name of the argument on the tool call to treat as intermediate state.
      }
    ]
  )
  # Pass the modified configuration to LangChain as an argument
  response = await ChatOpenAI(model="gpt-4o").ainvoke(
    [
      *state["messages"]
    ],
    modifiedConfig # pass the modified config
  )
  # Make sure to ALSO return the state from the node.
  return {
    *state,
    "outline": outline # <- the actual state object is always the final source of truth when the node ends
  }

  ```
---

# Streaming and Tool Calls
CoAgents support streaming your messages and tool calls to the frontend.
If you'd like to change how LangGraph agents behave as CoAgents you can utilize our Copilotkit SDK which provides a collection of functions and utilities for interacting with the agent's state or behavior. One example of this is the Copilotkit config which is a wrapper of the LangGraph `config` object. This allows you to extend the configuration of your LangGraph nodes to change how LangGraph and Copilotkit interact with each other. This allows you to change how messages and tool calls are emitted and streamed to the frontend.

##[Message Streaming](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/concepts/<#message-streaming>)
If you did not change anything in your LangGraph node, message streaming will be on by default. This allows for a message to be streamed to Copilotkit as it is being generated, allowing for a more responsive experience. However, you can disable this if you want to have the message only be sent after the agent has finished generating it.
```
config = copilotkit_customize_config(
  config,
  # True or False
  emit_messages=False,
)
```
## [Emitting Tool Calls](https://docs-qsp8t297y-copilot-kit.vercel.app/coagents/concepts/<#emitting-tool-calls>)
Emission of tool calls are off by default. This means that tool calls will not be sent to Copilotkit for processing and rendering. However, within a node you can extend the LangGraph `config` object to emit tool calls to Copilotkit. This is useful in situations where you may to emit what a potential tool call will look like prior to being executed.
```
config = copilotkit_customize_config(
  config,
  # Can set to True, False, or a list of tool call names to emit.
  emit_tool_calls=["tool_name"],
)
```

---

title: "Customizing Built-In UI Components in Copilot Kit"
summary: "This guide explains how to customize the built-in UI components of the Copilot Kit. It provides instructions and resources for tailoring the user interface to better fit user needs and preferences."

CSS Variables
Customize Look & Feel
# Customize the built-in components
Customize the look and feel of the built-in Copilot UI components.


#[CSS Variables](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#css-variables>)
CopilotKit uses CSS variables to make it easy to customize the look and feel of the components. Hover over the interactive UI elements below to see the available CSS variables.
Hover over the interactive UI elements below to see the available CSS variables.
Close CopilotKit
CopilotKit
Hi you! 👋 I can help you create a presentation on any topic.
Hello CopilotKit!
You can override / customize the CSS, customize the icons, customize labels, swap out react components, and use headless UI. See below.
### [CSS Variables (Easiest)](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#css-variables-easiest>)
The easiest way to change the colors is to override CopilotKit CSS variables. You can simply wrap CopilotKit in a div and override the CSS variables. CopilotKitCSSProperties will have the variable names as types:
```
<div
 style={
  {
   "--copilot-kit-primary-color": "#222222",
  } as CopilotKitCSSProperties
 }
>
 <CopilotSidebar .../>
</div>
```

### [Custom CSS](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#custom-css>)
In addition to customizing the colors, the CopilotKit CSS is structured to easily allow customization via CSS classes.
Have a look at the CSS classes defined [here](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<https:/github.com/CopilotKit/CopilotKit/blob/main/CopilotKit/packages/react-ui/src/css/>).
For example:
globals.css
```
.copilotKitButton {
 border-radius: 0;
}
```

## [Custom Icons](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#custom-icons>)
You can customize the icons by passing the `icons` property to the `CopilotSidebar`, `CopilotPopup` or `CopilotChat` component.
Below is a list of customizable icons:
  * `openIcon`: The icon to use for the open chat button.
  * `closeIcon`: The icon to use for the close chat button.
  * `headerCloseIcon`: The icon to use for the close chat button in the header.
  * `sendIcon`: The icon to use for the send button.
  * `activityIcon`: The icon to use for the activity indicator.
  * `spinnerIcon`: The icon to use for the spinner.
  * `stopIcon`: The icon to use for the stop button.
  * `regenerateIcon`: The icon to use for the regenerate button.
  * `pushToTalkIcon`: The icon to use for push to talk.

## [Custom Labels](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#custom-labels>)
To customize labels, pass the `labels` property to the `CopilotSidebar`, `CopilotPopup` or `CopilotChat` component.
Below is a list of customizable labels:
  * `initial`: The initial message(s) to display in the chat window.
  * `title`: The title to display in the header.
  * `placeholder`: The placeholder to display in the input.
  * `stopGenerating`: The label to display on the stop button.
  * `regenerateResponse`: The label to display on the regenerate button.


## [Headless UI (Full Customizability)](https://docs-qsp8t297y-copilot-kit.vercel.app/guides/custom-look-and-feel/<#headless-ui-full-customizability>)

## Customize Look & Feel

**Headless UI (Fully Custom)**

Fully customize your Copilot's UI from the ground up using headless UI
The built-in Copilot UI can be customized in many ways -- both through css and by passing in custom sub-components.
CopilotKit also offers **fully custom headless UI** , through the `useCopilotChat` hook. Everything built with the built-in UI (and more) can be implemented with the headless UI, providing deep customizability.

### EXAMPLE
```
import { useCopilotChat } from "@copilotkit/react-core";
import { Role, TextMessage } from "@copilotkit/runtime-client-gql";
export function CustomChatInterface() {
 const {
  visibleMessages,
  appendMessage,
  setMessages,
  deleteMessage,
  reloadMessages,
  stopGeneration,
  isLoading,
 } = useCopilotChat();
 const sendMessage = (content: string) => {
  appendMessage(new TextMessage({ content, role: Role.User }));
 };
 return (
  <div>
   {/* Implement your custom chat UI here */}
  </div>
 );
}
```