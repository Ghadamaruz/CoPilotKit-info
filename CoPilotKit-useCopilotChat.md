# useCopilotChat

`useCopilotChat` is a React hook that lets you directly interact with the Copilot instance. Use to implement a fully custom UI (headless UI) or to programmatically interact with the Copilot instance managed by the default UI.

## Usage

### Simple Usage

```tsx
import { useCopilotChat } from "@copilokit/react-core";
import { Role, TextMessage } from "@copilokit/runtime-client-api";

export function YourComponent() {
  const { appendMessage } = useCopilotChat();
  
  appendMessage(
    new TextMessage({
      content: "Hello World",
      role: Role.User,
    }),
  );
  
  // optionally, you can append a message without running chat completion
  appendMessage(yourMessage, { followup: false });
}
```

`useCopilotChat` returns an object with the following properties:

```tsx
const {
  visibleMessages, // An array of messages that are currently visible in the chat.
  appendMessage, // A function to append a message to the chat.
  setMessages, // A function to set the messages in the chat.
  deleteMessage, // A function to delete a message from the chat.
  reloadMessages, // A function to reload the messages from the API.
  stopGeneration, // A function to stop the generation of the next message.
  isLoading, // A boolean indicating if the chat is loading.
} = useCopilotChat();
```

## Parameters

### id `string`

A unique identifier for the chat. If not provided, a random one will be generated. When provided, the `useChat` hook with the same `id` will have shared state across components.

### headers `Record<string, string> | Headers`

HTTP headers to be sent with the API request.

### initialMessages `Message[]`

System messages of the chat. Defaults to an empty array.

### makeSystemMessage `SystemMessageFunction`

A function to generate the system message. Defaults to `defaultSystemMessage`.

---

**Previous**
useCopilotAdditionalInstructions

**Next**
useCopilotChatSuggestions