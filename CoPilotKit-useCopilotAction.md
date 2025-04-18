# useCopilotAction

The `useCopilotAction` hook allows your copilot to take action in the app.

![Example of useCopilotAction UI](image-url)

`useCopilotAction` is a React hook that lets you call in your application to provide custom actions that can be called by the AI. Essentially, it allows the copilot to execute these actions interactively during a chat, based on the user's interactions and needs.

Here's how it works:

1. Use `useCopilotAction` to set up actions that the Copilot can call. To provide more context to the Copilot, you can provide a name, a description, the inputs/params to expect (their the action does), what it returns/outputs it can be added, etc.)

2. Then you define the parameters of the action, which can be simple, e.g. primitives like strings or numbers, or complex, e.g. objects or arrays.

3. Finally, you define what happens when the action is called in response to a user message. CopilotKit takes care of automatically inferring the parameter types, so you get type safety and autocompletion for free.

To render a custom UI for the action, you can provide a `render` function. This function lets you render a custom component or return a string to display.

## Usage

### Simple Usage

```jsx
import { useCopilotAction } from "@copilokit/react-core";

export default function Component() {
  useCopilotAction({
    name: "greet",
    description: "Greet of the person by age group",
    parameters: [
      {
        name: "age",
        type: "number",
      },
    ],
    handler: async ({ age }) => {
      // ...
    },
  });
}
```

### Generative UI

This hooks enables you to dynamically generate UI elements and render them in the copilot chat. For more information, check out the [Generative UI docs](https://docs.copilokit.ai/pages/api/generativeUi).

## Parameters

### name `string` `required`

The function name relative to the Copilot. See [below](#).

### description `string` 

The name of the action.

### handler `(args: Parameters) => Promise<T>`

The handler of the action.

### description `string | ReactNode | (params: T) => ReactNode`

A description of the action. This is used to inform the Copilot on how to use the action.

### condition `"enabled" | "disabled" | "hidden"`

Use this property to control when the action is available to the Copilot. When set to `"hidden"`, the action is available only to trained agents.

### fileName `string?`

Default: './'.
Allows to import the result of a function call to the LLM which will then provide a follow-up response. Path: '/path_to_script'.

### parameters `Object[]?`

The parameters of the action. See [parameters](#).

### tags `string[]` `hidden`

The name of the parameters.

### type `string` `hidden`

The type of the parameter. One of:
- "boolean"
- "string"
- "number"
- "integer"
- "array"
- "object"
- "null"
- "function"
- "any"
- "undefined"
- "bigint"
- "symbol"

### description `string`

A description of the parameter. This is used to inform the Copilot on what this parameter is used for.

### type `string?`

For string parameters, you can provide an array of possible values.

### required `boolean`

Whether or not the parameter is required. Default is false.

### properties `{}`

If the parameter is of a complex type (e.g. `object` or `object[]`), this field lets you define the properties of the object. For example:

```tsx
// Parameters:
parameters: [{
  name: "data",
  type: "object",
  description: "The addresses included from the form.",
  properties: {
    home: {
      type: "object",
      properties: {
        street: {
          type: "string",
          description: "The street of the address.",
        },
      },
    },
    work: {
      type: "object",
      properties: {
        street: {
          type: "string",
          description: "The city of the address.",
        },
      },
    },
  },
}]
```

### render `string | ((data: ReturnType<typeof handler>) => JSX.Element)`

Render that lets you define a custom component or string to render the output. You can either pass a string or a function.

### output `"text/plain" | "text/markdown" | "application/json"`

- `"text/plain"`: Arguments are dynamically determined by the function, allowing you to adapt one of more than one
- `"text/markdown"`: The action handler is executing
- `"application/json"`: The action handler has completed execution

### show `{}`

This parameter should be true to show input in chat flow. When the param is `"text/plain"`, there are possible truncations.

### result `any`

This result describes the the action. It is only available when the action is `"completed"`.

### createAction<Params, Result> `(opts: ActionDefinition<Params, Result>) => UseActionResponse<Params, Result>`

This is similar to `useCopilotAction`, but provides a cleaner function that you could call before the hook happened. The function returned here creates function-valued props that could then be passed into 'useCopilotAction' to define what's passed to what function.

```tsx
const { handler, condition, description } = createAction({
  handler: async (params) => {
    // ...
  },
})

// Elsewhere
useCopilotAction({
  name: "actionName",
  handler,
  condition,
  description,
})
```

- `"text/plain"`: Arguments are dynamically determined by the function, allowing you to adapt one of more than one
- `"text/markdown"`: The action handler is executing
- `"application/json"`: The action handler has completed execution

### show `{}`

This parameter should be true to show input in chat flow. When the param is `"text/plain"`, there are possible truncations.

### result `any`

A function that must be called with the user's response. The response will be passed as the result to the action handler, only available when status is `"text/plain"`.

### output `"application/json" | "text/plain"`

This result describes the the action. It is only available when the action is `"completed"`.

### actionContext `any`

An action-specific parameters object.