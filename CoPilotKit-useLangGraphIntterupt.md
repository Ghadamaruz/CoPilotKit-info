<think>I'll convert this image to markdown format. The image shows documentation for a hook called `useLangGraphInterrupt` from CopilotKit. I'll structure it with proper headings, code blocks, and formatting to match the original image content.</think>Here's the markdown conversion of the useLangGraphInterrupt documentation:

```markdown
# useLangGraphInterrupt

The useLangGraphInterrupt hook allows setting the generative UI to be displayed on LangGraph's interrupt event.

## Usage

### Simple Usage

![Example UI showing a chat interface with an interrupt popup](example-interrupt-ui.png)

useLangGraphInterrupt is a React hook that you can use in your application to provide custom UI to be rendered when using `interrupt` by LangGraph. Once an interrupt event is emitted, that hook would execute, allowing to receive user input with a user experience to your choice.

#### app/page.tsx

```typescript
import { useLangGraphInterrupt } from "@copilotkit/react-core";

// ...

const YourMainContent = () => {
  // ...
  // styles omitted for brevity
  useLangGraphInterrupt({
    render: ({ name, result }) => {
      <div>
        <p>{result.value}</p>
        <form onSubmit={() => {
          e.preventDefault();
          result.response(formEl.current?.response.value);
        }}>
          <input type="text" name="response" placeholder="Enter your response" />
          <button type="submit">Submit</button>
        </form>
      </div>
    }
  });
  // ...
  
  return <div>{/* ... */}</div>;
}
```

## Parameters

**action** `Action` *required*

The action to perform when an interrupt event is emitted. Either `handler` or `render` must be defined as arguments.

**name** `string` *required*

The name of the action.

**handler** `(args: LangGraphInterruptRenderProps<T>) => any | Promise<any>`

A handler to programmatically resolve the interrupt, or perform operations which result will be passed to the `render` method.

**render** `(props: LangGraphInterruptRenderProps<T>) => string | React.ReactElement`

Render lets you define a custom component or string to render when an interrupt event is emitted.

**dependencies** `any[]`

An optional array of dependencies.
```
