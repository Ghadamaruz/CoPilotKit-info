# CopilotTextarea

An AI-powered textarea component for your applications, which serves as a drop-in replacement for regular textareas.

## Features

| Features | Current State |
| -------- | ------------- |
| AI completion | ✅ |
| Conversation history | ✅ |
| Feature 3 | ✅ |
| Feature 4 | Keyboard shortcut for autocomplete |

CopilotTextarea is a component aimed to have a similar experience to using GitHub Copilot, offering enhanced autocomplete features powered by AI. It's content-aware, integrating seamlessly with the [getResponseMessage](). Here it creates intelligent suggestions based on the application context.

In addition, it provides a hovering editor window (available by default for <kbd>⌘</kbd>+<kbd>K</kbd> and <kbd>Ctrl</kbd>+<kbd>K</kbd> shortcuts) that allows the user to suggest changes to the text, for example providing a summary or rephrasing the text.

## Example

```tsx
import { CopilotTextarea } from "@copilokit/react-textarea";
import { CopilotProvider } from "@copilokit/react-core";

function Example() {
  return (
    <CopilotProvider>
      <CopilotTextarea
        placeholder="Enter some text..."
      />
    </CopilotProvider>
  );
}
```

## Usage

### Install Dependencies

Make sure you have all of the [required dependencies](/guides/installation) installed.

```
npm install @copilokit/react-core @copilokit/react-textarea
```

### Usage

Use the CopilotTextarea component in your React application similarly to a standard `<textarea />`, with additional configuration for AI-powered features.

For example:

```tsx
import { useState } from "react";
import { CopilotProvider } from "@copilokit/react-core";
import { CopilotTextarea } from "@copilokit/react-textarea";

export default function Example() {
  const [value, setValue] = useState("");
  
  return (
    <CopilotProvider>
      <div className="w-full max-w-md mx-auto">
        <CopilotTextarea 
          className="textarea textarea-ghost" 
          value={value} 
          onChange={(e) => setValue(e.target.value)} 
          placeholder="Provide context or pieces of the template." 
          id="textarea"
          autoFocus={true}
          name="name"
        />
      </div>
    </CopilotProvider>
  );
}
```

## Look & Feel

By default, Copilokit components do not have any styles. You can import Copilokit's stylesheet at the root of your project:

```tsx
import "@copilokit/react-textarea/dist/style.css";
```

Or use your own styling:

```tsx
import { CopilotTextarea } from "@copilokit/react-textarea";

export default function Component() {
  return (
    <div>
      <CopilotTextarea />
    </div>
  );
}
```

For more information about how to customize the styles, check out the [Customize Look & Feel](#) guide.

## Properties

### showCompletions `boolean`

Determines whether the CopilotKit bubbles should be enabled. Default is `true`.

### completionThreshold `[number | undefined]`

Specifies the CSS index to apply to the autocomplete bar.

### suggestionOptions `[object | undefined]`

Specifies the CSS styles to apply to the suggestions list.

### surroundWithCursor `string`

A class name to apply to the editor element window.

### value `string`

The initial value of the textarea. Can be controlled via `initialValue`.

### initialValue `string`, `string[]` | `null`

Controls whether to show the name of the textarea changes.

### onChange `(value:string) => void`

Callback function when a change event is triggered on the textarea element.

### autoFocus `boolean`

The element to use to place the editor element window. Default is "root".

### editorComponent `[ReactNode | undefined]`

Configuration settings for the autocompletions feature. For full reference, check the [reference on GitHub](#).

### editorClassName `string`

The purpose of this field is plain text:

Example: "The body of the email explained"

### editorPlaceholder `[string | undefined]`

The real API configuration.

**NOTE**: The most precise specific is level one of `suggestionItemClassName` or `surroundWithCursor`.

### editorButtonLabel `[string | undefined]`

For full reference, please [see here](#).

### editorSubmitLabel `[string | undefined]`

For full reference, please [see here](#).

### disabled `boolean`

Whether the textarea is disabled.

### acceptCompletions `boolean`

Whether to enable the CopilotKit finishing.

### afterContentUpdate `[string | undefined]`

Specifies the CSS styles to apply to the suggestions bar.

### suggestionOptions `[object | undefined]`

Specifies the CSS styles to apply to the suggestions list.

### surroundWithCursor `string`

A class name to apply to the editor element window.

### value `string`

The initial value of the textarea. Can be controlled via `initialValue`.

### initialValue `string`, `string[]` | `null`

Controls whether to show the name of the textarea changes.

### onChange `(value:string) => void`

Callback function when a change event is triggered on the textarea element.