# CoPilotKit UI Components Guide

## Table of Contents
1. [Custom UI Components in Chat Window](#custom-ui-components-in-chat-window)
2. [Built-in Component Customization](#built-in-component-customization)
3. [Headless UI Implementation](#headless-ui-implementation)
4. [Self-Hosting Guide](#self-hosting-guide)

## Custom UI Components in Chat Window

### Basic Component Rendering
The `useCopilotAction` hook allows rendering custom UI components in the chat window.

```tsx
"use client"
import { useCopilotAction } from "@copilotkit/react-core"; 

export function YourComponent() {
  useCopilotAction({ 
    name: "showCalendarMeeting",
    description: "Displays calendar meeting information",
    parameters: [
      {
        name: "date",
        type: "string",
        description: "Meeting date (YYYY-MM-DD)",
        required: true
      },
      {
        name: "time",
        type: "string",
        description: "Meeting time (HH:mm)",
        required: true
      },
      {
        name: "meetingName",
        type: "string",
        description: "Name of the meeting",
        required: false
      }
    ],
    render: ({ status, args }) => {
      const { date, time, meetingName } = args;
      return status === 'inProgress' 
        ? <LoadingView />
        : <CalendarMeetingCardComponent {...{ date, time, meetingName }} />;
    },
  });

  return <></>;
}
```

### Data Fetching with Rendering
Combine handler and render methods for dynamic content:

```tsx
useCopilotAction({ 
  name: "showLastMeetingOfDay",
  description: "Displays the last calendar meeting for a given day",
  parameters: [
    {
      name: "date",
      type: "string",
      description: "Date to fetch the last meeting for (YYYY-MM-DD)",
      required: true
    }
  ],
  handler: async ({ date }) => {
    return await fetchLastMeeting(new Date(date));
  },
  render: ({ status, result }) => {
    if (status === 'executing' || status === 'inProgress') {
      return <LoadingView />;
    }
    return status === 'complete' 
      ? <CalendarMeetingCardComponent {...result} />
      : <div className="text-red-500">No meeting found</div>;
  },
});
```

### Human-in-the-Loop (HITL) Implementation
Use `renderAndWaitForResponse` for user interaction flows:

```tsx
useCopilotAction({ 
  name: "handleMeeting",
  description: "Handle a meeting by booking or canceling",
  parameters: [
    {
      name: "meeting",
      type: "string",
      description: "The meeting to handle",
      required: true,
    },
    {
      name: "date",
      type: "string",
      description: "The date of the meeting",
      required: true,
    },
    {
      name: "title",
      type: "string",
      description: "The title of the meeting",
      required: true,
    },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    const { meeting, date, title } = args;
    return (
      <MeetingConfirmationDialog
        meeting={meeting}
        date={date}
        title={title}
        onConfirm={() => respond?.('meeting confirmed')}
        onCancel={() => respond?.('meeting canceled')}
      />
    );
  },
});
```

### String Rendering
Simple message rendering:

```tsx
useCopilotAction({ 
  name: "simpleAction",
  description: "A simple action with string rendering",
  parameters: [
    {
      name: "taskName",
      type: "string",
      description: "Name of the task",
      required: true,
    },
  ],
  handler: async ({ taskName }) => {
    return await longRunningOperation(taskName);
  },
  render: ({ status, result }) => {
    return status === "complete" ? result : "Processing...";
  },
});
```

### Catch-all Rendering
Handle undefined actions:

```tsx
useCopilotAction({
  name: "*",
  render: ({ name, args, status, result }) => {
    return <div>Rendering action: {name}</div>;
  },
});
```

## Built-in Component Customization

### CSS Variables
Override default styles using CSS variables:

```tsx
<div
  style={
    {
      "--copilot-kit-primary-color": "#222222",
    } as CopilotKitCSSProperties
  }
>
  <CopilotSidebar />
</div>
```

### Custom CSS Classes
Add custom CSS classes for styling:

```css
/* globals.css */
.copilotKitButton {
  border-radius: 0;
}
```

### Customizable Elements
1. **Icons**
   - openIcon
   - closeIcon
   - headerCloseIcon
   - sendIcon
   - activityIcon
   - spinnerIcon
   - stopIcon
   - regenerateIcon
   - pushToTalkIcon

2. **Labels**
   - initial
   - title
   - placeholder
   - stopGenerating
   - regenerateResponse

## Headless UI Implementation

Create fully custom UI using the `useCopilotChat` hook:

```tsx
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
      {/* Custom chat UI implementation */}
    </div>
  );
}
```

## Self-Hosting Guide

### 1. Create an Endpoint

#### Environment Setup
```env
OPENAI_API_KEY=your_api_key_here
```

#### Next.js App Router Implementation
```typescript
// app/api/copilotkit/route.ts
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';

const serviceAdapter = new OpenAIAdapter();
const runtime = new CopilotRuntime();

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: '/api/copilotkit',
  });

  return handleRequest(req);
};
```

### 2. Configure CopilotKit Provider

```tsx
// layout.tsx
import { ReactNode } from "react";
import { CopilotKit } from "@copilotkit/react-core"; 

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body> 
        <CopilotKit runtimeUrl="/api/copilotkit"> 
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}