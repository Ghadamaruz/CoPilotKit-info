# CopilotKit Generative UI Guide

## Table of Contents
1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Implementation Examples](#implementation-examples)
4. [Backend Integration](#backend-integration)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)

## Overview

CopilotKit Generative UI enables the creation of dynamic, AI-driven user interfaces that seamlessly integrate with your Python FastAPI backend. This guide covers the implementation patterns and best practices.

## Component Architecture

### Basic Structure
```typescript
"use client"
import { useCopilotAction } from "@copilotkit/react-core";
import { useState } from "react";

export function AIComponent() {
  useCopilotAction({
    name: "actionName",
    description: "Action description for AI understanding",
    parameters: [/* parameter definitions */],
    handler: async (args) => {/* API call logic */},
    render: ({ status, result }) => {/* UI rendering logic */}
  });

  return null;
}
```

### Key Components
1. **Action Definition**: Defines how the AI understands and triggers the component
2. **Parameter Handling**: Structures the data required for the action
3. **API Integration**: Manages communication with the backend
4. **UI Rendering**: Controls the visual representation of results

## Implementation Examples

### 1. Research Results Component
```typescript
// frontend/components/research/research-viewer.tsx
"use client"
import { useCopilotAction } from "@copilotkit/react-core";

export function ResearchViewer() {
  useCopilotAction({
    name: "showResearchResults",
    description: "Displays research results from the AI researcher",
    parameters: [
      {
        name: "query",
        type: "string",
        description: "The research query",
        required: true
      },
      {
        name: "maxResults",
        type: "number",
        description: "Maximum number of results to display",
        required: false
      }
    ],
    handler: async ({ query, maxResults = 5 }) => {
      const response = await fetch('/api/copilotkit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query, maxResults })
      });
      return await response.json();
    },
    render: ({ status, result }) => {
      if (status === 'executing') {
        return <ResearchLoadingState />;
      }
      return status === 'complete' ? (
        <ResearchResultsCard results={result} />
      ) : (
        <div>Failed to load research results</div>
      );
    }
  });

  return null;
}
```

### 2. Results Display Component
```typescript
// frontend/components/research/research-results-card.tsx
interface ResearchResult {
  summary: string;
  references: string[];
  confidence: number;
}

function ResearchResultsCard({ results }: { results: ResearchResult }) {
  return (
    <div className="rounded-lg border p-4 bg-white shadow-sm">
      <h3 className="text-lg font-semibold mb-2">Research Summary</h3>
      <p className="text-gray-700 mb-4">{results.summary}</p>
      
      <h4 className="font-medium mb-2">References</h4>
      <ul className="space-y-1">
        {results.references.map((ref, index) => (
          <li key={index} className="text-sm text-gray-600">
            {ref}
          </li>
        ))}
      </ul>
      
      <div className="mt-4 text-sm text-gray-500">
        Confidence: {results.confidence * 100}%
      </div>
    </div>
  );
}
```

### 3. Loading State Component
```typescript
// frontend/components/research/research-loading-state.tsx
function ResearchLoadingState() {
  return (
    <div className="rounded-lg border p-4 bg-white shadow-sm animate-pulse">
      <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
      <div className="space-y-2">
        <div className="h-3 bg-gray-200 rounded"></div>
        <div className="h-3 bg-gray-200 rounded w-5/6"></div>
      </div>
    </div>
  );
}
```

## Backend Integration

### FastAPI Implementation
```python
# backend/agent/ai_researcher/demo.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

class ResearchRequest(BaseModel):
    query: str
    maxResults: Optional[int] = 5

class ResearchResponse(BaseModel):
    summary: str
    references: List[str]
    confidence: float

@app.post("/copilotkit")
async def handle_research(request: ResearchRequest) -> ResearchResponse:
    try:
        # Your AI researcher logic here
        result = await ai_researcher.research(
            query=request.query,
            max_results=request.maxResults
        )
        
        return ResearchResponse(
            summary=result.summary,
            references=result.references,
            confidence=result.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Advanced Features

### 1. Human-in-the-Loop Implementation
```typescript
useCopilotAction({
  name: "confirmResearch",
  description: "Confirm research results before proceeding",
  parameters: [
    {
      name: "summary",
      type: "string",
      description: "Research summary to confirm",
      required: true
    }
  ],
  renderAndWaitForResponse: ({ args, respond }) => {
    const { summary } = args;
    return (
      <div className="rounded-lg border p-4 bg-white shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Confirm Research</h3>
        <p className="text-gray-700 mb-4">{summary}</p>
        
        <div className="flex gap-2">
          <button
            onClick={() => respond('confirmed')}
            className="px-4 py-2 bg-green-500 text-white rounded"
          >
            Confirm
          </button>
          <button
            onClick={() => respond('rejected')}
            className="px-4 py-2 bg-red-500 text-white rounded"
          >
            Reject
          </button>
        </div>
      </div>
    );
  }
});
```

### 2. Real-time Updates
```typescript
useCopilotAction({
  name: "streamResearch",
  // ... other configurations
  handler: async ({ query }) => {
    const ws = new WebSocket('ws://localhost:8000/ws/research');
    ws.send(JSON.stringify({ query }));
    
    return new Promise((resolve) => {
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === 'complete') {
          resolve(data.results);
          ws.close();
        }
      };
    });
  }
});
```

## Best Practices

### 1. Type Safety
- Use TypeScript interfaces that match Python Pydantic models
- Implement proper error handling for type mismatches
- Validate data shapes at runtime

### 2. Error Handling
- Implement graceful fallbacks for failed requests
- Show meaningful error messages to users
- Log errors for debugging

### 3. Performance
- Implement proper loading states
- Use debouncing for frequent updates
- Optimize re-renders

### 4. Accessibility
- Include proper ARIA labels
- Ensure keyboard navigation
- Maintain color contrast ratios

### 5. State Management
- Handle all possible states (loading, success, error)
- Implement proper cleanup in useEffect
- Manage side effects properly

## Usage

1. Register components in your app:
```typescript
// frontend/app/layout.tsx
import { CopilotKit } from "@copilotkit/react-core";
import { ResearchViewer } from "@/components/research/research-viewer";

export default function RootLayout({ children }) {
  return (
    <CopilotKit>
      <ResearchViewer />
      {children}
    </CopilotKit>
  );
}
```

2. Natural Language Commands:
   - "Show me research about climate change"
   - "Research the impact of AI on healthcare"
   - "Analyze the latest developments in quantum computing"

## Troubleshooting

1. **Component Not Rendering**
   - Verify component registration
   - Check action name matches
   - Validate parameter types

2. **Backend Communication Issues**
   - Verify API endpoint configuration
   - Check CORS settings
   - Validate request/response formats

3. **State Management Problems**
   - Review cleanup functions
   - Check for memory leaks
   - Verify state updates

---

