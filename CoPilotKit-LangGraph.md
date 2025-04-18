# **CoAgent Routing Example** :

This file is a merged representation of the entire codebase, combined into a single document.

================================================================
Directory Structure
================================================================
agent/langgraph.json
agent/my_agent/demo.py
agent/my_agent/email_agent.py
agent/my_agent/joke_agent.py
agent/my_agent/model.py
agent/my_agent/pirate_agent.py
agent/pyproject.toml
README.md
ui/.eslintrc.json
ui/.gitignore
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/src/app/api/copilotkit/route.ts
ui/src/app/components/ModelSelector.tsx
ui/src/app/components/ui/select.tsx
ui/src/app/globals.css
ui/src/app/layout.tsx
ui/src/app/lib/model-selector-provider.tsx
ui/src/app/lib/utils.ts
ui/src/app/page.tsx
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "joke_agent": "./my_agent/joke_agent.py:joke_graph",
    "email_agent": "./my_agent/email_agent.py:email_graph",
    "pirate_agent": "./my_agent/pirate_agent.py:pirate_graph"
  },
  "env": ".env"
}

================
File: agent/my_agent/demo.py
================
"""
This is a demo of the CopilotKit SDK.
"""

import os
from dotenv import load_dotenv 
load_dotenv()

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, Action, LangGraphAgent
from my_agent.joke_agent import joke_graph
from my_agent.email_agent import email_graph
from my_agent.pirate_agent import pirate_graph

def greet_user(name):
    """Greet the user."""
    print(f"Hello, {name}!")
    return "The user has been greeted. YOU MUST tell them to check the console."

app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    actions=[
        Action(
            name="greet_user",
            description="Greet the user.",
            handler=greet_user,
            parameters=[
                {
                    "name": "name",
                    "description": "The name of the user to greet.",
                    "type": "string",
                }
            ]
        ),
    ],
    agents=[
        LangGraphAgent(
            name="joke_agent",
            description="Make a joke.",
            graph=joke_graph,
        ),
        LangGraphAgent(
            name="email_agent",
            description="Write an email.",
            graph=email_graph,
        ),
        LangGraphAgent(
            name="pirate_agent",
            description="Speak like a pirate.",
            graph=pirate_graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

# add new route for health check
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "my_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )

================
File: agent/my_agent/email_agent.py
================
"""Email Agent"""

from typing import Any, cast
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, ToolMessage
from copilotkit.langgraph import copilotkit_customize_config, copilotkit_exit
from pydantic import BaseModel, Field
from my_agent.model import get_model


class EmailAgentState(MessagesState):
    """Email Agent State"""
    email: str
    model: str


class write_email(BaseModel): # pylint: disable=invalid-name
    """
    Write an email.
    """
    the_email: str = Field(..., description="The email")


async def email_node(state: EmailAgentState, config: RunnableConfig):
    """
    Make a joke.
    """

    config = copilotkit_customize_config(
        config,
        emit_messages=True,
        emit_intermediate_state=[
            {
                "state_key": "email",
                "tool": "write_email",
                "tool_argument": "the_email"
            },
        ]
    )

    system_message = "You write emails."

    email_model = get_model(state).bind_tools(
        [write_email],
        tool_choice="write_email"
    )

    response = await email_model.ainvoke([
        SystemMessage(
            content=system_message
        ),
        *state["messages"]
    ], config)

    tool_calls = getattr(response, "tool_calls")

    email = tool_calls[0]["args"]["the_email"]

    await copilotkit_exit(config)

    return {
        "messages": [
            response,
            ToolMessage(
                name=tool_calls[0]["name"],
                content=email,
                tool_call_id=tool_calls[0]["id"]
            )
        ],
        "email": email,
    }

workflow = StateGraph(EmailAgentState)
workflow.add_node("email_node", cast(Any, email_node))
workflow.set_entry_point("email_node")

workflow.add_edge("email_node", END)
memory = MemorySaver()
email_graph = workflow.compile(checkpointer=memory)

================
File: agent/my_agent/joke_agent.py
================
"""Test Joker Agent"""

from typing import Any, cast

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, ToolMessage


from copilotkit.langgraph import copilotkit_customize_config, copilotkit_exit
from pydantic import BaseModel, Field
from my_agent.model import get_model

class JokeAgentState(MessagesState):
    """Joke Agent State"""
    joke: str
    model: str

class make_joke(BaseModel): # pylint: disable=invalid-name
    """
    Make a funny joke.
    """
    the_joke: str = Field(..., description="The joke")


async def joke_node(state: JokeAgentState, config: RunnableConfig):
    """
    Make a joke.
    """

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {
                "state_key": "joke",
                "tool": "make_joke",
                "tool_argument": "the_joke"
            },
        ]
    )

    system_message = "You make funny jokes."

    joke_model = get_model(state).bind_tools(
        [make_joke],
        tool_choice="make_joke"
    )

    response = await joke_model.ainvoke([
        SystemMessage(
            content=system_message
        ),
        *state["messages"]
    ], config)

    tool_calls = getattr(response, "tool_calls")

    joke = tool_calls[0]["args"]["the_joke"]

    await copilotkit_exit(config)

    return {
        "messages": [
            response,
            ToolMessage(
                name=tool_calls[0]["name"],
                content=joke,
                tool_call_id=tool_calls[0]["id"]
            )
        ],
        "joke": joke,
    }

workflow = StateGraph(JokeAgentState)
workflow.add_node("joke_node", cast(Any, joke_node))
workflow.set_entry_point("joke_node")

workflow.add_edge("joke_node", END)
memory = MemorySaver()
joke_graph = workflow.compile(checkpointer=memory)

================
File: agent/my_agent/model.py
================
"""
This module provides a function to get a model based on the configuration.
"""
from typing import Any
import os

def get_model(state: Any) -> Any:
    """
    Get a model based on the environment variable.
    """

    model = state.get("model")

    print(f"Using model: {model}")

    if model == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model="gpt-4o")
    if model == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620", timeout=None, stop=None)
    if model == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

    raise ValueError("Invalid model specified")

================
File: agent/my_agent/pirate_agent.py
================
"""Test Pirate Agent"""

from typing import Any, cast
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from copilotkit.langgraph import copilotkit_exit
from my_agent.model import get_model


class PirateAgentState(MessagesState):
    """Pirate Agent State"""
    model: str

async def pirate_node(state: PirateAgentState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Speaks like a pirate
    """

    system_message = "You speak like a pirate. Your name is Captain Copilot. " + \
        "If the user wants to stop talking, you will say (literally) " + \
        "'Arrr, I'll be here if you need me!'"

    pirate_model = get_model(state)

    response = await pirate_model.ainvoke([
        SystemMessage(
            content=system_message
        ),
        *state["messages"],        
    ], config)

    if response.content == "Arrr, I'll be here if you need me!":
        await copilotkit_exit(config)

    return {
        "messages": response,
    }

workflow = StateGraph(PirateAgentState)
workflow.add_node("pirate_node", cast(Any, pirate_node))
workflow.set_entry_point("pirate_node")
workflow.add_edge("pirate_node", END)
memory = MemorySaver()
pirate_graph = workflow.compile(checkpointer=memory)

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "my_agent"
version = "0.1.0"
description = "Starter"
authors = ["Ariel Weinberger <weinberger.ariel@gmail.com>"]
license = "MIT"

[project]
name = "my_agent"
version = "0.0.1"
dependencies = [
    "langgraph",
    "langchain-openai",
    "langchain-anthropic",
    "langchain",
    "openai",
    "langchain-community",
    "copilotkit==0.1.34",
    "uvicorn",
    "python-dotenv",
    "langchain-google-genai",
    "langchain-core",
    "langgraph-cli[inmem]==0.1.64"
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "^0.2.1"
langchain-anthropic = "^0.2.1"
langchain = "^0.3.1"
openai = "^1.51.0"
langchain-community = "^0.3.1"
copilotkit = "0.1.39"
uvicorn = "^0.31.0"
python-dotenv = "^1.0.1"
langchain-google-genai = "^2.0.4"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}

[tool.poetry.scripts]
demo = "my_agent.demo:main"

================
File: README.md
================
# CoAgents Agent Q&A Example

This example demonstrates using routing between multiple agents.

**These instructions assume you are in the `coagents-routing` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/my_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": "next/core-web-vitals"
}

================
File: ui/.gitignore
================
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.js
.yarn/install-state.gz

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local
.env
# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "my-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-textarea": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@copilotkit/shared": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^11.3.31",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/postcss.config.mjs
================
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;

================
File: ui/src/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  langGraphPlatformEndpoint,
  copilotKitEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const serviceAdapter = new OpenAIAdapter({ openai } as any);
const langsmithApiKey = process.env.LANGSMITH_API_KEY as string;

export const POST = async (req: NextRequest) => {
  const searchParams = req.nextUrl.searchParams;
  const deploymentUrl = searchParams.get("lgcDeploymentUrl");

  const remoteEndpoint = deploymentUrl
    ? langGraphPlatformEndpoint({
        deploymentUrl,
        langsmithApiKey,
        agents: [
          {
            name: "joke_agent",
            description: "Make a joke.",
          },
          {
            name: "email_agent",
            description: "Write an email.",
          },
          {
            name: "pirate_agent",
            description: "Speak like a pirate.",
          },
        ],
      })
    : copilotKitEndpoint({
        url:
          process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
      });

  const runtime = new CopilotRuntime({
    remoteEndpoints: [remoteEndpoint],
    actions: [
      {
        name: "greetUser",
        description: "Greet the user",
        parameters: [
          {
            name: "name",
            type: "string",
            description: "The name of the user to greet",
          },
        ],
        handler: async ({ name }) => {
          console.log("greetUser", name);
          return "I greeted the user. YOU MUST tell the user to check the console to see the message.";
        },
      },
      {
        name: "sayGoodbye",
        description: "Say goodbye to the user",
        parameters: [
          {
            name: "name",
            type: "string",
            description: "The name of the user to say goodbye to",
          },
        ],
        handler: async ({ name }) => {
          console.log("goodbye", name);
          return "I said goodbye to the user. YOU MUST tell the user to check the console to see the message.";
        },
      },
    ],
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/src/app/components/ModelSelector.tsx
================
"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { useModelSelectorContext } from "../lib/model-selector-provider";

export function ModelSelector() {
  const { model, setModel } = useModelSelectorContext();

  return (
    <div className="fixed bottom-0 left-0 p-4 z-50">
      <Select value={model} onValueChange={(v) => setModel(v)}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Theme" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="openai">OpenAI</SelectItem>
          <SelectItem value="anthropic">Anthropic</SelectItem>
          <SelectItem value="google_genai">Google Generative AI</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

================
File: ui/src/app/components/ui/select.tsx
================
"use client";

import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown, ChevronUp } from "lucide-react";

import { cn } from "../../lib/utils";

const Select = SelectPrimitive.Root;

const SelectGroup = SelectPrimitive.Group;

const SelectValue = SelectPrimitive.Value;

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
));
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName;

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronUp className="h-4 w-4" />
  </SelectPrimitive.ScrollUpButton>
));
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName;

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronDown className="h-4 w-4" />
  </SelectPrimitive.ScrollDownButton>
));
SelectScrollDownButton.displayName =
  SelectPrimitive.ScrollDownButton.displayName;

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
));
SelectContent.displayName = SelectPrimitive.Content.displayName;

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("py-1.5 pl-8 pr-2 text-sm font-semibold", className)}
    {...props}
  />
));
SelectLabel.displayName = SelectPrimitive.Label.displayName;

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>

    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
));
SelectItem.displayName = SelectPrimitive.Item.displayName;

const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
));
SelectSeparator.displayName = SelectPrimitive.Separator.displayName;

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
};

================
File: ui/src/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

================
File: ui/src/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/src/app/lib/model-selector-provider.tsx
================
"use client";

import React from "react";
import { createContext, useContext, useState, ReactNode } from "react";

type ModelSelectorContextType = {
  model: string;
  setModel: (model: string) => void;
  hidden: boolean;
  setHidden: (hidden: boolean) => void;
  lgcDeploymentUrl?: string | null;
};

const ModelSelectorContext = createContext<
  ModelSelectorContextType | undefined
>(undefined);

export const ModelSelectorProvider = ({
  children,
}: {
  children: ReactNode;
}) => {
  const model =
    globalThis.window === undefined
      ? "openai"
      : new URL(window.location.href).searchParams.get("coAgentsModel") ??
        "openai";

  const lgcDeploymentUrl = globalThis.window === undefined
      ? null
      : new URL(window.location.href).searchParams.get("lgcDeploymentUrl")

  const [hidden, setHidden] = useState<boolean>(false);

  const setModel = (model: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("coAgentsModel", model);
    window.location.href = url.toString();
  };

  return (
    <ModelSelectorContext.Provider
      value={{
        model,
        hidden,
        lgcDeploymentUrl,
        setModel,
        setHidden,
      }}
    >
      {children}
    </ModelSelectorContext.Provider>
  );
};

export const useModelSelectorContext = () => {
  const context = useContext(ModelSelectorContext);
  if (context === undefined) {
    throw new Error(
      "useModelSelectorContext must be used within a ModelSelectorProvider"
    );
  }
  return context;
};

================
File: ui/src/app/lib/utils.ts
================
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

================
File: ui/src/app/page.tsx
================
"use client";
import {
  CopilotKit,
  useCoAgent,
  useCoAgentStateRender,
  useCopilotChat,
} from "@copilotkit/react-core";
import {
  CopilotSidebar,
  useCopilotChatSuggestions,
} from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import {
  ModelSelectorProvider,
  useModelSelectorContext,
} from "./lib/model-selector-provider";
import { ModelSelector } from "./components/ModelSelector";
import { MessageRole, TextMessage } from "@copilotkit/runtime-client-gql";

export default function ModelSelectorWrapper() {
  return (
    <ModelSelectorProvider>
      <Home />
      <ModelSelector />
    </ModelSelectorProvider>
  );
}

function Home() {
  const { lgcDeploymentUrl } = useModelSelectorContext();

  return (
    <CopilotKit
      runtimeUrl={`/api/copilotkit?lgcDeploymentUrl=${lgcDeploymentUrl ?? ""}`}
    >
      <div className="min-h-screen bg-gray-100 p-4">
        <div className="max-w-2xl mx-auto bg-white shadow-md rounded-lg p-6 mt-4 flex justify-center">
          <ResetButton />
        </div>

        <div className="max-w-2xl mx-auto bg-white shadow-md rounded-lg p-6 mt-4">
          <Joke />
        </div>
        <div className="max-w-2xl mx-auto bg-white shadow-md rounded-lg p-6 mt-4">
          <Email />
        </div>
        <div className="max-w-2xl mx-auto bg-white shadow-md rounded-lg p-6 mt-4">
          <PirateMode />
        </div>
        <CopilotSidebar
          defaultOpen={true}
          clickOutsideToClose={false}
          className="mt-4"
        />
      </div>
    </CopilotKit>
  );
}

function ResetButton() {
  const { reset } = useCopilotChat();
  return (
    <button
      className="px-6 py-3 border-2 border-gray-300 bg-white text-gray-800 rounded-lg shadow-md hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 transition duration-300 ease-in-out"
      onClick={() => reset()}
    >
      Reset Everything
    </button>
  );
}

function usePirateAgent() {
  const { model } = useModelSelectorContext();
  return useCoAgent({
    name: "pirate_agent",
    initialState: {
      model,
    },
  });
}

function PirateMode() {
  useCopilotChatSuggestions({
    instructions: "Suggest to talk to a pirate about piratey things",
    maxSuggestions: 1,
  });
  const { running } = usePirateAgent();

  if (running) {
    return (
      <div
        data-test-id="container-pirate-mode-on"
        style={{ fontSize: "0.875rem", textAlign: "center" }}
      >
        Pirate mode is on
      </div>
    );
  } else {
    return (
      <div
        data-test-id="container-pirate-mode-off"
        style={{ fontSize: "0.875rem", textAlign: "center" }}
      >
        Pirate mode is off
      </div>
    );
  }
}

function RunPirateMode() {
  const { run } = usePirateAgent();
  return (
    <button
      onClick={() =>
        run(
          () =>
            new TextMessage({
              content: "Run pirate mode",
              role: MessageRole.User,
            })
        )
      }
      className="bg-white text-black border border-gray-300 rounded px-4 py-2 shadow hover:bg-gray-100"
    >
      Run Pirate Mode
    </button>
  );
}

function Joke() {
  const { model } = useModelSelectorContext();
  useCopilotChatSuggestions({
    instructions: "Suggest to make a joke about a specific subject",
    maxSuggestions: 1,
  });
  const { state } = useCoAgent({
    name: "joke_agent",
    initialState: {
      model,
      joke: "",
    },
  });

  useCoAgentStateRender({
    name: "joke_agent",
    render: ({ state, nodeName }) => {
      return <div>Generating joke: {state.joke}</div>;
    },
  });

  if (!state.joke) {
    return (
      <div
        data-test-id="container-joke-empty"
        style={{ fontSize: "0.875rem", textAlign: "center" }}
      >
        No joke generated yet
      </div>
    );
  } else {
    return <div data-test-id="container-joke-nonempty">Joke: {state.joke}</div>;
  }
}

function Email() {
  const { model } = useModelSelectorContext();
  useCopilotChatSuggestions({
    instructions: "Suggest to write an email to a famous person",
    maxSuggestions: 1,
  });
  const { state } = useCoAgent({
    name: "email_agent",
    initialState: {
      model,
      email: "",
    },
  });

  useCoAgentStateRender({
    name: "email_agent",
    render: ({ state, nodeName }) => {
      return <div>Generating email: {state.email}</div>;
    },
  });

  if (!state.email) {
    return (
      <div
        data-test-id="container-email-empty"
        style={{ fontSize: "0.875rem", textAlign: "center" }}
      >
        No email generated yet
      </div>
    );
  } else {
    return (
      <div data-test-id="container-email-nonempty">Email: {state.email}</div>
    );
  }
}

================================================================
End of Codebase
================================================================


# **CoAgents shared state Example**:

This file is a merged representation of the entire codebase, combined into a single document.

================================================================
Directory Structure
================================================================
agent/langgraph.json
agent/pyproject.toml
agent/translate_agent/agent.py
agent/translate_agent/demo.py
README.md
ui/.eslintrc.json
ui/.gitignore
ui/app/api/copilotkit/route.ts
ui/app/globals.css
ui/app/layout.tsx
ui/app/page.tsx
ui/app/Translator.tsx
ui/components.json
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "translate_agent": "./translate_agent/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "translate_agent"
version = "0.1.0"
description = "Starter"
authors = ["Ariel Weinberger <weinberger.ariel@gmail.com>"]
license = "MIT"

[project]
name = "translate_agent"
version = "0.0.1"
dependencies = [
    "langchain-openai",
    "langchain-anthropic",
    "langchain",
    "openai",
    "langchain-community",
    "copilotkit",
    "uvicorn",
    "python-dotenv",
    "langchain-core",
    "langgraph-cli"
]


[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "^0.2.1"
langchain-anthropic = "^0.2.1"
langchain = "^0.3.1"
openai = "^1.51.0"
langchain-community = "^0.3.1"
copilotkit = "0.1.39"
uvicorn = "^0.31.0"
python-dotenv = "^1.0.1"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}


[tool.poetry.scripts]
demo = "translate_agent.demo:main"

================
File: agent/translate_agent/agent.py
================
"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import

from typing import cast, TypedDict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from copilotkit.langgraph import copilotkit_customize_config

class Translations(TypedDict):
    """Contains the translations in four different languages."""
    translation_es: str
    translation_fr: str
    translation_de: str

class AgentState(MessagesState):
    """Contains the state of the agent."""
    translations: Translations
    input: str

async def translate_node(state: AgentState, config: RunnableConfig):
    """Chatbot that translates text"""

    config = copilotkit_customize_config(
        config,
        # config emits messages by default, so this is not needed:
        ## emit_messages=True,
        emit_intermediate_state=[
            {
                "state_key": "translations",
                "tool": "translate"
            }
        ]
    )

    model = ChatOpenAI(model="gpt-4o").bind_tools(
        [Translations],
        parallel_tool_calls=False,
        tool_choice=(
            None if state["messages"] and
            isinstance(state["messages"][-1], HumanMessage)
            else "Translations"
        )
    )

    response = await model.ainvoke([
        SystemMessage(
            content=f"""
            You are a helpful assistant that translates text to different languages 
            (Spanish, French and German).
            Don't ask for confirmation before translating.
            {
                'The user is currently working on translating this text: "' + 
                state["input"] + '"' if state.get("input") else ""
            }
            """
        ),
        *state["messages"],
    ], config)

    if hasattr(response, "tool_calls") and len(getattr(response, "tool_calls")) > 0:
        ai_message = cast(AIMessage, response)
        return {
            "messages": [
                response,
                ToolMessage(
                    content="Translated!",
                    tool_call_id=ai_message.tool_calls[0]["id"]
                )
            ],
            "translations": cast(AIMessage, response).tool_calls[0]["args"],
        }

    return {
        "messages": [           
            response,
        ],
    }

workflow = StateGraph(AgentState)
workflow.add_node("translate_node", cast(Any, translate_node))
workflow.set_entry_point("translate_node")
workflow.add_edge("translate_node", END)
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

================
File: agent/translate_agent/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from translate_agent.agent import graph


app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="translate_agent",
            description="Translate agent that translates text.",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "translate_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )

================
File: README.md
================
# CoAgents Shared State Example

This example demonstrates how to share state between the agent and the UI.

**These instructions assume you are in the `coagents-shared-state/` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/translate_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": "next/core-web-vitals"
}

================
File: ui/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const serviceAdapter = new OpenAIAdapter({ openai });

const runtime = new CopilotRuntime({
  remoteEndpoints: [
    {
      url: process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
    },
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

================
File: ui/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CoAgents Starter",
  description: "CoAgents Starter",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/app/page.tsx
================
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { Translator } from "./Translator";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-between">
      <CopilotKit runtimeUrl="/api/copilotkit" agent="translate_agent">
        <Translator />
      </CopilotKit>
    </main>
  );
}

================
File: ui/app/Translator.tsx
================
"use client";

import { useCoAgent, useCopilotChat } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import { MessageRole, TextMessage } from "@copilotkit/runtime-client-gql";

interface TranslateAgentState {
  input: string;
  translations?: {
    translation_es: string;
    translation_fr: string;
    translation_de: string;
  };
}

export function Translator() {
  const {
    state: translateAgentState,
    setState: setTranslateAgentState,
    run: runTranslateAgent,
  } = useCoAgent<TranslateAgentState>({
    name: "translate_agent",
    initialState: { input: "Hello World" },
  });

  const { isLoading } = useCopilotChat();

  console.log("state", translateAgentState);

  const handleTranslate = () => {
    runTranslateAgent(() => new TextMessage({ role: MessageRole.User, content: "Translate to all languages" }));
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <div>
        <input
          type="text"
          placeholder="Text to translate..."
          value={translateAgentState.input}
          onChange={(e) =>
            setTranslateAgentState({
              ...translateAgentState,
              input: e.target.value,
            })
          }
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleTranslate();
            }
          }}
          className="w-full p-2 border border-gray-300 rounded"
        />
        <button
          disabled={!translateAgentState.input || isLoading}
          onClick={handleTranslate}
          className="mt-2 w-full p-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
        >
          {isLoading ? "Translating..." : "Translate"}
        </button>
      </div>

      {translateAgentState.translations && (
        <div className="mt-8">
          <div>Spanish: {translateAgentState.translations.translation_es}</div>
          <div>French: {translateAgentState.translations.translation_fr}</div>
          <div>German: {translateAgentState.translations.translation_de}</div>
        </div>
      )}

      <CopilotPopup defaultOpen={true} />
    </div>
  );
}

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "app/globals.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils"
  }
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ai-researcher-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^11.3.31",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/postcss.config.mjs
================
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";
const plugin = require('tailwindcss/plugin')
const {
  default: flattenColorPalette,
} = require("tailwindcss/lib/util/flattenColorPalette");

const config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: "true",
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        meteor: {
          "0%": {
            transform: "rotate(215deg) translateX(0)",
            opacity: "1",
          },
          "70%": {
            opacity: "1",
          },
          "100%": {
            transform: "rotate(215deg) translateX(-500px)",
            opacity: "0",
          },
        },
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "meteor-effect": "meteor 5s linear infinite",
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate"), addVariablesForColors, plugin(capitalizeFirstLetter)],
} satisfies Config;

function capitalizeFirstLetter({ addUtilities }: any) {
  const newUtilities = {
    '.capitalize-first:first-letter': {
      textTransform: 'uppercase',
    },
  }
  addUtilities(newUtilities, ['responsive', 'hover'])
}

function addVariablesForColors({ addBase, theme }: any) {
  let allColors = flattenColorPalette(theme("colors"));
  let newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

export default config;

================
File: ui/tsconfig.json
================
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts", "tailwind.config.js"],
  "exclude": ["node_modules", "tailwind.config.ts"]
}

================================================================
End of Codebase
================================================================

# **CoAgents Wait User Input Example** :

This file is a merged representation of the entire codebase, combined into a single document.

================================================================
File Summary
================================================================

================================================================
Directory Structure
================================================================
agent/langgraph.json
agent/pyproject.toml
agent/weather_agent/agent.py
agent/weather_agent/demo.py
README.md
ui/.eslintrc.json
ui/.gitignore
ui/app/api/copilotkit/route.ts
ui/app/globals.css
ui/app/layout.tsx
ui/app/page.tsx
ui/app/WaitForUserInput.tsx
ui/components.json
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "weather_agent": "./weather_agent/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "weather_agent"
version = "0.1.0"
description = "Starter"
authors = ["Ariel Weinberger <weinberger.ariel@gmail.com>"]
license = "MIT"

[project]
name = "weather_agent"
version = "0.0.1"
dependencies = [
    "langchain-openai",
    "langchain-anthropic",
    "langchain",
    "openai",
    "langchain-community",
    "copilotkit",
    "uvicorn",
    "python-dotenv",
    "langchain-core",
    "langgraph-cli"
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "^0.2.1"
langchain-anthropic = "^0.2.1"
langchain = "^0.3.1"
openai = "^1.51.0"
langchain-community = "^0.3.1"
copilotkit = "0.1.39"
uvicorn = "^0.31.0"
python-dotenv = "^1.0.1"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}

[tool.poetry.scripts]
demo = "weather_agent.demo:main"

================
File: agent/weather_agent/agent.py
================
# Set up the state
from langgraph.graph import MessagesState, START

# Set up the tool
# We will have one real tool - a search tool
# We'll also have one "fake" tool - a "ask_human" tool
# Here we define any ACTUAL tools
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from copilotkit.langgraph import copilotkit_customize_config


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return f"I looked up: {query}. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)

# Set up the model
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
model = ChatOpenAI(model="gpt-4o")

from pydantic import BaseModel


# We are going "bind" all tools to the model
# We have the ACTUAL tools from above, but we also need a mock tool to ask a human
# Since `bind_tools` takes in tools but also just tool definitions,
# We can define a tool definition for `ask_human`
class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str


model = model.bind_tools(tools + [AskHuman])

# Define nodes and conditional edges


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # If tool call is asking Human, we return that node
    # You could also add logic here to let some system know that there's something that requires Human input
    # For example, send a slack message, etc
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state, config):

    config = copilotkit_customize_config(
        config,
        emit_tool_calls="AskHuman",
    )
    messages = state["messages"]
    response = model.invoke(messages, config=config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We define a fake node to ask the human
def ask_human(state):
    pass


# Build the graph

from langgraph.graph import END, StateGraph

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the three nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # We may ask the human
        "ask_human": "ask_human",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# After we get back the human response, we go back to the agent
workflow.add_edge("ask_human", "agent")

# Set up memory
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
# We add a breakpoint BEFORE the `ask_human` node so it never executes
graph = workflow.compile(checkpointer=memory, interrupt_after=["ask_human"])

================
File: agent/weather_agent/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from weather_agent.agent import graph


app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="weather_agent",
            description="This agent deals with everything weather related",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "weather_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )

================
File: README.md
================
# CoAgents Agent Q&A Example

This example is taken straight from the LangGraph documentation.

**These instructions assume you are in the `coagents-qa/` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/weather_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const serviceAdapter = new OpenAIAdapter({ openai });

const runtime = new CopilotRuntime({
  remoteEndpoints: [
    {
      url: process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
    },
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

================
File: ui/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CoAgents Starter",
  description: "CoAgents Starter",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/app/page.tsx
================
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { WaitForUserInput } from "./WaitForUserInput";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-between">
      <CopilotKit runtimeUrl="/api/copilotkit" agent="weather_agent">
        <WaitForUserInput />
      </CopilotKit>
    </main>
  );
}

================
File: ui/app/WaitForUserInput.tsx
================
"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";

export function WaitForUserInput() {
  useCopilotAction({
    name: "AskHuman",
    available: "remote",
    parameters: [
      {
        name: "question",
      },
    ],
    handler: async ({ question }) => {
      return window.prompt(question);
    },
  });

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <div className="text-2xl">LangGraph Wait For User Input Example</div>
      <div className="text-xs">
        (https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/#agent)
      </div>
      <div>
        Use the search tool to ask the user where they are, then look up the
        weather there
      </div>

      <CopilotPopup defaultOpen={true} clickOutsideToClose={false} />
    </div>
  );
}

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "app/globals.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils"
  }
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ai-researcher-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@copilotkit/shared": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^11.3.31",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";
const plugin = require('tailwindcss/plugin')
const {
  default: flattenColorPalette,
} = require("tailwindcss/lib/util/flattenColorPalette");

const config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: "true",
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        meteor: {
          "0%": {
            transform: "rotate(215deg) translateX(0)",
            opacity: "1",
          },
          "70%": {
            opacity: "1",
          },
          "100%": {
            transform: "rotate(215deg) translateX(-500px)",
            opacity: "0",
          },
        },
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "meteor-effect": "meteor 5s linear infinite",
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate"), addVariablesForColors, plugin(capitalizeFirstLetter)],
} satisfies Config;

function capitalizeFirstLetter({ addUtilities }: any) {
  const newUtilities = {
    '.capitalize-first:first-letter': {
      textTransform: 'uppercase',
    },
  }
  addUtilities(newUtilities, ['responsive', 'hover'])
}

function addVariablesForColors({ addBase, theme }: any) {
  let allColors = flattenColorPalette(theme("colors"));
  let newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

export default config;

================
File: ui/tsconfig.json
================
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts", "tailwind.config.js"],
  "exclude": ["node_modules", "tailwind.config.ts"]
}



================================================================
End of Codebase
================================================================


# CoAgents-qa Example: 

This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

P
================================================================
Directory Structure
================================================================
agent/.gitignore
agent/.vscode/cspell.json
agent/.vscode/settings.json
agent/email_agent/agent.py
agent/email_agent/demo.py
agent/langgraph.json
agent/pyproject.toml
README.md
ui/.eslintrc.json
ui/.gitignore
ui/app/api/copilotkit/route.ts
ui/app/globals.css
ui/app/layout.tsx
ui/app/Mailer.tsx
ui/app/page.tsx
ui/components.json
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent/.gitignore
================
venv/
__pycache__/
*.pyc
.env
.vercel

================
File: agent/.vscode/cspell.json
================
{
  "version": "0.2",
  "language": "en",
  "words": [
    "langgraph",
    "langchain",
    "perplexity",
    "openai",
    "ainvoke",
    "pydantic",
    "tavily",
    "copilotkit",
    "fastapi",
    "uvicorn",
    "checkpointer",
    "dotenv"
  ]
}

================
File: agent/.vscode/settings.json
================
{
  "python.analysis.typeCheckingMode": "basic"
}

================
File: agent/email_agent/agent.py
================
"""Test Human in the Loop Agent"""

import os
from typing import Any, cast
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from copilotkit.langgraph import (
    copilotkit_exit,
    copilotkit_emit_message
)
from pydantic import BaseModel, Field


def get_model():
    """
    Get a model based on the environment variable.
    """
    model = os.getenv("MODEL", "openai")

    if model == "openai":
        return ChatOpenAI(temperature=0, model="gpt-4o")
    if model == "anthropic":
        return ChatAnthropic(
            temperature=0,
            model_name="claude-3-5-sonnet-20240620",
            timeout=None,
            stop=None
        )

    raise ValueError("Invalid model specified")


class EmailAgentState(MessagesState):
    """Email Agent State"""
    email: str

class EmailTool(BaseModel):
    """
    Write an email.
    """
    email_draft: str = Field(description="The draft of the email to be written.")


async def draft_email_node(state: EmailAgentState, config: RunnableConfig):
    """
    Write an email.
    """

    instructions = "You write emails."

    email_model = get_model().bind_tools(
        [EmailTool],
        tool_choice="EmailTool"
    )

    response = await email_model.ainvoke([
        *state["messages"],
        HumanMessage(
            content=instructions
        )
    ], config)

    tool_calls = cast(Any, response).tool_calls

    # the email content is the argument passed to the email tool
    email = tool_calls[0]["args"]["email_draft"]

    return {
        "messages": response,
        "email": email,
    }

async def send_email_node(state: EmailAgentState, config: RunnableConfig):
    """
    Send an email.
    """

    await copilotkit_exit(config)

    # get the last message and cast to ToolMessage
    last_message = cast(ToolMessage, state["messages"][-1])
    message_to_add = ""
    if last_message.content == "CANCEL":
        message_to_add = "❌ Cancelled sending email."
    else:
        message_to_add = "✅ Sent email."

    await copilotkit_emit_message(config, message_to_add)
    return {
        "messages": state["messages"] + [AIMessage(content=message_to_add)],
    }


workflow = StateGraph(EmailAgentState)
workflow.add_node("draft_email_node", draft_email_node)
workflow.add_node("send_email_node", send_email_node)
workflow.set_entry_point("draft_email_node")

workflow.add_edge("draft_email_node", "send_email_node")
workflow.add_edge("send_email_node", END)
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_after=["draft_email_node"])

================
File: agent/email_agent/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from email_agent.agent import graph


app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="email_agent",
            description="This agent sends emails",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "email_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "email_agent": "./email_agent/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "email_agent"
version = "0.1.0"
description = "Starter"
authors = ["Ariel Weinberger <weinberger.ariel@gmail.com>"]
license = "MIT"

[project]
name = "email_agent"
version = "0.0.1"
dependencies = [
  "langgraph",
  "langchain_core",
  "langchain_openai",
  "langchain",
  "openai",
  "langchain-community",
  "copilotkit",
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "^0.2.1"
langchain-anthropic = "^0.2.1"
langchain = "^0.3.1"
openai = "^1.51.0"
langchain-community = "^0.3.1"
copilotkit = "0.1.39"
uvicorn = "^0.31.0"
python-dotenv = "^1.0.1"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}

[tool.poetry.scripts]
demo = "email_agent.demo:main"

================
File: README.md
================
# CoAgents Agent Q&A Example

This example demonstrates sending a question to the user that gets displayed in the chat window.

**These instructions assume you are in the `coagents-qa/` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/email_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": "next/core-web-vitals"
}

================
File: ui/.gitignore
================
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.js
.yarn/install-state.gz

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local

.env

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

================
File: ui/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const serviceAdapter = new OpenAIAdapter({ openai } as any);

const runtime = new CopilotRuntime({
  remoteEndpoints: [
    {
      url: process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
    },
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

================
File: ui/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CoAgents Starter",
  description: "CoAgents Starter",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/app/Mailer.tsx
================
"use client";

import React from "react";
import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";

export function Mailer() {
  useCopilotAction({
    name: "EmailTool",
    available: "remote",
    parameters: [
      {
        name: "email_draft",
        type: "string",
        description: "The email content",
        required: true,
      },
    ],
    renderAndWait: ({ args, status, handler }) => (
      <EmailConfirmation
        emailContent={args.email_draft || ""}
        isExecuting={status === "executing"}
        onCancel={() => handler?.("CANCEL")} // the handler is undefined while status is "executing"
        onSend={() => handler?.("SEND")} // the handler is undefined while status is "executing"
      />
    ),
  });

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <div className="text-2xl">Email Q&A example</div>
      <div>e.g. write an email to the CEO of OpenAI asking for a meeting</div>

      <CopilotPopup defaultOpen={true} clickOutsideToClose={false} />
    </div>
  );
}

interface EmailConfirmationProps {
  emailContent: string;
  isExecuting: boolean;
  onCancel: () => void;
  onSend: () => void;
}

const EmailConfirmation: React.FC<EmailConfirmationProps> = ({
  emailContent,
  isExecuting,
  onCancel,
  onSend,
}) => {
  return (
    <div className="p-4 bg-gray-100 rounded-lg">
      <div className="font-bold text-lg mb-2">Send this email?</div>
      <div className="text-gray-700">{emailContent}</div>
      {isExecuting && (
        <div className="mt-4 flex justify-end space-x-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-slate-400 text-white rounded"
          >
            Cancel
          </button>
          <button
            onClick={onSend}
            className="px-4 py-2 bg-blue-500 text-white rounded"
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
};

================
File: ui/app/page.tsx
================
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { Mailer } from "./Mailer";
import "@copilotkit/react-ui/styles.css";

export default function Home() {
  return (
    <main className="flex flex-col items-center justify-between">
      <CopilotKit runtimeUrl="/api/copilotkit" agent="email_agent">
        <Mailer />
      </CopilotKit>
    </main>
  );
}

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "app/globals.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils"
  }
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ai-researcher-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-textarea": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@copilotkit/shared": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^11.3.31",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/postcss.config.mjs
================
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;
================
File: ui/README.md
================
This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";
const plugin = require('tailwindcss/plugin')
const {
  default: flattenColorPalette,
} = require("tailwindcss/lib/util/flattenColorPalette");

const config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: "true",
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        meteor: {
          "0%": {
            transform: "rotate(215deg) translateX(0)",
            opacity: "1",
          },
          "70%": {
            opacity: "1",
          },
          "100%": {
            transform: "rotate(215deg) translateX(-500px)",
            opacity: "0",
          },
        },
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "meteor-effect": "meteor 5s linear infinite",
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate"), addVariablesForColors, plugin(capitalizeFirstLetter)],
} satisfies Config;

function capitalizeFirstLetter({ addUtilities }: any) {
  const newUtilities = {
    '.capitalize-first:first-letter': {
      textTransform: 'uppercase',
    },
  }
  addUtilities(newUtilities, ['responsive', 'hover'])
}

function addVariablesForColors({ addBase, theme }: any) {
  let allColors = flattenColorPalette(theme("colors"));
  let newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

export default config;

================
File: ui/tsconfig.json
================
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts", "tailwind.config.js"],
  "exclude": ["node_modules", "tailwind.config.ts"]
}



================================================================
End of Codebase
================================================================


# CoAgents-ai-researcher example:

This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
agent/.gitignore
agent/.vscode/cspell.json
agent/ai_researcher/agent.py
agent/ai_researcher/demo.py
agent/ai_researcher/extract.py
agent/ai_researcher/model.py
agent/ai_researcher/search.py
agent/ai_researcher/state.py
agent/ai_researcher/steps.py
agent/ai_researcher/summarize.py
agent/langgraph.json
agent/pyproject.toml
README.md
ui/.eslintrc.json
ui/.gitignore
ui/app/api/copilotkit-lgc/route.ts
ui/app/api/copilotkit/route.ts
ui/app/globals.css
ui/app/layout.tsx
ui/app/page.tsx
ui/components.json
ui/components/AnswerMarkdown.tsx
ui/components/HomeView.tsx
ui/components/ModelSelector.tsx
ui/components/Progress.tsx
ui/components/ResearchWrapper.tsx
ui/components/ResultsView.tsx
ui/components/SkeletonLoader.tsx
ui/components/ui/accordion.tsx
ui/components/ui/background-beams.tsx
ui/components/ui/button.tsx
ui/components/ui/card.tsx
ui/components/ui/input.tsx
ui/components/ui/meteors.tsx
ui/components/ui/select.tsx
ui/components/ui/skeleton.tsx
ui/components/ui/textarea.tsx
ui/lib/model-selector-provider.tsx
ui/lib/research-provider.tsx
ui/lib/types.ts
ui/lib/utils.ts
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent/.gitignore
================
venv/
__pycache__/
*.pyc
.env
.vercel

================
File: agent/.vscode/cspell.json
================
{
  "version": "0.2",
  "language": "en",
  "words": [
    "langgraph",
    "langchain",
    "perplexity",
    "openai",
    "ainvoke",
    "pydantic",
    "tavily",
    "copilotkit",
    "fastapi",
    "uvicorn",
    "checkpointer"
  ]
}

================
File: agent/ai_researcher/agent.py
================
"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ai_researcher.state import AgentState
from ai_researcher.steps import steps_node
from ai_researcher.search import search_node
from ai_researcher.summarize import summarize_node
from ai_researcher.extract import extract_node

def route(state):
    """Route to research nodes."""
    if not state.get("steps", None):
        return END

    current_step = next((step for step in state["steps"] if step["status"] == "pending"), None)

    if not current_step:
        return "summarize_node"

    if current_step["type"] == "search":
        return "search_node"

    raise ValueError(f"Unknown step type: {current_step['type']}")

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("steps_node", steps_node)
workflow.add_node("search_node", search_node)
workflow.add_node("summarize_node", summarize_node)
workflow.add_node("extract_node", extract_node)
# Chatbot
workflow.set_entry_point("steps_node")

workflow.add_conditional_edges(
    "steps_node", 
    route,
    ["summarize_node", "search_node", END]
)

workflow.add_edge("search_node", "extract_node")

workflow.add_conditional_edges(
    "extract_node",
    route,
    ["summarize_node", "search_node"]
)

workflow.add_edge("summarize_node", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

================
File: agent/ai_researcher/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from ai_researcher.agent import graph

app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="ai_researcher",
            description="Search agent.",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

# add new route for health check
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "ai_researcher.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../sdk-python/copilotkit"]
             if os.path.exists("../../../sdk-python/copilotkit")
             else []
             )
        )
    )

================
File: agent/ai_researcher/extract.py
================
"""
The extract node is responsible for extracting information from a tavily search.
"""
import json

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from ai_researcher.state import AgentState
from ai_researcher.model import get_model

async def extract_node(state: AgentState, config: RunnableConfig):
    """
    The extract node is responsible for extracting information from a tavily search.
    """

    current_step = next((step for step in state["steps"] if step["status"] == "pending"), None)

    if current_step is None:
        raise ValueError("No current step")

    if current_step["type"] != "search":
        raise ValueError("Current step is not of type search")

    system_message = f"""
This step was just executed: {json.dumps(current_step)}

This is the result of the search:

Please summarize ONLY the result of the search and include all relevant information from the search and reference links.
DO NOT INCLUDE ANY EXTRA INFORMATION. ALL OF THE INFORMATION YOU ARE LOOKING FOR IS IN THE SEARCH RESULTS.

DO NOT answer the user's query yet. Just summarize the search results.

Use markdown formatting and put the references inline and the links at the end.
Like this:
This is a sentence with a reference to a source [source 1][1] and another reference [source 2][2].
[1]: http://example.com/source1 "Title of Source 1"
[2]: http://example.com/source2 "Title of Source 2"
"""

    response = await get_model(state).ainvoke([
        state["messages"][0],
        HumanMessage(
            content=system_message
        )
    ], config)

    current_step["result"] = response.content
    current_step["search_result"] = None
    current_step["status"] = "complete"
    current_step["updates"] = [*current_step["updates"], "Done."]

    next_step = next((step for step in state["steps"] if step["status"] == "pending"), None)
    if next_step:
        next_step["updates"] = ["Searching the web..."]

    return state

================
File: agent/ai_researcher/model.py
================
"""
This module provides a function to get a model based on the configuration.
"""
import os
from ai_researcher.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel

def get_model(state: AgentState) -> BaseChatModel:
    """
    Get a model based on the environment variable.
    """

    state_model = state.get("model")
    model = os.getenv("MODEL", state_model)

    print(f"Using model: {model}")

    if model == "openai":
        from langchain_openai import ChatOpenAI # pylint: disable=import-outside-toplevel
        return ChatOpenAI(temperature=0, model="gpt-4o-mini")
    if model == "anthropic":
        from langchain_anthropic import ChatAnthropic # pylint: disable=import-outside-toplevel
        return ChatAnthropic(temperature=0, model="claude-3-5-sonnet-20240620")
    if model == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI # pylint: disable=import-outside-toplevel
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

    raise ValueError("Invalid model specified")

================
File: agent/ai_researcher/search.py
================
"""
The search node is responsible for searching the internet for information.
"""
import json
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import TavilySearchResults
from ai_researcher.state import AgentState
from ai_researcher.model import get_model
async def search_node(state: AgentState, config: RunnableConfig):
    """
    The search node is responsible for searching the internet for information.
    """
    tavily_tool = TavilySearchResults(
        max_results=10,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )

    current_step = next((step for step in state["steps"] if step["status"] == "pending"), None)

    if current_step is None:
        raise ValueError("No step to search for")

    if current_step["type"] != "search":
        raise ValueError("Current step is not a search step")

    instructions = f"""
This is a step in a series of steps that are being executed to answer the user's query.
These are all of the steps: {json.dumps(state["steps"])}

You are responsible for carrying out the step: {json.dumps(current_step)}

The current date is {datetime.now().strftime("%Y-%m-%d")}.

This is what you need to search for, please come up with a good search query: {current_step["description"]}
"""
    model = get_model(state).bind_tools(
        [tavily_tool],
        tool_choice=tavily_tool.name
    )

    response = await model.ainvoke([
        HumanMessage(
            content=instructions
        )
    ], config)

    tool_msg = tavily_tool.invoke(response.tool_calls[0])

    current_step["search_result"] = json.loads(tool_msg.content)
    current_step["updates"] = [*current_step["updates"],"Extracting information..."]

    return state

================
File: agent/ai_researcher/state.py
================
"""
This is the state definition for the AI.
It defines the state of the agent and the state of the conversation.
"""

from typing import List, TypedDict, Optional
from langgraph.graph import MessagesState

class Step(TypedDict):
    """
    Represents a step taken in the research process.
    """
    id: str
    description: str
    status: str
    type: str
    description: str
    search_result: Optional[str]
    result: Optional[str]
    updates: Optional[List[str]]

class AgentState(MessagesState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """
    model: str = "openai"
    steps: List[Step]
    answer: Optional[str]

================
File: agent/ai_researcher/steps.py
================
"""
This node is responsible for creating the steps for the research process.
"""

# pylint: disable=line-too-long

from typing import List
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from copilotkit.langgraph import copilotkit_customize_config
from pydantic import BaseModel, Field
from ai_researcher.state import AgentState
from ai_researcher.model import get_model


class SearchStep(BaseModel):
    """Model for a search step"""

    id: str = Field(description="The id of the step. This is used to identify the step in the state. Just make sure it is unique.")
    description: str = Field(description='The description of the step, i.e. "search for information about the latest AI news"')
    status: str = Field(description='The status of the step. Always "pending".', enum=['pending'])
    type: str = Field(description='The type of step.', enum=['search'])


@tool
def SearchTool(steps: List[SearchStep]): # pylint: disable=invalid-name,unused-argument
    """
    Break the user's query into smaller steps.
    Use step type "search" to search the web for information.
    Make sure to add all the steps needed to answer the user's query.
    """


async def steps_node(state: AgentState, config: RunnableConfig):
    """
    The steps node is responsible for building the steps in the research process.
    """

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {
                "state_key": "steps",
                "tool": "SearchTool",
                "tool_argument": "steps"
            },
        ]
    )

    instructions = f"""
You are a search assistant. Your task is to help the user with complex search queries by breaking the down into smaller steps.

These steps are then executed serially. In the end, a final answer is produced in markdown format.

The current date is {datetime.now().strftime("%Y-%m-%d")}.
"""

    response = await get_model(state).bind_tools(
        [SearchTool],
        tool_choice="SearchTool"
    ).ainvoke([
        state["messages"][0],
        HumanMessage(
            content=instructions
        ),
    ], config)

    if len(response.tool_calls) == 0:
        steps = []
    else:
        steps = response.tool_calls[0]["args"]["steps"]

    if len(steps) != 0:
        steps[0]["updates"] = ["Searching the web..."]

    return {
        "steps": steps,
    }

================
File: agent/ai_researcher/summarize.py
================
"""
The summarize node is responsible for summarizing the information.
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from copilotkit.langgraph import copilotkit_customize_config
from pydantic import BaseModel, Field
from ai_researcher.state import AgentState
from ai_researcher.model import get_model

class Reference(BaseModel):
    """Model for a reference"""

    title: str = Field(description="The title of the reference.")
    url: str = Field(description="The url of the reference.")

class SummarizeInput(BaseModel):
    """Input for the summarize tool"""
    markdown: str = Field(description="""
                          The markdown formatted summary of the final result.
                          If you add any headings, make sure to start at the top level (#).
                          """)
    references: list[Reference] = Field(description="A list of references.")

@tool(args_schema=SummarizeInput)
def SummarizeTool(summary: str, references: list[Reference]): # pylint: disable=invalid-name,unused-argument
    """
    Summarize the final result. Make sure that the summary is complete and 
    includes all relevant information and reference links.
    """


async def summarize_node(state: AgentState, config: RunnableConfig):
    """
    The summarize node is responsible for summarizing the information.
    """

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {
                "state_key": "answer",
                "tool": "SummarizeTool",
            }
        ]
    )

    system_message = f"""
The system has performed a series of steps to answer the user's query.
These are all of the steps: {json.dumps(state["steps"])}

Please summarize the final result and include all relevant information and reference links.
"""

    response = await get_model(state).bind_tools(
        [SummarizeTool],
        tool_choice="SummarizeTool"
    ).ainvoke([
        HumanMessage(
            content=system_message
        ),
    ], config)

    return {
        "answer": response.tool_calls[0]["args"],
    }

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "ai_researcher": "./ai_researcher/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "ai_researcher"
version = "0.1.0"
description = "AI Researcher Demo"
authors = ["Markus Ecker <markus.ecker@gmail.com>"]
license = "MIT"

[project]
name = "ai_researcher"
version = "0.0.1"
dependencies = [
  "langgraph",
  "langchain_core",
  "langchain_openai",
  "langchain-google-genai",
  "langchain",
  "openai",
  "langchain-community",
  "tavily-python",
  "python-dotenv",
  "uvicorn",
  "copilotkit==0.1.34",
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "0.2.3"
langchain-anthropic = "0.2.3"
langchain-google-genai = "2.0.0"
langchain = "0.3.4"
openai = "^1.52.1"
langchain-community = "^0.3.1"
tavily-python = "^0.5.0"
copilotkit = "0.1.39"
python-dotenv = "^1.0.1"
uvicorn = "^0.31.0"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}

[tool.poetry.scripts]
demo = "ai_researcher.demo:main"

================
File: README.md
================
# AI Researcher Example

This example is a simple AI based search engine.

You can find an online demo of this example [here](https://examples-coagents-ai-researcher-ui.vercel.app).

**These instructions assume you are in the `coagents-ai-researcher/` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/my_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": "next/core-web-vitals"
}

================
File: ui/.gitignore
================
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.js
.yarn/install-state.gz

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local

.env

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

================
File: ui/app/api/copilotkit-lgc/route.ts
================
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";
import { NextRequest } from "next/server";
import { langGraphPlatformEndpoint } from "@copilotkit/runtime";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const serviceAdapter = new OpenAIAdapter({ openai } as any);

const deploymentUrl = process.env.LGC_DEPLOYMENT_URL as string
const langsmithApiKey = process.env.LANGSMITH_API_KEY as string

const runtime = new CopilotRuntime({
  remoteEndpoints: [
    langGraphPlatformEndpoint({
      deploymentUrl,
      langsmithApiKey,
      agents: [{
        name: 'ai_researcher',
        description: 'Search agent.',
      }],
    }),
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit-lgc",
  });

  return handleRequest(req);
};

================
File: ui/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const serviceAdapter = new OpenAIAdapter({ openai } as any);

const runtime = new CopilotRuntime({
  remoteEndpoints: [
    {
      url: process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
    },
  ],
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
    --chart-1: 12 76% 61%;
    --chart-2: 173 58% 39%;
    --chart-3: 197 37% 24%;
    --chart-4: 43 74% 66%;
    --chart-5: 27 87% 67%;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
    --chart-1: 220 70% 50%;
    --chart-2: 160 60% 45%;
    --chart-3: 30 80% 55%;
    --chart-4: 280 65% 60%;
    --chart-5: 340 75% 55%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}

================
File: ui/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Researcher",
  description: "AI Researcher",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/app/page.tsx
================
"use client";

import { ModelSelector } from "@/components/ModelSelector";
import { ResearchWrapper } from "@/components/ResearchWrapper";
import { ModelSelectorProvider, useModelSelectorContext } from "@/lib/model-selector-provider";
import { ResearchProvider } from "@/lib/research-provider";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export default function ModelSelectorWrapper() {
  return (
      <main className="flex flex-col items-center justify-between">
        <ModelSelectorProvider>
            <Home/>
          <ModelSelector />
        </ModelSelectorProvider>
      </main>
  );
}

function Home() {
  const { useLgc } = useModelSelectorContext();

  return (
      <CopilotKit runtimeUrl={useLgc ? "/api/copilotkit-lgc" : "/api/copilotkit"} agent="ai_researcher">
        <ResearchProvider>
          <ResearchWrapper />
        </ResearchProvider>
      </CopilotKit>
  );
}

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "app/globals.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils"
  }
}

================
File: ui/components/AnswerMarkdown.tsx
================
import Markdown from "react-markdown";

export function AnswerMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="markdown-wrapper">
      <Markdown>{markdown}</Markdown>
    </div>
  );
}

================
File: ui/components/HomeView.tsx
================
"use client";

import { useEffect, useState } from "react";
import { Textarea } from "./ui/textarea";
import { cn } from "@/lib/utils";
import { Button } from "./ui/button";
import { CornerDownLeftIcon } from "lucide-react";
import { useResearchContext } from "@/lib/research-provider";
import { motion } from "framer-motion";
import { useCoAgent } from "@copilotkit/react-core";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";
import type { AgentState } from "../lib/types";
import { useModelSelectorContext } from "@/lib/model-selector-provider";

const MAX_INPUT_LENGTH = 250;

export function HomeView() {
  const { setResearchQuery, researchInput, setResearchInput } =
    useResearchContext();
  const { model } = useModelSelectorContext();
  const [isInputFocused, setIsInputFocused] = useState(false);
  const {
    run: runResearchAgent,
  } = useCoAgent<AgentState>({
    name: "ai_researcher",
    initialState: {
      model,
    },
  });

  const handleResearch = (query: string) => {
    setResearchQuery(query);
    runResearchAgent(() => {
      return new TextMessage({
        role: MessageRole.User,
        content: query,
      });
    });
  };

  const suggestions = [
    { label: "Electric cars sold in 2024 vs 2023", icon: "🚙" },
    { label: "Top 10 richest people in the world", icon: "💰" },
    { label: "Population of the World", icon: "🌍 " },
    { label: "Weather in Seattle VS New York", icon: "⛅️" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="h-screen w-full flex flex-col gap-y-2 justify-center items-center p-4 lg:p-0"
    >
      <h1 className="text-4xl font-extralight mb-6">
        What would you like to know?
      </h1>

      <div
        className={cn(
          "w-full bg-slate-100/50 border shadow-sm rounded-md transition-all",
          {
            "ring-1 ring-slate-300": isInputFocused,
          }
        )}
      >
        <Textarea
          placeholder="Ask anything..."
          className="bg-transparent p-4 resize-none focus-visible:ring-0 focus-visible:ring-offset-0 border-0 w-full"
          onFocus={() => setIsInputFocused(true)}
          onBlur={() => setIsInputFocused(false)}
          value={researchInput}
          onChange={(e) => setResearchInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleResearch(researchInput);
            }
          }}
          maxLength={MAX_INPUT_LENGTH}
        />
        <div className="text-xs p-4 flex items-center justify-between">
          <div
            className={cn("transition-all duration-300 mt-4 text-slate-500", {
              "opacity-0": !researchInput,
              "opacity-100": researchInput,
            })}
          >
            {researchInput.length} / {MAX_INPUT_LENGTH}
          </div>
          <Button
            size="sm"
            className={cn("rounded-full transition-all duration-300", {
              "opacity-0 pointer-events-none": !researchInput,
              "opacity-100": researchInput,
            })}
            onClick={() => handleResearch(researchInput)}
          >
            Research
            <CornerDownLeftIcon className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-2 w-full gap-2 text-sm">
        {suggestions.map((suggestion) => (
          <div
            key={suggestion.label}
            onClick={() => handleResearch(suggestion.label)}
            className="p-2 bg-slate-100/50 rounded-md border col-span-2 lg:col-span-1 flex cursor-pointer items-center space-x-2 hover:bg-slate-100 transition-all duration-300"
          >
            <span className="text-base">{suggestion.icon}</span>
            <span className="flex-1">{suggestion.label}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

================
File: ui/components/ModelSelector.tsx
================
"use client"

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModelSelectorContext } from "@/lib/model-selector-provider";

export function ModelSelector() {
  const { model, setModel } = useModelSelectorContext();

  return (
    <div className="fixed bottom-0 left-0 p-4 z-50">
      <Select value={model} onValueChange={v => setModel(v)}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Theme" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="openai">OpenAI</SelectItem>
          <SelectItem value="anthropic">Anthropic</SelectItem>
          <SelectItem value="google_genai">Google Generative AI</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

================
File: ui/components/Progress.tsx
================
import { cn } from "@/lib/utils";
import { CheckIcon, LoaderCircle } from "lucide-react";

export function Progress({
  steps,
}: {
  steps: {
    description: string;
    status: "complete" | "done";
    updates: string[];
  }[];
}) {
  if (steps.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="border border-slate-200 bg-slate-100/30 shadow-md rounded-lg overflow-hidden text-sm py-2">
        {steps.map((step, index) => (
          <div key={index} className="flex">
            <div className="w-8">
              <div className="w-4 h-4 bg-slate-700 flex items-center justify-center rounded-full mt-[10px] ml-[12px]">
                {step.status === "complete" ? (
                  <CheckIcon className="w-3 h-3 text-white" />
                ) : (
                  <LoaderCircle className="w-3 h-3 text-white animate-spin" />
                )}
              </div>
              {index < steps.length - 1 && (
                <div
                  className={cn("h-full w-[1px] bg-slate-200 ml-[20px]")}
                ></div>
              )}
            </div>
            <div className="flex-1 flex justify-center py-2 pl-2 pr-4">
              <div className="flex-1 flex items-center">{step.description}</div>
              <div className="text-slate-400">
                {step.updates?.length > 0 && step.updates[step.updates.length - 1]}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

================
File: ui/components/ResearchWrapper.tsx
================
import { HomeView } from "./HomeView";
import { ResultsView } from "./ResultsView";
import { AnimatePresence } from "framer-motion";
import { useResearchContext } from "@/lib/research-provider";

export function ResearchWrapper() {
  const { researchQuery, setResearchInput } = useResearchContext();

  return (
    <>
      <div className="flex flex-col items-center justify-center relative z-10">
        <div className="flex-1">
          {researchQuery ? (
            <AnimatePresence
              key="results"
              onExitComplete={() => {
                setResearchInput("");
              }}
              mode="wait"
            >
              <ResultsView key="results" />
            </AnimatePresence>
          ) : (
            <AnimatePresence key="home" mode="wait">
              <HomeView key="home" />
            </AnimatePresence>
          )}
        </div>
        <footer className="text-xs p-2">
          <a
            href="https://copilotkit.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-600 font-medium hover:underline"
          >
            Powered by CopilotKit 🪁
          </a>
        </footer>
      </div>
    </>
  );
}

================
File: ui/components/ResultsView.tsx
================
"use client";

import { useResearchContext } from "@/lib/research-provider";
import { motion } from "framer-motion";
import { BookOpenIcon, LoaderCircleIcon, SparkleIcon } from "lucide-react";
import { SkeletonLoader } from "./SkeletonLoader";
import { useCoAgent } from "@copilotkit/react-core";
import { Progress } from "./Progress";
import { AnswerMarkdown } from "./AnswerMarkdown";
import { AgentState } from "@/lib/types";
import { useModelSelectorContext } from "@/lib/model-selector-provider";

export function ResultsView() {
  const { researchQuery } = useResearchContext();
  const { model } = useModelSelectorContext();
  const { state: agentState } = useCoAgent<AgentState>({
    name: "ai_researcher",
    initialState: {
      model,
    },
  });

  const steps =
    agentState?.steps?.map((step: any) => {
      return {
        description: step.description || "",
        status: step.status || "pending",
        updates: step.updates || [],
      };
    }) || [];

  const isLoading = !agentState?.answer?.markdown;

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -50 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="max-w-[1000px] p-8 lg:p-4 flex flex-col gap-y-8 mt-4 lg:mt-6 text-sm lg:text-base">
        <div className="space-y-4">
          <h1 className="text-3xl lg:text-4xl font-extralight">
            {researchQuery}
          </h1>
        </div>

        <Progress steps={steps} />

        <div className="grid grid-cols-12 gap-8">
          <div className="col-span-12 lg:col-span-8 flex flex-col">
            <h2 className="flex items-center gap-x-2">
              {isLoading ? (
                <LoaderCircleIcon className="animate-spin w-4 h-4 text-slate-500" />
              ) : (
                <SparkleIcon className="w-4 h-4 text-slate-500" />
              )}
              Answer
            </h2>
            <div className="text-slate-500 font-light">
              {isLoading ? (
                null
              ) : (
                <AnswerMarkdown markdown={agentState?.answer?.markdown} />
              )}
            </div>
          </div>

          {agentState?.answer?.references?.length && (
            <div className="flex col-span-12 lg:col-span-4 flex-col gap-y-4 w-[200px]">
              <h2 className="flex items-center gap-x-2">
                <BookOpenIcon className="w-4 h-4 text-slate-500" />
                References
              </h2>
              <ul className="text-slate-900 font-light text-sm flex flex-col gap-y-2">
                {agentState?.answer?.references?.map(
                  (ref: any, idx: number) => (
                    <li key={idx}>
                      <a
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {idx + 1}. {ref.title}
                      </a>
                    </li>
                  )
                )}
              </ul>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

================
File: ui/components/SkeletonLoader.tsx
================
import { Skeleton } from "@/components/ui/skeleton";

export function SkeletonLoader() {
  return (
    <div className="grid grid-cols-10 gap-x-2 gap-y-4">
      <Skeleton className="h-4 col-span-2" />
      <Skeleton className="h-4 col-span-4" />
      <Skeleton className="h-4 col-span-4" />

      <Skeleton className="h-4 col-span-4" />
      <Skeleton className="h-4 col-span-6" />

      <Skeleton className="h-4 col-span-3" />
      <Skeleton className="h-4 col-span-3" />
      <Skeleton className="h-4 col-span-4" />

      <Skeleton className="h-4 col-span-5" />
      <Skeleton className="h-4 col-span-3" />
      <Skeleton className="h-4 col-span-2" />

      <Skeleton className="h-4 col-span-2" />
      <Skeleton className="h-4 col-span-4" />
      <Skeleton className="h-4 col-span-3" />
      <Skeleton className="h-4 col-span-1" />
    </div>
  )
}

================
File: ui/components/ui/accordion.tsx
================
"use client"

import * as React from "react"
import * as AccordionPrimitive from "@radix-ui/react-accordion"
import { ChevronDown } from "lucide-react"

import { cn } from "@/lib/utils"

const Accordion = AccordionPrimitive.Root

const AccordionItem = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>
>(({ className, ...props }, ref) => (
  <AccordionPrimitive.Item
    ref={ref}
    className={cn("", className)}
    {...props}
  />
))
AccordionItem.displayName = "AccordionItem"

const AccordionTrigger = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger> & {
    hideChevron?: boolean;
  }
>(({ className, hideChevron = false, children, ...props }, ref) => (
  <AccordionPrimitive.Header className="flex">
    <AccordionPrimitive.Trigger
      ref={ref}
      className={cn(
        "flex flex-1 items-center justify-between py-2 transition-all hover:bg-slate-100 px-2 rounded-md [&[data-state=open]>svg]:rotate-180",
        className
      )}
      {...props}
    >
      {children}
      {
        !hideChevron && (
          <ChevronDown className="h-4 w-4 mt-1 shrink-0 transition-transform duration-200" />
        )
      }
    </AccordionPrimitive.Trigger>
  </AccordionPrimitive.Header>
))
AccordionTrigger.displayName = AccordionPrimitive.Trigger.displayName

const AccordionContent = React.forwardRef<
  React.ElementRef<typeof AccordionPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <AccordionPrimitive.Content
    ref={ref}
    className="overflow-hidden text-sm transition-all data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
    {...props}
  >
    <div className={cn("pb-0 px-2 py-1", className)}>{children}</div>
  </AccordionPrimitive.Content>
))

AccordionContent.displayName = AccordionPrimitive.Content.displayName

export { Accordion, AccordionItem, AccordionTrigger, AccordionContent }

================
File: ui/components/ui/background-beams.tsx
================
"use client";
import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export const BackgroundBeams = React.memo(
  ({ className }: { className?: string }) => {
    const paths = [
      "M-380 -189C-380 -189 -312 216 152 343C616 470 684 875 684 875",
      "M-373 -197C-373 -197 -305 208 159 335C623 462 691 867 691 867",
      "M-366 -205C-366 -205 -298 200 166 327C630 454 698 859 698 859",
      "M-359 -213C-359 -213 -291 192 173 319C637 446 705 851 705 851",
      "M-352 -221C-352 -221 -284 184 180 311C644 438 712 843 712 843",
      "M-345 -229C-345 -229 -277 176 187 303C651 430 719 835 719 835",
      "M-338 -237C-338 -237 -270 168 194 295C658 422 726 827 726 827",
      "M-331 -245C-331 -245 -263 160 201 287C665 414 733 819 733 819",
      "M-324 -253C-324 -253 -256 152 208 279C672 406 740 811 740 811",
      "M-317 -261C-317 -261 -249 144 215 271C679 398 747 803 747 803",
      "M-310 -269C-310 -269 -242 136 222 263C686 390 754 795 754 795",
      "M-303 -277C-303 -277 -235 128 229 255C693 382 761 787 761 787",
      "M-296 -285C-296 -285 -228 120 236 247C700 374 768 779 768 779",
      "M-289 -293C-289 -293 -221 112 243 239C707 366 775 771 775 771",
      "M-282 -301C-282 -301 -214 104 250 231C714 358 782 763 782 763",
      "M-275 -309C-275 -309 -207 96 257 223C721 350 789 755 789 755",
      "M-268 -317C-268 -317 -200 88 264 215C728 342 796 747 796 747",
      "M-261 -325C-261 -325 -193 80 271 207C735 334 803 739 803 739",
      "M-254 -333C-254 -333 -186 72 278 199C742 326 810 731 810 731",
      "M-247 -341C-247 -341 -179 64 285 191C749 318 817 723 817 723",
      "M-240 -349C-240 -349 -172 56 292 183C756 310 824 715 824 715",
      "M-233 -357C-233 -357 -165 48 299 175C763 302 831 707 831 707",
      "M-226 -365C-226 -365 -158 40 306 167C770 294 838 699 838 699",
      "M-219 -373C-219 -373 -151 32 313 159C777 286 845 691 845 691",
      "M-212 -381C-212 -381 -144 24 320 151C784 278 852 683 852 683",
      "M-205 -389C-205 -389 -137 16 327 143C791 270 859 675 859 675",
      "M-198 -397C-198 -397 -130 8 334 135C798 262 866 667 866 667",
      "M-191 -405C-191 -405 -123 0 341 127C805 254 873 659 873 659",
      "M-184 -413C-184 -413 -116 -8 348 119C812 246 880 651 880 651",
      "M-177 -421C-177 -421 -109 -16 355 111C819 238 887 643 887 643",
      "M-170 -429C-170 -429 -102 -24 362 103C826 230 894 635 894 635",
      "M-163 -437C-163 -437 -95 -32 369 95C833 222 901 627 901 627",
      "M-156 -445C-156 -445 -88 -40 376 87C840 214 908 619 908 619",
      "M-149 -453C-149 -453 -81 -48 383 79C847 206 915 611 915 611",
      "M-142 -461C-142 -461 -74 -56 390 71C854 198 922 603 922 603",
      "M-135 -469C-135 -469 -67 -64 397 63C861 190 929 595 929 595",
      "M-128 -477C-128 -477 -60 -72 404 55C868 182 936 587 936 587",
      "M-121 -485C-121 -485 -53 -80 411 47C875 174 943 579 943 579",
      "M-114 -493C-114 -493 -46 -88 418 39C882 166 950 571 950 571",
      "M-107 -501C-107 -501 -39 -96 425 31C889 158 957 563 957 563",
      "M-100 -509C-100 -509 -32 -104 432 23C896 150 964 555 964 555",
      "M-93 -517C-93 -517 -25 -112 439 15C903 142 971 547 971 547",
      "M-86 -525C-86 -525 -18 -120 446 7C910 134 978 539 978 539",
      "M-79 -533C-79 -533 -11 -128 453 -1C917 126 985 531 985 531",
      "M-72 -541C-72 -541 -4 -136 460 -9C924 118 992 523 992 523",
      "M-65 -549C-65 -549 3 -144 467 -17C931 110 999 515 999 515",
      "M-58 -557C-58 -557 10 -152 474 -25C938 102 1006 507 1006 507",
      "M-51 -565C-51 -565 17 -160 481 -33C945 94 1013 499 1013 499",
      "M-44 -573C-44 -573 24 -168 488 -41C952 86 1020 491 1020 491",
      "M-37 -581C-37 -581 31 -176 495 -49C959 78 1027 483 1027 483",
    ];
    return (
      <div
        className={cn(
          "absolute  h-full w-full inset-0  [mask-size:40px] [mask-repeat:no-repeat] flex items-center justify-center",
          className
        )}
      >
        <svg
          className=" z-0 h-full w-full pointer-events-none absolute "
          width="100%"
          height="100%"
          viewBox="0 0 696 316"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M-380 -189C-380 -189 -312 216 152 343C616 470 684 875 684 875M-373 -197C-373 -197 -305 208 159 335C623 462 691 867 691 867M-366 -205C-366 -205 -298 200 166 327C630 454 698 859 698 859M-359 -213C-359 -213 -291 192 173 319C637 446 705 851 705 851M-352 -221C-352 -221 -284 184 180 311C644 438 712 843 712 843M-345 -229C-345 -229 -277 176 187 303C651 430 719 835 719 835M-338 -237C-338 -237 -270 168 194 295C658 422 726 827 726 827M-331 -245C-331 -245 -263 160 201 287C665 414 733 819 733 819M-324 -253C-324 -253 -256 152 208 279C672 406 740 811 740 811M-317 -261C-317 -261 -249 144 215 271C679 398 747 803 747 803M-310 -269C-310 -269 -242 136 222 263C686 390 754 795 754 795M-303 -277C-303 -277 -235 128 229 255C693 382 761 787 761 787M-296 -285C-296 -285 -228 120 236 247C700 374 768 779 768 779M-289 -293C-289 -293 -221 112 243 239C707 366 775 771 775 771M-282 -301C-282 -301 -214 104 250 231C714 358 782 763 782 763M-275 -309C-275 -309 -207 96 257 223C721 350 789 755 789 755M-268 -317C-268 -317 -200 88 264 215C728 342 796 747 796 747M-261 -325C-261 -325 -193 80 271 207C735 334 803 739 803 739M-254 -333C-254 -333 -186 72 278 199C742 326 810 731 810 731M-247 -341C-247 -341 -179 64 285 191C749 318 817 723 817 723M-240 -349C-240 -349 -172 56 292 183C756 310 824 715 824 715M-233 -357C-233 -357 -165 48 299 175C763 302 831 707 831 707M-226 -365C-226 -365 -158 40 306 167C770 294 838 699 838 699M-219 -373C-219 -373 -151 32 313 159C777 286 845 691 845 691M-212 -381C-212 -381 -144 24 320 151C784 278 852 683 852 683M-205 -389C-205 -389 -137 16 327 143C791 270 859 675 859 675M-198 -397C-198 -397 -130 8 334 135C798 262 866 667 866 667M-191 -405C-191 -405 -123 0 341 127C805 254 873 659 873 659M-184 -413C-184 -413 -116 -8 348 119C812 246 880 651 880 651M-177 -421C-177 -421 -109 -16 355 111C819 238 887 643 887 643M-170 -429C-170 -429 -102 -24 362 103C826 230 894 635 894 635M-163 -437C-163 -437 -95 -32 369 95C833 222 901 627 901 627M-156 -445C-156 -445 -88 -40 376 87C840 214 908 619 908 619M-149 -453C-149 -453 -81 -48 383 79C847 206 915 611 915 611M-142 -461C-142 -461 -74 -56 390 71C854 198 922 603 922 603M-135 -469C-135 -469 -67 -64 397 63C861 190 929 595 929 595M-128 -477C-128 -477 -60 -72 404 55C868 182 936 587 936 587M-121 -485C-121 -485 -53 -80 411 47C875 174 943 579 943 579M-114 -493C-114 -493 -46 -88 418 39C882 166 950 571 950 571M-107 -501C-107 -501 -39 -96 425 31C889 158 957 563 957 563M-100 -509C-100 -509 -32 -104 432 23C896 150 964 555 964 555M-93 -517C-93 -517 -25 -112 439 15C903 142 971 547 971 547M-86 -525C-86 -525 -18 -120 446 7C910 134 978 539 978 539M-79 -533C-79 -533 -11 -128 453 -1C917 126 985 531 985 531M-72 -541C-72 -541 -4 -136 460 -9C924 118 992 523 992 523M-65 -549C-65 -549 3 -144 467 -17C931 110 999 515 999 515M-58 -557C-58 -557 10 -152 474 -25C938 102 1006 507 1006 507M-51 -565C-51 -565 17 -160 481 -33C945 94 1013 499 1013 499M-44 -573C-44 -573 24 -168 488 -41C952 86 1020 491 1020 491M-37 -581C-37 -581 31 -176 495 -49C959 78 1027 483 1027 483M-30 -589C-30 -589 38 -184 502 -57C966 70 1034 475 1034 475M-23 -597C-23 -597 45 -192 509 -65C973 62 1041 467 1041 467M-16 -605C-16 -605 52 -200 516 -73C980 54 1048 459 1048 459M-9 -613C-9 -613 59 -208 523 -81C987 46 1055 451 1055 451M-2 -621C-2 -621 66 -216 530 -89C994 38 1062 443 1062 443M5 -629C5 -629 73 -224 537 -97C1001 30 1069 435 1069 435M12 -637C12 -637 80 -232 544 -105C1008 22 1076 427 1076 427M19 -645C19 -645 87 -240 551 -113C1015 14 1083 419 1083 419"
            stroke="url(#paint0_radial_242_278)"
            strokeOpacity="0.05"
            strokeWidth="0.5"
          ></path>

          {paths.map((path, index) => (
            <motion.path
              key={`path-` + index}
              d={path}
              stroke={`url(#linearGradient-${index})`}
              strokeOpacity="0.4"
              strokeWidth="0.5"
            ></motion.path>
          ))}
          <defs>
            {paths.map((path, index) => (
              <motion.linearGradient
                id={`linearGradient-${index}`}
                key={`gradient-${index}`}
                initial={{
                  x1: "0%",
                  x2: "0%",
                  y1: "0%",
                  y2: "0%",
                }}
                animate={{
                  x1: ["0%", "100%"],
                  x2: ["0%", "95%"],
                  y1: ["0%", "100%"],
                  y2: ["0%", `${93 + Math.random() * 8}%`],
                }}
                transition={{
                  duration: Math.random() * 10 + 10,
                  ease: "easeInOut",
                  repeat: Infinity,
                  delay: Math.random() * 10,
                }}
              >
                <stop stopColor="#18CCFC" stopOpacity="0"></stop>
                <stop stopColor="#18CCFC"></stop>
                <stop offset="32.5%" stopColor="#6344F5"></stop>
                <stop offset="100%" stopColor="#AE48FF" stopOpacity="0"></stop>
              </motion.linearGradient>
            ))}

            <radialGradient
              id="paint0_radial_242_278"
              cx="0"
              cy="0"
              r="1"
              gradientUnits="userSpaceOnUse"
              gradientTransform="translate(352 34) rotate(90) scale(555 1560.62)"
            >
              <stop offset="0.0666667" stopColor="var(--neutral-300)"></stop>
              <stop offset="0.243243" stopColor="var(--neutral-300)"></stop>
              <stop offset="0.43594" stopColor="white" stopOpacity="0"></stop>
            </radialGradient>
          </defs>
        </svg>
      </div>
    );
  }
);

BackgroundBeams.displayName = "BackgroundBeams";

================
File: ui/components/ui/button.tsx
================
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }

================
File: ui/components/ui/card.tsx
================
import * as React from "react"

import { cn } from "@/lib/utils"

const Card = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-lg border bg-card text-card-foreground shadow-sm",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }

================
File: ui/components/ui/input.tsx
================
import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }

================
File: ui/components/ui/meteors.tsx
================
import { cn } from "@/lib/utils";
import React from "react";

export const Meteors = ({
  number,
  className,
}: {
  number?: number;
  className?: string;
}) => {
  const meteors = new Array(number || 20).fill(true);
  return (
    <>
      {meteors.map((el, idx) => (
        <span
          key={"meteor" + idx}
          className={cn(
            "animate-meteor-effect absolute top-1/2 left-1/2 h-0.5 w-0.5 rounded-[9999px] bg-slate-500 shadow-[0_0_0_1px_#ffffff10] rotate-[215deg]",
            "before:content-[''] before:absolute before:top-1/2 before:transform before:-translate-y-[50%] before:w-[50px] before:h-[1px] before:bg-gradient-to-r before:from-[#64748b] before:to-transparent",
            className
          )}
          style={{
            top: 0,
            left: Math.floor(Math.random() * (400 - -400) + -400) + "px",
            animationDelay: Math.random() * (0.8 - 0.2) + 0.2 + "s",
            animationDuration: Math.floor(Math.random() * (10 - 2) + 2) + "s",
          }}
        ></span>
      ))}
    </>
  );
};

================
File: ui/components/ui/select.tsx
================
"use client"

import * as React from "react"
import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown, ChevronUp } from "lucide-react"

import { cn } from "@/lib/utils"

const Select = SelectPrimitive.Root

const SelectGroup = SelectPrimitive.Group

const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronUp className="h-4 w-4" />
  </SelectPrimitive.ScrollUpButton>
))
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronDown className="h-4 w-4" />
  </SelectPrimitive.ScrollDownButton>
))
SelectScrollDownButton.displayName =
  SelectPrimitive.ScrollDownButton.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("py-1.5 pl-8 pr-2 text-sm font-semibold", className)}
    {...props}
  />
))
SelectLabel.displayName = SelectPrimitive.Label.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>

    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
))
SelectSeparator.displayName = SelectPrimitive.Separator.displayName

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
}

================
File: ui/components/ui/skeleton.tsx
================
import { cn } from "@/lib/utils"

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }

================
File: ui/components/ui/textarea.tsx
================
import * as React from "react"

import { cn } from "@/lib/utils"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

export { Textarea }

================
File: ui/lib/model-selector-provider.tsx
================
"use client";

import React from "react";
import { createContext, useContext, useState, ReactNode } from "react";

type ModelSelectorContextType = {
  model: string;
  setModel: (model: string) => void;
  hidden: boolean;
  setHidden: (hidden: boolean) => void;
  useLgc: boolean;
};

const ModelSelectorContext = createContext<
  ModelSelectorContextType | undefined
>(undefined);

export const ModelSelectorProvider = ({
  children,
}: {
  children: ReactNode;
}) => {
  const model =
    globalThis.window === undefined
      ? "openai"
      : new URL(window.location.href).searchParams.get("coAgentsModel") ??
        "openai";
  const [hidden, setHidden] = useState<boolean>(false);

  const setModel = (model: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("coAgentsModel", model);
    window.location.href = url.toString();
  };

  const useLgc =
    globalThis.window === undefined
      ? false
      : !!new URL(window.location.href).searchParams.get("lgc") ||
        process.env.NEXT_PUBLIC_FORCE_LGC === "true";

  return (
    <ModelSelectorContext.Provider
      value={{
        model,
        hidden,
        useLgc,
        setModel,
        setHidden,
      }}
    >
      {children}
    </ModelSelectorContext.Provider>
  );
};

export const useModelSelectorContext = () => {
  const context = useContext(ModelSelectorContext);
  if (context === undefined) {
    throw new Error(
      "useModelSelectorContext must be used within a ModelSelectorProvider"
    );
  }
  return context;
};

================
File: ui/lib/research-provider.tsx
================
import { createContext, useContext, useState, ReactNode, useEffect } from "react";

type ResearchContextType = {
  researchQuery: string;
  setResearchQuery: (query: string) => void;
  researchInput: string;
  setResearchInput: (input: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  researchResult: ResearchResult | null;
  setResearchResult: (result: ResearchResult) => void;
};

type ResearchResult = {
  answer: string;
  sources: string[];
}

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider = ({ children }: { children: ReactNode }) => {
  const [researchQuery, setResearchQuery] = useState<string>("");
  const [researchInput, setResearchInput] = useState<string>("");
  const [researchResult, setResearchResult] = useState<ResearchResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!researchQuery) {
      setResearchResult(null);
      setResearchInput("");
    }
  }, [researchQuery, researchResult]);

  return (
    <ResearchContext.Provider
      value={{
        researchQuery,
        setResearchQuery,
        researchInput,
        setResearchInput,
        isLoading,
        setIsLoading,
        researchResult,
        setResearchResult,
      }}
    >
      {children}
    </ResearchContext.Provider>
  );
};

export const useResearchContext = () => {
  const context = useContext(ResearchContext);
  if (context === undefined) {
    throw new Error("useResearchContext must be used within a ResearchProvider");
  }
  return context;
};

================
File: ui/lib/types.ts
================
export type AgentState = {
  model: string;
  steps: any[];
  answer: {
    markdown: string;
    references: any[];
  };
}

================
File: ui/lib/utils.ts
================
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ai-researcher-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^12.0.0-alpha.2",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-hotkeys-hook": "^4.5.1",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7",
    "usehooks-ts": "^3.1.0"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/postcss.config.mjs
================
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;

================
File: ui/public/next.svg
================
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 394 80"><path fill="#000" d="M262 0h68.5v12.7h-27.2v66.6h-13.6V12.7H262V0ZM149 0v12.7H94v20.4h44.3v12.6H94v21h55v12.6H80.5V0h68.7zm34.3 0h-17.8l63.8 79.4h17.9l-32-39.7 32-39.6h-17.9l-23 28.6-23-28.6zm18.3 56.7-9-11-27.1 33.7h17.8l18.3-22.7z"/><path fill="#000" d="M81 79.3 17 0H0v79.3h13.6V17l50.2 62.3H81Zm252.6-.4c-1 0-1.8-.4-2.5-1s-1.1-1.6-1.1-2.6.3-1.8 1-2.5 1.6-1 2.6-1 1.8.3 2.5 1a3.4 3.4 0 0 1 .6 4.3 3.7 3.7 0 0 1-3 1.8zm23.2-33.5h6v23.3c0 2.1-.4 4-1.3 5.5a9.1 9.1 0 0 1-3.8 3.5c-1.6.8-3.5 1.3-5.7 1.3-2 0-3.7-.4-5.3-1s-2.8-1.8-3.7-3.2c-.9-1.3-1.4-3-1.4-5h6c.1.8.3 1.6.7 2.2s1 1.2 1.6 1.5c.7.4 1.5.5 2.4.5 1 0 1.8-.2 2.4-.6a4 4 0 0 0 1.6-1.8c.3-.8.5-1.8.5-3V45.5zm30.9 9.1a4.4 4.4 0 0 0-2-3.3 7.5 7.5 0 0 0-4.3-1.1c-1.3 0-2.4.2-3.3.5-.9.4-1.6 1-2 1.6a3.5 3.5 0 0 0-.3 4c.3.5.7.9 1.3 1.2l1.8 1 2 .5 3.2.8c1.3.3 2.5.7 3.7 1.2a13 13 0 0 1 3.2 1.8 8.1 8.1 0 0 1 3 6.5c0 2-.5 3.7-1.5 5.1a10 10 0 0 1-4.4 3.5c-1.8.8-4.1 1.2-6.8 1.2-2.6 0-4.9-.4-6.8-1.2-2-.8-3.4-2-4.5-3.5a10 10 0 0 1-1.7-5.6h6a5 5 0 0 0 3.5 4.6c1 .4 2.2.6 3.4.6 1.3 0 2.5-.2 3.5-.6 1-.4 1.8-1 2.4-1.7a4 4 0 0 0 .8-2.4c0-.9-.2-1.6-.7-2.2a11 11 0 0 0-2.1-1.4l-3.2-1-3.8-1c-2.8-.7-5-1.7-6.6-3.2a7.2 7.2 0 0 1-2.4-5.7 8 8 0 0 1 1.7-5 10 10 0 0 1 4.3-3.5c2-.8 4-1.2 6.4-1.2 2.3 0 4.4.4 6.2 1.2 1.8.8 3.2 2 4.3 3.4 1 1.4 1.5 3 1.5 5h-5.8z"/></svg>

================
File: ui/public/vercel.svg
================
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 283 64"><path fill="black" d="M141 16c-11 0-19 7-19 18s9 18 20 18c7 0 13-3 16-7l-7-5c-2 3-6 4-9 4-5 0-9-3-10-7h28v-3c0-11-8-18-19-18zm-9 15c1-4 4-7 9-7s8 3 9 7h-18zm117-15c-11 0-19 7-19 18s9 18 20 18c6 0 12-3 16-7l-8-5c-2 3-5 4-8 4-5 0-9-3-11-7h28l1-3c0-11-8-18-19-18zm-10 15c2-4 5-7 10-7s8 3 9 7h-19zm-39 3c0 6 4 10 10 10 4 0 7-2 9-5l8 5c-3 5-9 8-17 8-11 0-19-7-19-18s8-18 19-18c8 0 14 3 17 8l-8 5c-2-3-5-5-9-5-6 0-10 4-10 10zm83-29v46h-9V5h9zM37 0l37 64H0L37 0zm92 5-27 48L74 5h10l18 30 17-30h10zm59 12v10l-3-1c-6 0-10 4-10 10v15h-9V17h9v9c0-5 6-9 13-9z"/></svg>

================
File: ui/README.md
================
This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";
const plugin = require('tailwindcss/plugin')
const {
  default: flattenColorPalette,
} = require("tailwindcss/lib/util/flattenColorPalette");

const config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: "true",
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        meteor: {
          "0%": {
            transform: "rotate(215deg) translateX(0)",
            opacity: "1",
          },
          "70%": {
            opacity: "1",
          },
          "100%": {
            transform: "rotate(215deg) translateX(-500px)",
            opacity: "0",
          },
        },
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "meteor-effect": "meteor 5s linear infinite",
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate"), addVariablesForColors, plugin(capitalizeFirstLetter)],
} satisfies Config;

function capitalizeFirstLetter({ addUtilities }: any) {
  const newUtilities = {
    '.capitalize-first:first-letter': {
      textTransform: 'uppercase',
    },
  }
  addUtilities(newUtilities, ['responsive', 'hover'])
}

function addVariablesForColors({ addBase, theme }: any) {
  let allColors = flattenColorPalette(theme("colors"));
  let newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

export default config;

================
File: ui/tsconfig.json
================
{
  "compilerOptions": {
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": [
        "./*"
      ]
    },
    "target": "ES2017"
  },
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx",
    ".next/types/**/*.ts",
    "tailwind.config.js"
  ],
  "exclude": [
    "node_modules",
    "tailwind.config.ts"
  ]
}



================================================================
End of Codebase
================================================================


# CoAgents-qa-native example:

This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
File Summary
================================================================

Purpose:
--------
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

File Format:
------------
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A separator line (================)
  b. The file path (File: path/to/file)
  c. Another separator line
  d. The full contents of the file
  e. A blank line

Usage Guidelines:
-----------------
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

Notes:
------
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

Additional Info:
----------------

================================================================
Directory Structure
================================================================
agent-js/.gitignore
agent-js/.vscode/cspell.json
agent-js/.vscode/settings.json
agent-js/langgraph.json
agent-js/package.json
agent-js/src/agent.ts
agent-js/src/model.ts
agent-js/src/state.ts
agent-js/tsconfig.json
agent/.gitignore
agent/.vscode/cspell.json
agent/.vscode/settings.json
agent/email_agent/agent.py
agent/email_agent/demo.py
agent/email_agent/model.py
agent/email_agent/state.py
agent/langgraph.json
agent/pyproject.toml
README.md
ui/.eslintrc.json
ui/.gitignore
ui/app/api/copilotkit/route.ts
ui/app/globals.css
ui/app/layout.tsx
ui/app/Mailer.tsx
ui/app/page.tsx
ui/components.json
ui/components/ModelSelector.tsx
ui/components/ui/select.tsx
ui/lib/model-selector-provider.tsx
ui/lib/utils.ts
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/public/next.svg
ui/public/vercel.svg
ui/README.md
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent-js/.gitignore
================
venv/
__pycache__/
*.pyc
.env
.vercel

# LangGraph API
.langgraph_api

================
File: agent-js/.vscode/cspell.json
================
{
  "version": "0.2",
  "language": "en",
  "words": [
    "langgraph",
    "langchain",
    "perplexity",
    "openai",
    "ainvoke",
    "pydantic",
    "tavily",
    "copilotkit",
    "fastapi",
    "uvicorn",
    "checkpointer",
    "dotenv"
  ]
}

================
File: agent-js/.vscode/settings.json
================
{
  "python.analysis.typeCheckingMode": "basic"
}

================
File: agent-js/langgraph.json
================
{
  "node_version": "20",
  "dockerfile_lines": ["RUN npm i -g corepack@latest"],
  "dependencies": ["."],
  "graphs": {
    "email_agent": "./src/agent.ts:graph"
  },
  "env": ".env"
}

================
File: agent-js/package.json
================
{
  "name": "agent_js",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "packageManager": "pnpm@9.5.0",
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "@langchain/langgraph-cli": "^0.0.10",
    "@types/html-to-text": "^9.0.4",
    "@types/node": "^22.9.0",
    "typescript": "^5.6.3"
  },
  "dependencies": {
    "@copilotkit/sdk-js": "^1.5.13",
    "@langchain/anthropic": "^0.3.8",
    "@langchain/core": "^0.3.18",
    "@langchain/google-genai": "^0.1.4",
    "@langchain/langgraph": "^0.2.44",
    "@langchain/openai": "^0.3.14",
    "zod": "^3.23.8"
  }
}

================
File: agent-js/src/agent.ts
================
/**
 * Test Q&A Agent
 */

import { RunnableConfig } from "@langchain/core/runnables";
import {
  copilotkitExit,
  convertActionsToDynamicStructuredTools,
} from "@copilotkit/sdk-js/langgraph";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { getModel } from "./model";
import { END, MemorySaver, StateGraph, interrupt } from "@langchain/langgraph";
import { AgentState, AgentStateAnnotation } from "./state";
import { copilotKitInterrupt } from "@copilotkit/sdk-js/langgraph";

export async function email_node(state: AgentState, config: RunnableConfig) {
  /**
   * Write an email.
   */

  const sender = state.sender ?? interrupt('Please provide a sender name which will appear in the email');
  let senderCompany = state.senderCompany
  let interruptMessages = []
  if (!senderCompany?.length) {
    const { answer, messages } = copilotKitInterrupt({ message: 'Ah, forgot to ask, which company are you working for?' });
    senderCompany = answer;
    interruptMessages = messages;
  }
  const instructions = `You write emails. The email is by the following sender: ${sender}, working for: ${senderCompany}`;

  const email_model = getModel(state).bindTools!(
    convertActionsToDynamicStructuredTools(state.copilotkit.actions),
    {
      tool_choice: "EmailTool",
    }
  );

  const response = await email_model.invoke(
    [...state.messages, ...interruptMessages, new HumanMessage({ content: instructions })],
    config
  );

  const tool_calls = response.tool_calls;

  const email = tool_calls?.[0]?.args.the_email;

  return {
    messages: response,
    email: email,
    sender,
    senderCompany,
  };
}

export async function send_email_node(
  state: AgentState,
  config: RunnableConfig
) {
  /**
   * Send an email.
   */

  await copilotkitExit(config);

  const lastMessage = state.messages[state.messages.length - 1] as ToolMessage;
  const content =
    lastMessage.content === "CANCEL"
      ? "❌ Cancelled sending email."
      : "✅ Sent email.";

  return {
    messages: new AIMessage({ content }),
  };
}

const workflow = new StateGraph(AgentStateAnnotation)
  .addNode("email_node", email_node)
  .addNode("send_email_node", send_email_node)
  .setEntryPoint("email_node")
  .addEdge("email_node", "send_email_node")
  .addEdge("send_email_node", END);

const memory = new MemorySaver();

export const graph = workflow.compile({
  checkpointer: memory,
  interruptAfter: ["email_node"],
});

================
File: agent-js/src/model.ts
================
/**
 * This module provides a function to get a model based on the configuration.
 */
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AgentState } from "./state";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

function getModel(state: AgentState): BaseChatModel {
  /**
   * Get a model based on the environment variable.
   */
  const stateModel = state.model;
  const model = process.env.MODEL || stateModel;

  console.log(`Using model: ${model}`);

  if (model === "openai") {
    return new ChatOpenAI({ temperature: 0, model: "gpt-4o" });
  }
  if (model === "anthropic") {
    return new ChatAnthropic({
      temperature: 0,
      modelName: "claude-3-5-sonnet-20240620",
    });
  }
  if (model === "google_genai") {
    return new ChatGoogleGenerativeAI({
      temperature: 0,
      model: "gemini-1.5-pro",
      apiKey: process.env.GOOGLE_API_KEY || undefined,
    });
  }

  throw new Error("Invalid model specified");
}

export { getModel };

================
File: agent-js/src/state.ts
================
import { Annotation } from "@langchain/langgraph";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";

// Define the AgentState annotation, extending MessagesState
export const AgentStateAnnotation = Annotation.Root({
  model: Annotation<string>,
  email: Annotation<string>,
  sender: Annotation<string>,
  senderCompany: Annotation<string>,
  ...CopilotKitStateAnnotation.spec,
});

export type AgentState = typeof AgentStateAnnotation.State;

================
File: agent-js/tsconfig.json
================
{
  "compilerOptions": {
    /* Visit https://aka.ms/tsconfig to read more about this file */

    /* Projects */
    // "incremental": true,                              /* Save .tsbuildinfo files to allow for incremental compilation of projects. */
    // "composite": true,                                /* Enable constraints that allow a TypeScript project to be used with project references. */
    // "tsBuildInfoFile": "./.tsbuildinfo",              /* Specify the path to .tsbuildinfo incremental compilation file. */
    // "disableSourceOfProjectReferenceRedirect": true,  /* Disable preferring source files instead of declaration files when referencing composite projects. */
    // "disableSolutionSearching": true,                 /* Opt a project out of multi-project reference checking when editing. */
    // "disableReferencedProjectLoad": true,             /* Reduce the number of projects loaded automatically by TypeScript. */

    /* Language and Environment */
    "target": "es2016",                                  /* Set the JavaScript language version for emitted JavaScript and include compatible library declarations. */
    // "lib": [],                                        /* Specify a set of bundled library declaration files that describe the target runtime environment. */
    // "jsx": "preserve",                                /* Specify what JSX code is generated. */
    // "experimentalDecorators": true,                   /* Enable experimental support for legacy experimental decorators. */
    // "emitDecoratorMetadata": true,                    /* Emit design-type metadata for decorated declarations in source files. */
    // "jsxFactory": "",                                 /* Specify the JSX factory function used when targeting React JSX emit, e.g. 'React.createElement' or 'h'. */
    // "jsxFragmentFactory": "",                         /* Specify the JSX Fragment reference used for fragments when targeting React JSX emit e.g. 'React.Fragment' or 'Fragment'. */
    // "jsxImportSource": "",                            /* Specify module specifier used to import the JSX factory functions when using 'jsx: react-jsx*'. */
    // "reactNamespace": "",                             /* Specify the object invoked for 'createElement'. This only applies when targeting 'react' JSX emit. */
    // "noLib": true,                                    /* Disable including any library files, including the default lib.d.ts. */
    // "useDefineForClassFields": true,                  /* Emit ECMAScript-standard-compliant class fields. */
    // "moduleDetection": "auto",                        /* Control what method is used to detect module-format JS files. */

    /* Modules */
    "module": "commonjs",                                /* Specify what module code is generated. */
    // "rootDir": "./",                                  /* Specify the root folder within your source files. */
    // "moduleResolution": "node10",                     /* Specify how TypeScript looks up a file from a given module specifier. */
    // "baseUrl": "./",                                  /* Specify the base directory to resolve non-relative module names. */
    // "paths": {},                                      /* Specify a set of entries that re-map imports to additional lookup locations. */
    // "rootDirs": [],                                   /* Allow multiple folders to be treated as one when resolving modules. */
    // "typeRoots": [],                                  /* Specify multiple folders that act like './node_modules/@types'. */
    // "types": [],                                      /* Specify type package names to be included without being referenced in a source file. */
    // "allowUmdGlobalAccess": true,                     /* Allow accessing UMD globals from modules. */
    // "moduleSuffixes": [],                             /* List of file name suffixes to search when resolving a module. */
    // "allowImportingTsExtensions": true,               /* Allow imports to include TypeScript file extensions. Requires '--moduleResolution bundler' and either '--noEmit' or '--emitDeclarationOnly' to be set. */
    // "resolvePackageJsonExports": true,                /* Use the package.json 'exports' field when resolving package imports. */
    // "resolvePackageJsonImports": true,                /* Use the package.json 'imports' field when resolving imports. */
    // "customConditions": [],                           /* Conditions to set in addition to the resolver-specific defaults when resolving imports. */
    // "noUncheckedSideEffectImports": true,             /* Check side effect imports. */
    // "resolveJsonModule": true,                        /* Enable importing .json files. */
    // "allowArbitraryExtensions": true,                 /* Enable importing files with any extension, provided a declaration file is present. */
    // "noResolve": true,                                /* Disallow 'import's, 'require's or '<reference>'s from expanding the number of files TypeScript should add to a project. */

    /* JavaScript Support */
    // "allowJs": true,                                  /* Allow JavaScript files to be a part of your program. Use the 'checkJS' option to get errors from these files. */
    // "checkJs": true,                                  /* Enable error reporting in type-checked JavaScript files. */
    // "maxNodeModuleJsDepth": 1,                        /* Specify the maximum folder depth used for checking JavaScript files from 'node_modules'. Only applicable with 'allowJs'. */

    /* Emit */
    // "declaration": true,                              /* Generate .d.ts files from TypeScript and JavaScript files in your project. */
    // "declarationMap": true,                           /* Create sourcemaps for d.ts files. */
    // "emitDeclarationOnly": true,                      /* Only output d.ts files and not JavaScript files. */
    // "sourceMap": true,                                /* Create source map files for emitted JavaScript files. */
    // "inlineSourceMap": true,                          /* Include sourcemap files inside the emitted JavaScript. */
    // "noEmit": true,                                   /* Disable emitting files from a compilation. */
    // "outFile": "./",                                  /* Specify a file that bundles all outputs into one JavaScript file. If 'declaration' is true, also designates a file that bundles all .d.ts output. */
    // "outDir": "./",                                   /* Specify an output folder for all emitted files. */
    // "removeComments": true,                           /* Disable emitting comments. */
    // "importHelpers": true,                            /* Allow importing helper functions from tslib once per project, instead of including them per-file. */
    // "downlevelIteration": true,                       /* Emit more compliant, but verbose and less performant JavaScript for iteration. */
    // "sourceRoot": "",                                 /* Specify the root path for debuggers to find the reference source code. */
    // "mapRoot": "",                                    /* Specify the location where debugger should locate map files instead of generated locations. */
    // "inlineSources": true,                            /* Include source code in the sourcemaps inside the emitted JavaScript. */
    // "emitBOM": true,                                  /* Emit a UTF-8 Byte Order Mark (BOM) in the beginning of output files. */
    // "newLine": "crlf",                                /* Set the newline character for emitting files. */
    // "stripInternal": true,                            /* Disable emitting declarations that have '@internal' in their JSDoc comments. */
    // "noEmitHelpers": true,                            /* Disable generating custom helper functions like '__extends' in compiled output. */
    // "noEmitOnError": true,                            /* Disable emitting files if any type checking errors are reported. */
    // "preserveConstEnums": true,                       /* Disable erasing 'const enum' declarations in generated code. */
    // "declarationDir": "./",                           /* Specify the output directory for generated declaration files. */

    /* Interop Constraints */
    // "isolatedModules": true,                          /* Ensure that each file can be safely transpiled without relying on other imports. */
    // "verbatimModuleSyntax": true,                     /* Do not transform or elide any imports or exports not marked as type-only, ensuring they are written in the output file's format based on the 'module' setting. */
    // "isolatedDeclarations": true,                     /* Require sufficient annotation on exports so other tools can trivially generate declaration files. */
    // "allowSyntheticDefaultImports": true,             /* Allow 'import x from y' when a module doesn't have a default export. */
    "esModuleInterop": true,                             /* Emit additional JavaScript to ease support for importing CommonJS modules. This enables 'allowSyntheticDefaultImports' for type compatibility. */
    // "preserveSymlinks": true,                         /* Disable resolving symlinks to their realpath. This correlates to the same flag in node. */
    "forceConsistentCasingInFileNames": true,            /* Ensure that casing is correct in imports. */

    /* Type Checking */
    "strict": true,                                      /* Enable all strict type-checking options. */
    // "noImplicitAny": true,                            /* Enable error reporting for expressions and declarations with an implied 'any' type. */
    // "strictNullChecks": true,                         /* When type checking, take into account 'null' and 'undefined'. */
    // "strictFunctionTypes": true,                      /* When assigning functions, check to ensure parameters and the return values are subtype-compatible. */
    // "strictBindCallApply": true,                      /* Check that the arguments for 'bind', 'call', and 'apply' methods match the original function. */
    // "strictPropertyInitialization": true,             /* Check for class properties that are declared but not set in the constructor. */
    // "strictBuiltinIteratorReturn": true,              /* Built-in iterators are instantiated with a 'TReturn' type of 'undefined' instead of 'any'. */
    // "noImplicitThis": true,                           /* Enable error reporting when 'this' is given the type 'any'. */
    // "useUnknownInCatchVariables": true,               /* Default catch clause variables as 'unknown' instead of 'any'. */
    // "alwaysStrict": true,                             /* Ensure 'use strict' is always emitted. */
    // "noUnusedLocals": true,                           /* Enable error reporting when local variables aren't read. */
    // "noUnusedParameters": true,                       /* Raise an error when a function parameter isn't read. */
    // "exactOptionalPropertyTypes": true,               /* Interpret optional property types as written, rather than adding 'undefined'. */
    // "noImplicitReturns": true,                        /* Enable error reporting for codepaths that do not explicitly return in a function. */
    // "noFallthroughCasesInSwitch": true,               /* Enable error reporting for fallthrough cases in switch statements. */
    // "noUncheckedIndexedAccess": true,                 /* Add 'undefined' to a type when accessed using an index. */
    // "noImplicitOverride": true,                       /* Ensure overriding members in derived classes are marked with an override modifier. */
    // "noPropertyAccessFromIndexSignature": true,       /* Enforces using indexed accessors for keys declared using an indexed type. */
    // "allowUnusedLabels": true,                        /* Disable error reporting for unused labels. */
    // "allowUnreachableCode": true,                     /* Disable error reporting for unreachable code. */

    /* Completeness */
    // "skipDefaultLibCheck": true,                      /* Skip type checking .d.ts files that are included with TypeScript. */
    "skipLibCheck": true                                 /* Skip type checking all .d.ts files. */
  }
}

================
File: agent/.gitignore
================
venv/
__pycache__/
*.pyc
.env
.vercel

================
File: agent/.vscode/cspell.json
================
{
  "version": "0.2",
  "language": "en",
  "words": [
    "langgraph",
    "langchain",
    "perplexity",
    "openai",
    "ainvoke",
    "pydantic",
    "tavily",
    "copilotkit",
    "fastapi",
    "uvicorn",
    "checkpointer",
    "dotenv"
  ]
}

================
File: agent/.vscode/settings.json
================
{
  "python.analysis.typeCheckingMode": "basic"
}

================
File: agent/email_agent/agent.py
================
"""Test Q&A Agent"""

from typing import Any, cast
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from copilotkit.langgraph import (
  copilotkit_customize_config, copilotkit_exit, copilotkit_emit_message, copilotkit_interrupt
)
from langgraph.types import interrupt
from email_agent.model import get_model
from email_agent.state import EmailAgentState


async def email_node(state: EmailAgentState, config: RunnableConfig):
    """
    Write an email.
    """

    sender = state.get("sender", None)
    if sender is None:
        sender = interrupt('Please provide a sender name which will appear in the email')

    sender_company = state.get("sender_company", None)
    if sender_company is None:
        sender_company, new_messages = copilotkit_interrupt(message='Ah, forgot to ask, which company are you working for?')
        state["messages"] = state["messages"] + new_messages

    config = copilotkit_customize_config(
        config,
        emit_tool_calls=True,
    )

    instructions = f"You write emails. The email is by the following sender: {sender}, working for: {sender_company}"

    cpk_actions = state.get("copilotkit", {}).get("actions", [])
    email_model = get_model(state).bind_tools(
        cpk_actions,
        tool_choice="EmailTool"
    )

    response = await email_model.ainvoke([
        *state["messages"],
        HumanMessage(
            content=instructions
        )
    ], config)

    tool_calls = cast(Any, response).tool_calls

    email = tool_calls[0]["args"]["the_email"]

    return {
        "messages": response,
        "email": email,
        "sender": sender,
        "sender_company": sender_company
    }

async def send_email_node(state: EmailAgentState, config: RunnableConfig):
    """
    Send an email.
    """

    config = copilotkit_customize_config(
        config,
        emit_messages=True,
    )


    await copilotkit_exit(config)

    # get the last message and cast to ToolMessage
    last_message = cast(ToolMessage, state["messages"][-1])

    if last_message.content == "CANCEL":
        text_message = "❌ Cancelled sending email."
    else:
        text_message = "✅ Sent email."
    
    await copilotkit_emit_message(config, text_message)


    return {
        "messages": AIMessage(content=text_message),
    }


workflow = StateGraph(EmailAgentState)
workflow.add_node("email_node", email_node)
workflow.add_node("send_email_node", send_email_node)
workflow.set_entry_point("email_node")
workflow.add_edge("email_node", "send_email_node")
workflow.add_edge("send_email_node", END)
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_after=["email_node"])

================
File: agent/email_agent/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from email_agent.agent import graph

app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="email_agent",
            description="This agent sends emails",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "email_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

================
File: agent/email_agent/model.py
================
"""
This module provides a function to get a model based on the configuration.
"""
import os
from langchain_core.language_models.chat_models import BaseChatModel
from email_agent.state import EmailAgentState


def get_model(state: EmailAgentState) -> BaseChatModel:
    """
    Get a model based on the environment variable.
    """

    state_model = state.get("model")
    model = os.getenv("MODEL", state_model)

    print(f"Using model: {model}")

    if model == "openai":
        from langchain_openai import ChatOpenAI # pylint: disable=import-outside-toplevel
        return ChatOpenAI(temperature=0, model="gpt-4o-mini")
    if model == "anthropic":
        from langchain_anthropic import ChatAnthropic # pylint: disable=import-outside-toplevel
        return ChatAnthropic(
            temperature=0, 
            model_name="claude-3-5-sonnet-20240620",
            timeout=None,
            stop=None
        )
    if model == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI # pylint: disable=import-outside-toplevel
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro")

    raise ValueError("Invalid model specified")

================
File: agent/email_agent/state.py
================
"""
This is the state definition for the AI.
It defines the state of the agent and the state of the conversation.
"""

from copilotkit import CopilotKitState

class EmailAgentState(CopilotKitState):
    """Email Agent State"""
    email: str
    model: str
    sender: str
    sender_company: str

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "email_agent": "./email_agent/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "email_agent"
version = "0.1.0"
description = "Starter"
authors = ["Ariel Weinberger <weinberger.ariel@gmail.com>"]
license = "MIT"

[project]
name = "email_agent"
version = "0.0.1"
dependencies = [
  "langgraph",
  "langchain_core",
  "langchain_openai",
  "langchain-google-genai",
  "langchain",
  "openai",
  "langchain-community",
  "copilotkit==0.1.35a3"
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
langchain-openai = "0.2.3"
langchain-anthropic = "0.2.3"
langchain-google-genai = "2.0.0"
langchain = "0.3.4"
openai = "^1.52.1"
langchain-community = "^0.3.1"
copilotkit = "0.1.39"
langgraph = ">=0.2.50"
uvicorn = "^0.31.0"
python-dotenv = "^1.0.1"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}

[tool.poetry.scripts]
demo = "email_agent.demo:main"

================
File: README.md
================
# CoAgents Agent Q&A Example

This example demonstrates sending a question to the user that gets rendered in an alert.

You can find an online demo of this example [here](https://examples-coagents-qa-native-ui.vercel.app).

**These instructions assume you are in the `coagents-qa-native/` directory**

## Running the Agent

First, install the dependencies:

```sh
cd agent
poetry install
```

Then, create a `.env` file inside `./agent` with the following:

```
OPENAI_API_KEY=...
```

IMPORTANT:
Make sure the OpenAI API Key you provide, supports gpt-4o.

Then, run the demo:

```sh
poetry run demo
```

## Running the UI

First, install the dependencies:

```sh
cd ./ui
pnpm i
```

Then, create a `.env` file inside `./ui` with the following:

```
OPENAI_API_KEY=...
```

Then, run the Next.js project:

```sh
pnpm run dev
```

## Usage

Navigate to [http://localhost:3000](http://localhost:3000).

# LangGraph Studio

Run LangGraph studio, then load the `./agent` folder into it.

Make sure to create teh `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/email_agent/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": "next/core-web-vitals"
}

================
File: ui/.gitignore
================
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.js
.yarn/install-state.gz

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local

.env

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

================
File: ui/app/api/copilotkit/route.ts
================
import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  langGraphPlatformEndpoint,
  copilotKitEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";

const openai = new OpenAI();
const llmAdapter = new OpenAIAdapter({ openai } as any);
const langsmithApiKey = process.env.LANGSMITH_API_KEY as string;

export const POST = async (req: NextRequest) => {
  const searchParams = req.nextUrl.searchParams;
  const deploymentUrl = searchParams.get("lgcDeploymentUrl");

  const remoteEndpoint = deploymentUrl
    ? langGraphPlatformEndpoint({
        deploymentUrl,
        langsmithApiKey,
        agents: [
          {
            name: "email_agent",
            description: "This agent sends emails",
          },
        ],
      })
    : copilotKitEndpoint({
        url:
          process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
      });

  const runtime = new CopilotRuntime({
    remoteEndpoints: [remoteEndpoint],
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: llmAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};

================
File: ui/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
    --chart-1: 12 76% 61%;
    --chart-2: 173 58% 39%;
    --chart-3: 197 37% 24%;
    --chart-4: 43 74% 66%;
    --chart-5: 27 87% 67%;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
    --chart-1: 220 70% 50%;
    --chart-2: 160 60% 45%;
    --chart-3: 30 80% 55%;
    --chart-4: 280 65% 60%;
    --chart-5: 340 75% 55%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}

================
File: ui/app/layout.tsx
================
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CoAgents Starter",
  description: "CoAgents Starter",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={inter.className}>{children}</body>
    </html>
  );
}

================
File: ui/app/Mailer.tsx
================
"use client";

import { useModelSelectorContext } from "@/lib/model-selector-provider";
import { useCoAgent, useCopilotAction } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import { useState } from "react";
import { useCopilotChatSuggestions } from "@copilotkit/react-ui";
import { useLangGraphInterrupt } from "@copilotkit/react-core";

const InterruptForm = ({ event, resolve }: { event: { value: string }, resolve: (value: string) => void }) => {
  const [name, setName] = useState<string>("");
  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="text-lg font-medium">{event.value}</div>
      <input 
        type="text"
        placeholder="Your name"
        className="border p-2 rounded"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <button
        onClick={() => resolve(name)} 
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Submit
      </button>
    </div>
  );
};

export function Mailer() {
  const { model } = useModelSelectorContext();
  const [messageState, setMessageState] = useState<"SEND" | "CANCEL" | null>(
    null
  );

  useCopilotChatSuggestions({
    instructions: "Write an email to the CEO of OpenAI asking for a meeting",
  });

  useCoAgent({
    name: "email_agent",
    initialState: {
      model,
    },
  });

  useCopilotAction({
    name: "EmailTool",
    available: "remote",
    parameters: [
      {
        name: "the_email",
      },
    ],

    handler: async ({ the_email }) => {
      const result = window.confirm(the_email);
      const action = result ? "SEND" : "CANCEL";
      setMessageState(action);
      return action;
    },
  });

  useLangGraphInterrupt({
    render: ({ event, resolve }) => <InterruptForm event={event} resolve={resolve} />,
    enabled: ({ eventValue, agentMetadata }) => {
      return eventValue === "Please provide a sender name which will appear in the email"
          && agentMetadata.agentName === 'email_agent'
          && agentMetadata.nodeName === 'email_node';
    }
  });

  return (
    <div
      className="flex flex-col items-center justify-center h-screen"
      data-test-id="mailer-container"
    >
      <div className="text-2xl" data-test-id="mailer-title">
        Email Q&A example
      </div>
      <div data-test-id="mailer-example">
        e.g. write an email to the CEO of OpenAI asking for a meeting
      </div>

      <CopilotPopup
        defaultOpen={true}
        clickOutsideToClose={false}
        data-test-id="mailer-popup"
      />

      <div
        data-test-id="email-success-message"
        className={messageState === "SEND" ? "" : "hidden"}
      >
        ✅ Sent email.
      </div>
      <div
        data-test-id="email-cancel-message"
        className={messageState === "CANCEL" ? "" : "hidden"}
      >
        ❌ Cancelled sending email.
      </div>
    </div>
  );
}

================
File: ui/app/page.tsx
================
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { Mailer } from "./Mailer";
import "@copilotkit/react-ui/styles.css";
import { ModelSelectorProvider, useModelSelectorContext } from "@/lib/model-selector-provider";
import { ModelSelector } from "@/components/ModelSelector";

export default function ModelSelectorWrapper() {
    return (
        <main className="flex flex-col items-center justify-between">
            <ModelSelectorProvider>
                <Home/>
                <ModelSelector/>
            </ModelSelectorProvider>
        </main>
    );
}

function Home() {
  const { lgcDeploymentUrl } = useModelSelectorContext();

  return (
      <CopilotKit runtimeUrl={`/api/copilotkit?lgcDeploymentUrl=${lgcDeploymentUrl ?? ''}`} agent="email_agent">
          <Mailer />
      </CopilotKit>
  );
}

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "default",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "app/globals.css",
    "baseColor": "slate",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils"
  }
}

================
File: ui/components/ModelSelector.tsx
================
"use client"

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModelSelectorContext } from "@/lib/model-selector-provider";

export function ModelSelector() {
  const { model, setModel } = useModelSelectorContext();

  return (
    <div className="fixed bottom-0 left-0 p-4 z-50">
      <Select value={model} onValueChange={v => setModel(v)}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Theme" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="openai">OpenAI</SelectItem>
          <SelectItem value="anthropic">Anthropic</SelectItem>
          <SelectItem value="google_genai">Google Generative AI</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}

================
File: ui/components/ui/select.tsx
================
"use client"

import * as React from "react"
import * as SelectPrimitive from "@radix-ui/react-select"
import { Check, ChevronDown, ChevronUp } from "lucide-react"

import { cn } from "@/lib/utils"

const Select = SelectPrimitive.Root

const SelectGroup = SelectPrimitive.Group

const SelectValue = SelectPrimitive.Value

const SelectTrigger = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Trigger
    ref={ref}
    className={cn(
      "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <ChevronDown className="h-4 w-4 opacity-50" />
    </SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>
))
SelectTrigger.displayName = SelectPrimitive.Trigger.displayName

const SelectScrollUpButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollUpButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollUpButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollUpButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronUp className="h-4 w-4" />
  </SelectPrimitive.ScrollUpButton>
))
SelectScrollUpButton.displayName = SelectPrimitive.ScrollUpButton.displayName

const SelectScrollDownButton = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.ScrollDownButton>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.ScrollDownButton>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.ScrollDownButton
    ref={ref}
    className={cn(
      "flex cursor-default items-center justify-center py-1",
      className
    )}
    {...props}
  >
    <ChevronDown className="h-4 w-4" />
  </SelectPrimitive.ScrollDownButton>
))
SelectScrollDownButton.displayName =
  SelectPrimitive.ScrollDownButton.displayName

const SelectContent = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
>(({ className, children, position = "popper", ...props }, ref) => (
  <SelectPrimitive.Portal>
    <SelectPrimitive.Content
      ref={ref}
      className={cn(
        "relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        position === "popper" &&
          "data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",
        className
      )}
      position={position}
      {...props}
    >
      <SelectScrollUpButton />
      <SelectPrimitive.Viewport
        className={cn(
          "p-1",
          position === "popper" &&
            "h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"
        )}
      >
        {children}
      </SelectPrimitive.Viewport>
      <SelectScrollDownButton />
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>
))
SelectContent.displayName = SelectPrimitive.Content.displayName

const SelectLabel = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Label>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Label>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Label
    ref={ref}
    className={cn("py-1.5 pl-8 pr-2 text-sm font-semibold", className)}
    {...props}
  />
))
SelectLabel.displayName = SelectPrimitive.Label.displayName

const SelectItem = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
>(({ className, children, ...props }, ref) => (
  <SelectPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <Check className="h-4 w-4" />
      </SelectPrimitive.ItemIndicator>
    </span>

    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
  </SelectPrimitive.Item>
))
SelectItem.displayName = SelectPrimitive.Item.displayName

const SelectSeparator = React.forwardRef<
  React.ElementRef<typeof SelectPrimitive.Separator>,
  React.ComponentPropsWithoutRef<typeof SelectPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <SelectPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 my-1 h-px bg-muted", className)}
    {...props}
  />
))
SelectSeparator.displayName = SelectPrimitive.Separator.displayName

export {
  Select,
  SelectGroup,
  SelectValue,
  SelectTrigger,
  SelectContent,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectScrollUpButton,
  SelectScrollDownButton,
}

================
File: ui/lib/model-selector-provider.tsx
================
"use client";

import React from "react";
import { createContext, useContext, useState, ReactNode } from "react";

type ModelSelectorContextType = {
  model: string;
  setModel: (model: string) => void;
  hidden: boolean;
  setHidden: (hidden: boolean) => void;
  lgcDeploymentUrl?: string | null;
};

const ModelSelectorContext = createContext<
  ModelSelectorContextType | undefined
>(undefined);

export const ModelSelectorProvider = ({
  children,
}: {
  children: ReactNode;
}) => {
  const model =
    globalThis.window === undefined
      ? "openai"
      : new URL(window.location.href).searchParams.get("coAgentsModel") ??
        "openai";
  const [hidden, setHidden] = useState<boolean>(false);

  const setModel = (model: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("coAgentsModel", model);
    window.location.href = url.toString();
  };

  const lgcDeploymentUrl = globalThis.window === undefined
      ? null
      : new URL(window.location.href).searchParams.get("lgcDeploymentUrl")

  return (
    <ModelSelectorContext.Provider
      value={{
        model,
        hidden,
        lgcDeploymentUrl,
        setModel,
        setHidden,
      }}
    >
      {children}
    </ModelSelectorContext.Provider>
  );
};

export const useModelSelectorContext = () => {
  const context = useContext(ModelSelectorContext);
  if (context === undefined) {
    throw new Error(
      "useModelSelectorContext must be used within a ModelSelectorProvider"
    );
  }
  return context;
};

================
File: ui/lib/utils.ts
================
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ai-researcher-demo",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "1.5.20",
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@copilotkit/runtime-client-gql": "1.5.20",
    "@copilotkit/shared": "1.5.20",
    "@radix-ui/react-accordion": "^1.2.0",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "framer-motion": "^11.3.31",
    "lucide-react": "^0.436.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "react-markdown": "^9.0.1",
    "tailwind-merge": "^2.5.2",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "19.0.1",
    "@types/react-dom": "19.0.2",
    "eslint": "^9.0.0",
    "eslint-config-next": "15.1.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
  },
  "pnpm": {
    "overrides": {
      "@types/react": "19.0.1",
      "@types/react-dom": "19.0.2"
    }
  }
}

================
File: ui/postcss.config.mjs
================
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;