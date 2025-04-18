# **LangGraph Tutorial Customer Support Example Directory:**

This file is a merged representation of the entire codebase, combined into a single document by Repomix.

================================================================
Directory Structure
================================================================
agent-py/.gitignore
agent-py/pyproject.toml
agent-py/tutorial_customer_support/agent.py
agent-py/tutorial_customer_support/demo.py
agent-py/tutorial_customer_support/seed.py
agent-py/tutorial_customer_support/tools.py
agent-py/tutorial_customer_support/utilities.py
README.md
ui/.eslintrc.json
ui/.gitignore
ui/next.config.ts
ui/package.json
ui/postcss.config.mjs
ui/public/file.svg
ui/public/globe.svg
ui/public/next.svg
ui/public/vercel.svg
ui/public/window.svg
ui/README.md
ui/src/app/api/copilotkit/route.ts
ui/src/app/globals.css
ui/src/app/layout.tsx
ui/src/app/page.tsx
ui/tailwind.config.ts
ui/tsconfig.json

================================================================
Files
================================================================

================
File: agent-py/.gitignore
================
venv/
__pycache__/
*.pyc
.env
.vercel
.langgraph_api
travel2.sqlite
travel2.backup.sqlite

================
File: agent-py/pyproject.toml
================
[tool.poetry]
name = "tutorial_customer_support"
version = "0.1.0"
description = "Customer Support Tutorial"
authors = ["Markus Ecker <markus.ecker@gmail.com>"]
license = "MIT"

[project]
name = "tutorial_customer_support"
version = "0.0.1"
dependencies = [
  "langgraph==0.2.52",
  "langchain_core",
  "langchain-google-genai",
  "langchain",
  "openai",
  "langchain-community",
  "langchain-anthropic",
  "tavily-python",
  "copilotkit==0.1.28a6",
  "python-dotenv",
  "uvicorn",
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
copilotkit = "0.1.39"
langgraph = "0.2.52"
langchain-core = "^0.3.12"
langchain-community = "^0.3.1"
langchain-anthropic = "0.2.3"
langchain = "0.3.4"
openai = "^1.52.1"
tavily-python = "^0.5.0"
python-dotenv = "^1.0.1"
uvicorn = "^0.31.0"
pandas = "^2.2.3"

[tool.poetry.scripts]
demo = "tutorial_customer_support.demo:main"
seed = "tutorial_customer_support.seed:main"

================
File: agent-py/tutorial_customer_support/agent.py
================
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.prompts import ChatPromptTemplate
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from copilotkit import CopilotKitState


from tutorial_customer_support.utilities import create_tool_node_with_fallback

from tutorial_customer_support.tools import (
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
)



class State(CopilotKitState):
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You could swap LLMs, though you will likely want to update the prompts when
# doing so!
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

part_1_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

================
File: agent-py/tutorial_customer_support/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv 
load_dotenv()

# pylint: disable=wrong-import-position
from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, LangGraphAgent
from tutorial_customer_support.agent import part_1_graph

app = FastAPI()
sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="customer_support_agent",
            description="Customer support agent.",
            graph=part_1_graph,
            config={
                "configurable": {
                    # The passenger_id is used in our flight tools to
                    # fetch the user's flight information
                    "passenger_id": "3442 587242",                
                }
            }
        ),
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
    uvicorn.run("tutorial_customer_support.demo:app", host="0.0.0.0", port=port)

================
File: agent-py/tutorial_customer_support/seed.py
================
import os
import shutil
import sqlite3

import pandas as pd
import requests

db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
local_file = "travel2.sqlite"
# The backup lets us restart for each tutorial section
backup_file = "travel2.backup.sqlite"
overwrite = False
if overwrite or not os.path.exists(local_file):
    response = requests.get(db_url)
    response.raise_for_status()  # Ensure the request was successful
    with open(local_file, "wb") as f:
        f.write(response.content)
    # Backup - we will use this to "reset" our DB in each section
    shutil.copy(local_file, backup_file)


# Convert the flights to present time for our tutorial
def update_dates(file):
    shutil.copy(backup_file, file)
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    return file


def main():
    db = update_dates(local_file)

================
File: agent-py/tutorial_customer_support/tools.py
================
import re
import requests
import numpy as np
import openai
from langchain_core.tools import tool

response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
db = "travel2.sqlite"


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes performing other 'write' events."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

import sqlite3
from datetime import date, datetime
from typing import Optional

import pytz
from langchain_core.runnables import RunnableConfig


@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments.

    Returns:
        A list of dictionaries where each dictionary contains the ticket details,
        associated flight details, and the seat assignments for each ticket belonging to the user.
    """
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = """
    SELECT 
        t.ticket_no, t.book_ref,
        f.flight_id, f.flight_no, f.departure_airport, f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
        bp.seat_no, tf.fare_conditions
    FROM 
        tickets t
        JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
        JOIN flights f ON tf.flight_id = f.flight_id
        JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE 
        t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM flights WHERE 1 = 1"
    params = []

    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)

    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)

    if start_time:
        query += " AND scheduled_departure >= ?"
        params.append(start_time)

    if end_time:
        query += " AND scheduled_departure <= ?"
        params.append(end_time)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results


@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT departure_airport, arrival_airport, scheduled_departure FROM flights WHERE flight_id = ?",
        (new_flight_id,),
    )
    new_flight = cursor.fetchone()
    if not new_flight:
        cursor.close()
        conn.close()
        return "Invalid new flight ID provided."
    column_names = [column[0] for column in cursor.description]
    new_flight_dict = dict(zip(column_names, new_flight))
    timezone = pytz.timezone("Etc/GMT-3")
    current_time = datetime.now(tz=timezone)
    departure_time = datetime.strptime(
        new_flight_dict["scheduled_departure"], "%Y-%m-%d %H:%M:%S.%f%z"
    )
    time_until = (departure_time - current_time).total_seconds()
    if time_until < (3 * 3600):
        return f"Not permitted to reschedule to a flight that is less than 3 hours from the current time. Selected flight is at {departure_time}."

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    current_flight = cursor.fetchone()
    if not current_flight:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT * FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    # In a real application, you'd likely add additional checks here to enforce business logic,
    # like "does the new departure airport match the current ticket", etc.
    # While it's best to try to be *proactive* in 'type-hinting' policies to the LLM
    # it's inevitably going to get things wrong, so you **also** need to ensure your
    # API enforces valid behavior
    cursor.execute(
        "UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?",
        (new_flight_id, ticket_no),
    )
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    configuration = config.get("configurable", {})
    passenger_id = configuration.get("passenger_id", None)
    if not passenger_id:
        raise ValueError("No passenger ID configured.")
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,)
    )
    existing_ticket = cursor.fetchone()
    if not existing_ticket:
        cursor.close()
        conn.close()
        return "No existing ticket found for the given ticket number."

    # Check the signed-in user actually has this ticket
    cursor.execute(
        "SELECT flight_id FROM tickets WHERE ticket_no = ? AND passenger_id = ?",
        (ticket_no, passenger_id),
    )
    current_ticket = cursor.fetchone()
    if not current_ticket:
        cursor.close()
        conn.close()
        return f"Current signed-in passenger with ID {passenger_id} not the owner of ticket {ticket_no}"

    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()

    cursor.close()
    conn.close()
    return "Ticket successfully cancelled."

from datetime import date, datetime
from typing import Optional, Union


@tool
def search_car_rentals(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for car rentals based on location, name, price tier, start date, and end date.

    Args:
        location (Optional[str]): The location of the car rental. Defaults to None.
        name (Optional[str]): The name of the car rental company. Defaults to None.
        price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
        start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

    Returns:
        list[dict]: A list of car rental dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM car_rentals WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For our tutorial, we will let you match on any dates and price tier.
    # (since our toy dataset doesn't have much data)
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_car_rental(rental_id: int) -> str:
    """
    Book a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to book.

    Returns:
        str: A message indicating whether the car rental was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully booked."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."


@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a car rental's start and end dates by its ID.

    Args:
        rental_id (int): The ID of the car rental to update.
        start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
        end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

    Returns:
        str: A message indicating whether the car rental was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date, rental_id),
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully updated."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."


@tool
def cancel_car_rental(rental_id: int) -> str:
    """
    Cancel a car rental by its ID.

    Args:
        rental_id (int): The ID of the car rental to cancel.

    Returns:
        str: A message indicating whether the car rental was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Car rental {rental_id} successfully cancelled."
    else:
        conn.close()
        return f"No car rental found with ID {rental_id}."
  
@tool
def search_hotels(
    location: Optional[str] = None,
    name: Optional[str] = None,
    price_tier: Optional[str] = None,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date.

    Args:
        location (Optional[str]): The location of the hotel. Defaults to None.
        name (Optional[str]): The name of the hotel. Defaults to None.
        price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
        checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

    Returns:
        list[dict]: A list of hotel dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM hotels WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    # For the sake of this tutorial, we will let you match on any dates and price tier.
    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_hotel(hotel_id: int) -> str:
    """
    Book a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to book.

    Returns:
        str: A message indicating whether the hotel was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully booked."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    Update a hotel's check-in and check-out dates by its ID.

    Args:
        hotel_id (int): The ID of the hotel to update.
        checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
        checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

    Returns:
        str: A message indicating whether the hotel was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    if checkin_date:
        cursor.execute(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
        )
    if checkout_date:
        cursor.execute(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date, hotel_id),
        )

    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully updated."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."


@tool
def cancel_hotel(hotel_id: int) -> str:
    """
    Cancel a hotel by its ID.

    Args:
        hotel_id (int): The ID of the hotel to cancel.

    Returns:
        str: A message indicating whether the hotel was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        conn.close()
        return f"No hotel found with ID {hotel_id}."

@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()

    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully booked."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id),
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully updated."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"Trip recommendation {recommendation_id} successfully cancelled."
    else:
        conn.close()
        return f"No trip recommendation found with ID {recommendation_id}."

================
File: agent-py/tutorial_customer_support/utilities.py
================
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

================
File: README.md
================
# LangGraph customer support agent

## Overview

This is a customer support agent example that uses LangGraph to handle customer support requests.

See: https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/

## Setup

In `agent-py`, run:

```
poetry install
poetry run seed
```

## Run

In `agent-py`:

```
poetry run demo
```

In `ui`:

```
pnpm install
pnpm dev
```

================
File: ui/.eslintrc.json
================
{
  "extends": ["next/core-web-vitals", "next/typescript"]
}

================
File: ui/.gitignore
================
# See https://help.github.com/articles/ignoring-files/ for more about ignoring files.

# dependencies
/node_modules
/.pnp
.pnp.*
.yarn/*
!.yarn/patches
!.yarn/plugins
!.yarn/releases
!.yarn/versions

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

# env files (can opt-in for committing if needed)
.env*

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

================
File: ui/next.config.ts
================
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ui",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@copilotkit/react-core": "^1.5.20",
    "@copilotkit/react-ui": "^1.5.20",
    "@copilotkit/runtime": "^1.5.20",
    "next": "15.0.3",
    "openai": "^4.85.1",
    "react": "19.0.0-rc-66855b96-20241106",
    "react-dom": "19.0.0-rc-66855b96-20241106"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "eslint": "^8",
    "eslint-config-next": "15.0.3",
    "postcss": "^8",
    "tailwindcss": "^3.4.1",
    "typescript": "^5"
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
File: ui/public/file.svg
================
<svg fill="none" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M14.5 13.5V5.41a1 1 0 0 0-.3-.7L9.8.29A1 1 0 0 0 9.08 0H1.5v13.5A2.5 2.5 0 0 0 4 16h8a2.5 2.5 0 0 0 2.5-2.5m-1.5 0v-7H8v-5H3v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1M9.5 5V2.12L12.38 5zM5.13 5h-.62v1.25h2.12V5zm-.62 3h7.12v1.25H4.5zm.62 3h-.62v1.25h7.12V11z" clip-rule="evenodd" fill="#666" fill-rule="evenodd"/></svg>

================
File: ui/public/globe.svg
================
<svg fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><g clip-path="url(#a)"><path fill-rule="evenodd" clip-rule="evenodd" d="M10.27 14.1a6.5 6.5 0 0 0 3.67-3.45q-1.24.21-2.7.34-.31 1.83-.97 3.1M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m.48-1.52a7 7 0 0 1-.96 0H7.5a4 4 0 0 1-.84-1.32q-.38-.89-.63-2.08a40 40 0 0 0 3.92 0q-.25 1.2-.63 2.08a4 4 0 0 1-.84 1.31zm2.94-4.76q1.66-.15 2.95-.43a7 7 0 0 0 0-2.58q-1.3-.27-2.95-.43a18 18 0 0 1 0 3.44m-1.27-3.54a17 17 0 0 1 0 3.64 39 39 0 0 1-4.3 0 17 17 0 0 1 0-3.64 39 39 0 0 1 4.3 0m1.1-1.17q1.45.13 2.69.34a6.5 6.5 0 0 0-3.67-3.44q.65 1.26.98 3.1M8.48 1.5l.01.02q.41.37.84 1.31.38.89.63 2.08a40 40 0 0 0-3.92 0q.25-1.2.63-2.08a4 4 0 0 1 .85-1.32 7 7 0 0 1 .96 0m-2.75.4a6.5 6.5 0 0 0-3.67 3.44 29 29 0 0 1 2.7-.34q.31-1.83.97-3.1M4.58 6.28q-1.66.16-2.95.43a7 7 0 0 0 0 2.58q1.3.27 2.95.43a18 18 0 0 1 0-3.44m.17 4.71q-1.45-.12-2.69-.34a6.5 6.5 0 0 0 3.67 3.44q-.65-1.27-.98-3.1" fill="#666"/></g><defs><clipPath id="a"><path fill="#fff" d="M0 0h16v16H0z"/></clipPath></defs></svg>

================
File: ui/public/next.svg
================
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 394 80"><path fill="#000" d="M262 0h68.5v12.7h-27.2v66.6h-13.6V12.7H262V0ZM149 0v12.7H94v20.4h44.3v12.6H94v21h55v12.6H80.5V0h68.7zm34.3 0h-17.8l63.8 79.4h17.9l-32-39.7 32-39.6h-17.9l-23 28.6-23-28.6zm18.3 56.7-9-11-27.1 33.7h17.8l18.3-22.7z"/><path fill="#000" d="M81 79.3 17 0H0v79.3h13.6V17l50.2 62.3H81Zm252.6-.4c-1 0-1.8-.4-2.5-1s-1.1-1.6-1.1-2.6.3-1.8 1-2.5 1.6-1 2.6-1 1.8.3 2.5 1a3.4 3.4 0 0 1 .6 4.3 3.7 3.7 0 0 1-3 1.8zm23.2-33.5h6v23.3c0 2.1-.4 4-1.3 5.5a9.1 9.1 0 0 1-3.8 3.5c-1.6.8-3.5 1.3-5.7 1.3-2 0-3.7-.4-5.3-1s-2.8-1.8-3.7-3.2c-.9-1.3-1.4-3-1.4-5h6c.1.8.3 1.6.7 2.2s1 1.2 1.6 1.5c.7.4 1.5.5 2.4.5 1 0 1.8-.2 2.4-.6a4 4 0 0 0 1.6-1.8c.3-.8.5-1.8.5-3V45.5zm30.9 9.1a4.4 4.4 0 0 0-2-3.3 7.5 7.5 0 0 0-4.3-1.1c-1.3 0-2.4.2-3.3.5-.9.4-1.6 1-2 1.6a3.5 3.5 0 0 0-.3 4c.3.5.7.9 1.3 1.2l1.8 1 2 .5 3.2.8c1.3.3 2.5.7 3.7 1.2a13 13 0 0 1 3.2 1.8 8.1 8.1 0 0 1 3 6.5c0 2-.5 3.7-1.5 5.1a10 10 0 0 1-4.4 3.5c-1.8.8-4.1 1.2-6.8 1.2-2.6 0-4.9-.4-6.8-1.2-2-.8-3.4-2-4.5-3.5a10 10 0 0 1-1.7-5.6h6a5 5 0 0 0 3.5 4.6c1 .4 2.2.6 3.4.6 1.3 0 2.5-.2 3.5-.6 1-.4 1.8-1 2.4-1.7a4 4 0 0 0 .8-2.4c0-.9-.2-1.6-.7-2.2a11 11 0 0 0-2.1-1.4l-3.2-1-3.8-1c-2.8-.7-5-1.7-6.6-3.2a7.2 7.2 0 0 1-2.4-5.7 8 8 0 0 1 1.7-5 10 10 0 0 1 4.3-3.5c2-.8 4-1.2 6.4-1.2 2.3 0 4.4.4 6.2 1.2 1.8.8 3.2 2 4.3 3.4 1 1.4 1.5 3 1.5 5h-5.8z"/></svg>

================
File: ui/public/vercel.svg
================
<svg fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1155 1000"><path d="m577.3 0 577.4 1000H0z" fill="#fff"/></svg>

================
File: ui/public/window.svg
================
<svg fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.5 2.5h13v10a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1zM0 1h16v11.5a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 0 12.5zm3.75 4.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M7 4.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0m1.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5" fill="#666"/></svg>

================
File: ui/README.md
================
This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

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

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

================
File: ui/src/app/api/copilotkit/route.ts
================
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";
import { NextRequest } from "next/server";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const serviceAdapter = new OpenAIAdapter({ openai });
const runtime = new CopilotRuntime({
  remoteActions: [
    {
      url: "http://localhost:8000/copilotkit",
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
File: ui/src/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --background: #ffffff;
  --foreground: #171717;
}

@media (prefers-color-scheme: dark) {
  :root {
    --background: #0a0a0a;
    --foreground: #ededed;
  }
}

body {
  color: var(--foreground);
  background: var(--background);
  font-family: Arial, Helvetica, sans-serif;
}

================
File: ui/src/app/layout.tsx
================
import { ReactNode } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import "./globals.css";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Make sure to use the URL you configured in the previous step  */}
        <CopilotKit runtimeUrl="/api/copilotkit" agent="customer_support_agent">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}

================
File: ui/src/app/page.tsx
================
"use client";
import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import { useState } from "react";

export default function YourApp() {
  return (
    <>
      <CopilotPopup
        instructions={
          "You are assisting the user as best as you can. Answer in the best way possible given the data you have."
        }
        defaultOpen={true}
        labels={{
          title: "Popup Assistant",
          initial: "Need any help?",
        }}
      />
    </>
  );
}

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
    },
  },
  plugins: [],
} satisfies Config;

================
File: ui/tsconfig.json
================
{
  "compilerOptions": {
    "target": "ES2017",
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
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}



================================================================
End of Codebase
================================================================


# coagents-research-canvas Example:

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
agent/.vscode/settings.json
agent/langgraph.json
agent/pyproject.toml
agent/research_canvas/agent.py
agent/research_canvas/chat.py
agent/research_canvas/delete.py
agent/research_canvas/demo.py
agent/research_canvas/download.py
agent/research_canvas/model.py
agent/research_canvas/search.py
agent/research_canvas/state.py
dockerize.sh
readme.md
ui/.eslintrc.json
ui/.gitignore
ui/components.json
ui/next.config.mjs
ui/package.json
ui/postcss.config.mjs
ui/README.md
ui/src/app/api/copilotkit/route.ts
ui/src/app/globals.css
ui/src/app/layout.tsx
ui/src/app/Main.tsx
ui/src/app/page.tsx
ui/src/components/AddResourceDialog.tsx
ui/src/components/EditResourceDialog.tsx
ui/src/components/ModelSelector.tsx
ui/src/components/Progress.tsx
ui/src/components/ResearchCanvas.tsx
ui/src/components/Resources.tsx
ui/src/components/ui/button.tsx
ui/src/components/ui/card.tsx
ui/src/components/ui/dialog.tsx
ui/src/components/ui/input.tsx
ui/src/components/ui/select.tsx
ui/src/components/ui/textarea.tsx
ui/src/lib/model-selector-provider.tsx
ui/src/lib/types.ts
ui/src/lib/utils.ts
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
.langgraph_api

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
    "aiohttp",
    "dotenv",
    "khtml"
  ]
}

================
File: agent/.vscode/settings.json
================
{
  "python.analysis.typeCheckingMode": "basic"
}

================
File: agent/langgraph.json
================
{
  "python_version": "3.12",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "research_agent": "./research_canvas/agent.py:graph"
  },
  "env": ".env"
}

================
File: agent/pyproject.toml
================
[tool.poetry]
name = "research_canvas"
version = "0.1.0"
description = "Research Canvas"
authors = ["Markus Ecker <markus.ecker@gmail.com>"]
license = "MIT"

[project]
name = "research_canvas"
version = "0.0.1"
dependencies = [
  "langgraph",
  "langchain_core",
  "langchain-google-genai",
  "langchain_openai",
  "langchain",
  "openai",
  "langchain-community",
  "langchain-anthropic",
  "tavily-python",
  "python-dotenv",
  "uvicorn",
  "html2text",
  "copilotkit==0.1.34",
  "aiohttp",
  "aiosqlite",
  "requests"
]

[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[tool.poetry.dependencies]
python = "^3.12"
copilotkit = "0.1.39"
langchain-openai = "0.2.3"
langchain-community = "^0.3.1"
langchain-anthropic = "0.3.1"
langchain-google-genai = "2.0.5"
langchain = "0.3.4"
openai = "^1.52.1"
tavily-python = "^0.5.0"
python-dotenv = "^1.0.1"
uvicorn = "^0.31.0"
requests = "^2.32.3"
html2text = "^2024.2.26"
langchain-core = "^0.3.25"
langgraph-cli = {extras = ["inmem"], version = "^0.1.64"}
langgraph-checkpoint-sqlite = "^2.0.1"
aiosqlite = "^0.20.0"

[tool.poetry.scripts]
demo = "research_canvas.demo:main"

================
File: agent/research_canvas/agent.py
================
"""
This is the main entry point for the AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import
import json
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from research_canvas.state import AgentState
from research_canvas.download import download_node
from research_canvas.chat import chat_node
from research_canvas.search import search_node
from research_canvas.delete import delete_node, perform_delete_node

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("download", download_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("search_node", search_node)
workflow.add_node("delete_node", delete_node)
workflow.add_node("perform_delete_node", perform_delete_node)


memory = MemorySaver()
workflow.set_entry_point("download")
workflow.add_edge("download", "chat_node")
workflow.add_edge("delete_node", "perform_delete_node")
workflow.add_edge("perform_delete_node", "chat_node")
workflow.add_edge("search_node", "download")
graph = workflow.compile(checkpointer=memory, interrupt_after=["delete_node"])

================
File: agent/research_canvas/chat.py
================
"""Chat Node"""

from typing import List, cast, Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langgraph.types import Command
from copilotkit.langgraph import copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model
from research_canvas.download import get_resource


@tool
def Search(queries: List[str]): # pylint: disable=invalid-name,unused-argument
    """A list of one or more search queries to find good resources to support the research."""

@tool
def WriteReport(report: str): # pylint: disable=invalid-name,unused-argument
    """Write the research report."""

@tool
def WriteResearchQuestion(research_question: str): # pylint: disable=invalid-name,unused-argument
    """Write the research question."""

@tool
def DeleteResources(urls: List[str]): # pylint: disable=invalid-name,unused-argument
    """Delete the URLs from the resources."""


async def chat_node(state: AgentState, config: RunnableConfig) -> \
    Command[Literal["search_node", "chat_node", "delete_node", "__end__"]]:
    """
    Chat Node
    """

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "report",
            "tool": "WriteReport",
            "tool_argument": "report",
        }, {
            "state_key": "research_question",
            "tool": "WriteResearchQuestion",
            "tool_argument": "research_question",
        }],
    )

    state["resources"] = state.get("resources", [])
    research_question = state.get("research_question", "")
    report = state.get("report", "")

    resources = []

    for resource in state["resources"]:
        content = get_resource(resource["url"])
        if content == "ERROR":
            continue
        resources.append({
            **resource,
            "content": content
        })

    model = get_model(state)
    # Prepare the kwargs for the ainvoke method
    ainvoke_kwargs = {}
    if model.__class__.__name__ in ["ChatOpenAI"]:
        ainvoke_kwargs["parallel_tool_calls"] = False

    response = await model.bind_tools(
        [
            Search,
            WriteReport,
            WriteResearchQuestion,
            DeleteResources,
        ],
        **ainvoke_kwargs  # Pass the kwargs conditionally
    ).ainvoke([
        SystemMessage(
            content=f"""
            You are a research assistant. You help the user with writing a research report.
            Do not recite the resources, instead use them to answer the user's question.
            You should use the search tool to get resources before answering the user's question.
            If you finished writing the report, ask the user proactively for next steps, changes etc, make it engaging.
            To write the report, you should use the WriteReport tool. Never EVER respond with the report, only use the tool.
            If a research question is provided, YOU MUST NOT ASK FOR IT AGAIN.

            This is the research question:
            {research_question}

            This is the research report:
            {report}

            Here are the resources that you have available:
            {resources}
            """
        ),
        *state["messages"],
    ], config)

    ai_message = cast(AIMessage, response)

    if ai_message.tool_calls:
        if ai_message.tool_calls[0]["name"] == "WriteReport":
            report = ai_message.tool_calls[0]["args"].get("report", "")
            return Command(
                goto="chat_node",
                update={
                    "report": report,
                    "messages": [ai_message, ToolMessage(
                    tool_call_id=ai_message.tool_calls[0]["id"],
                    content="Report written."
                    )]
                }
            )
        if ai_message.tool_calls[0]["name"] == "WriteResearchQuestion":
            return Command(
                goto="chat_node",
                update={
                    "research_question": ai_message.tool_calls[0]["args"]["research_question"],
                    "messages": [ai_message, ToolMessage(
                        tool_call_id=ai_message.tool_calls[0]["id"],
                        content="Research question written."
                    )]
                }
            )
       
    goto = "__end__"
    if ai_message.tool_calls and ai_message.tool_calls[0]["name"] == "Search":
        goto = "search_node"
    elif ai_message.tool_calls and ai_message.tool_calls[0]["name"] == "DeleteResources":
        goto = "delete_node"


    return Command(
        goto=goto,
        update={
            "messages": response
        }
    )

================
File: agent/research_canvas/delete.py
================
"""Delete Resources"""

import json
from typing import cast
from research_canvas.state import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, AIMessage
async def delete_node(state: AgentState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Delete Node
    """
    return state

async def perform_delete_node(state: AgentState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Perform Delete Node
    """
    ai_message = cast(AIMessage, state["messages"][-2])
    tool_message = cast(ToolMessage, state["messages"][-1])
    if tool_message.content == "YES":
        if ai_message.tool_calls:
            urls = ai_message.tool_calls[0]["args"]["urls"]
        else:
            parsed_tool_call = json.loads(ai_message.additional_kwargs["function_call"]["arguments"])
            urls = parsed_tool_call["urls"]

        state["resources"] = [
            resource for resource in state["resources"] if resource["url"] not in urls
        ]

    return state

================
File: agent/research_canvas/demo.py
================
"""Demo"""

import os
from dotenv import load_dotenv
load_dotenv()

# pylint: disable=wrong-import-position
from fastapi import FastAPI
import uvicorn
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from research_canvas.agent import graph

# from contextlib import asynccontextmanager
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# @asynccontextmanager
# async def lifespan(fastapi_app: FastAPI):
#     """Lifespan for the FastAPI app."""
#     async with AsyncSqliteSaver.from_conn_string(
#         ":memory:"
#     ) as checkpointer:
#         # Create an async graph
#         graph = workflow.compile(checkpointer=checkpointer)

#         # Create SDK with the graph
#         sdk = CopilotKitRemoteEndpoint(
#             agents=[
#                 LangGraphAgent(
#                     name="research_agent",
#                     description="Research agent.",
#                     graph=graph,
#                 ),
#                 LangGraphAgent(
#                     name="research_agent_google_genai",
#                     description="Research agent.",
#                     graph=graph
#                 ),
#             ],
#         )

#         # Add the CopilotKit FastAPI endpoint
#         add_fastapi_endpoint(fastapi_app, sdk, "/copilotkit")
#         yield

# app = FastAPI(lifespan=lifespan)


app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="research_agent",
            description="Research agent.",
            graph=graph,
        ),
        LangGraphAgent(
            name="research_agent_google_genai",
            description="Research agent.",
            graph=graph
        ),
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "research_canvas.demo:app",
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
File: agent/research_canvas/download.py
================
"""
This module contains the implementation of the download_node function.
"""

import aiohttp
import html2text
from copilotkit.langgraph import copilotkit_emit_state
from langchain_core.runnables import RunnableConfig
from research_canvas.state import AgentState

_RESOURCE_CACHE = {}

def get_resource(url: str):
    """
    Get a resource from the cache.
    """
    return _RESOURCE_CACHE.get(url, "")


_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" # pylint: disable=line-too-long

async def _download_resource(url: str):
    """
    Download a resource from the internet asynchronously.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                html_content = await response.text()
                markdown_content = html2text.html2text(html_content)
                _RESOURCE_CACHE[url] = markdown_content
                return markdown_content
    except Exception as e: # pylint: disable=broad-except
        _RESOURCE_CACHE[url] = "ERROR"
        return f"Error downloading resource: {e}"

async def download_node(state: AgentState, config: RunnableConfig):
    """
    Download resources from the internet.
    """
    state["resources"] = state.get("resources", [])
    state["logs"] = state.get("logs", [])
    resources_to_download = []

    logs_offset = len(state["logs"])

    # Find resources that are not downloaded
    for resource in state["resources"]:
        if not get_resource(resource["url"]):
            resources_to_download.append(resource)
            state["logs"].append({
                "message": f"Downloading {resource['url']}",
                "done": False
            })

    # Emit the state to let the UI update
    await copilotkit_emit_state(config, state)

    # Download the resources
    for i, resource in enumerate(resources_to_download):
        await _download_resource(resource["url"])
        state["logs"][logs_offset + i]["done"] = True

        # update UI
        await copilotkit_emit_state(config, state)

    return state

================
File: agent/research_canvas/model.py
================
"""
This module provides a function to get a model based on the configuration.
"""
import os
from typing import cast, Any
from langchain_core.language_models.chat_models import BaseChatModel
from research_canvas.state import AgentState

def get_model(state: AgentState) -> BaseChatModel:
    """
    Get a model based on the environment variable.
    """

    state_model = state.get("model")
    model = os.getenv("MODEL", state_model)

    print(f"Using model: {model}")

    if model == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model="gpt-4o-mini")
    if model == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            temperature=0,
            model_name="claude-3-5-sonnet-20240620",
            timeout=None,
            stop=None
        )
    if model == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            temperature=0,
            model="gemini-1.5-pro",
            api_key=cast(Any, os.getenv("GOOGLE_API_KEY")) or None
        )

    raise ValueError("Invalid model specified")

================
File: agent/research_canvas/search.py
================
"""
The search node is responsible for searching the internet for information.
"""

import os
from typing import cast, List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain.tools import tool
from tavily import TavilyClient
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from research_canvas.state import AgentState
from research_canvas.model import get_model

class ResourceInput(BaseModel):
    """A resource with a short description"""
    url: str = Field(description="The URL of the resource")
    title: str = Field(description="The title of the resource")
    description: str = Field(description="A short description of the resource")

@tool
def ExtractResources(resources: List[ResourceInput]): # pylint: disable=invalid-name,unused-argument
    """Extract the 3-5 most relevant resources from a search result."""

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def search_node(state: AgentState, config: RunnableConfig):
    """
    The search node is responsible for searching the internet for resources.
    """

    ai_message = cast(AIMessage, state["messages"][-1])

    state["resources"] = state.get("resources", [])
    state["logs"] = state.get("logs", [])
    queries = ai_message.tool_calls[0]["args"]["queries"]

    for query in queries:
        state["logs"].append({
            "message": f"Search for {query}",
            "done": False
        })

    await copilotkit_emit_state(config, state)

    search_results = []

    for i, query in enumerate(queries):
        response = tavily_client.search(query)
        search_results.append(response)
        state["logs"][i]["done"] = True
        await copilotkit_emit_state(config, state)

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "resources",
            "tool": "ExtractResources",
            "tool_argument": "resources",
        }],
    )

    model = get_model(state)
    ainvoke_kwargs = {}
    if model.__class__.__name__ in ["ChatOpenAI"]:
        ainvoke_kwargs["parallel_tool_calls"] = False

    # figure out which resources to use
    response = await model.bind_tools(
        [ExtractResources],
        tool_choice="ExtractResources",
        **ainvoke_kwargs
    ).ainvoke([
        SystemMessage(
            content="""
            You need to extract the 3-5 most relevant resources from the following search results.
            """
        ),
        *state["messages"],
        ToolMessage(
        tool_call_id=ai_message.tool_calls[0]["id"],
        content=f"Performed search: {search_results}"
    )
    ], config)

    state["logs"] = []
    await copilotkit_emit_state(config, state)

    ai_message_response = cast(AIMessage, response)
    resources = ai_message_response.tool_calls[0]["args"]["resources"]

    state["resources"].extend(resources)

    state["messages"].append(ToolMessage(
        tool_call_id=ai_message.tool_calls[0]["id"],
        content=f"Added the following resources: {resources}"
    ))

    return state

================
File: agent/research_canvas/state.py
================
"""
This is the state definition for the AI.
It defines the state of the agent and the state of the conversation.
"""

from typing import List, TypedDict
from langgraph.graph import MessagesState

class Resource(TypedDict):
    """
    Represents a resource. Give it a good title and a short description.
    """
    url: str
    title: str
    description: str

class Log(TypedDict):
    """
    Represents a log of an action performed by the agent.
    """
    message: str
    done: bool

class AgentState(MessagesState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """
    model: str
    research_question: str
    report: str
    resources: List[Resource]
    logs: List[Log]

================
File: dockerize.sh
================
docker build -t ${IMAGE_TAG} . --platform=linux/amd64 -f ./examples/Dockerfile.ui --build-arg APP_DIR=${APP_DIR} --push

================
File: readme.md
================
# CoAgents Research Canvas Example

This example demonstrates a research canvas UI.

Live demo: https://examples-coagents-research-canvas-ui.vercel.app/

Tutorial Video:

[![IMAGE ALT TEXT](http://img.youtube.com/vi/0b6BVqPwqA0/0.jpg)](http://www.youtube.com/watch?v=0b6BVqPwqA0 "Build Agent-Native Apps with LangGraph & CoAgents (tutorial)")



---

## Running the Agent

**These instructions assume you are in the `coagents-research-canvas/` directory**

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

Make sure to create the `.env` mentioned above first!

# Troubleshooting

A few things to try if you are running into trouble:

1. Make sure there is no other local application server running on the 8000 port.
2. Under `/agent/research_canvas/demo.py`, change `0.0.0.0` to `127.0.0.1` or to `localhost`

================
File: ui/.eslintrc.json
================
{
  "extends": ["next/core-web-vitals", "next/typescript"],
  "rules": {
    "@typescript-eslint/no-explicit-any": "off",
    "@typescript-eslint/no-unused-vars": "off"
  }
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

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts

.env

================
File: ui/components.json
================
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "tailwind.config.ts",
    "css": "src/app/globals.css",
    "baseColor": "neutral",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib",
    "hooks": "@/hooks"
  }
}

================
File: ui/next.config.mjs
================
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone"
};

export default nextConfig;

================
File: ui/package.json
================
{
  "name": "ui",
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
    "@copilotkit/react-ui": "1.5.20",
    "@copilotkit/runtime": "1.5.20",
    "@radix-ui/react-dialog": "^1.1.2",
    "@radix-ui/react-icons": "^1.3.2",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-slot": "^1.1.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "lucide-react": "^0.451.0",
    "next": "15.1.0",
    "openai": "^4.85.1",
    "react": "19.0.0",
    "react-dom": "19.0.0",
    "tailwind-merge": "^2.5.3",
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
This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

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

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

================
File: ui/src/app/api/copilotkit/route.ts
================
import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
  langGraphPlatformEndpoint, copilotKitEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";
import { NextRequest } from "next/server";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const llmAdapter = new OpenAIAdapter({ openai } as any);
const langsmithApiKey = process.env.LANGSMITH_API_KEY as string

export const POST = async (req: NextRequest) => {
  const searchParams = req.nextUrl.searchParams
  const deploymentUrl = searchParams.get('lgcDeploymentUrl') || process.env.LGC_DEPLOYMENT_URL;

  const remoteEndpoint = deploymentUrl ? langGraphPlatformEndpoint({
    deploymentUrl,
    langsmithApiKey,
    agents: [
      {
        name: "research_agent",
        description: "Research agent",
      },
      {
        name: "research_agent_google_genai",
        description: "Research agent",
        assistantId: "9dc0ca3b-1aa6-547d-93f0-e21597d2011c",
      },
    ],
  }) : copilotKitEndpoint({
    url: process.env.REMOTE_ACTION_URL || "http://localhost:8000/copilotkit",
  })

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
File: ui/src/app/globals.css
================
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: Arial, Helvetica, sans-serif;
}

@layer utilities {
  .text-balance {
    text-wrap: balance;
  }
}

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 0 0% 3.9%;
    --card: 0 0% 100%;
    --card-foreground: 0 0% 3.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 0 0% 3.9%;
    --primary: 0 0% 9%;
    --primary-foreground: 0 0% 98%;
    --secondary: 0 0% 96.1%;
    --secondary-foreground: 0 0% 9%;
    --muted: 0 0% 96.1%;
    --muted-foreground: 0 0% 45.1%;
    --accent: 0 0% 96.1%;
    --accent-foreground: 0 0% 9%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 0 0% 98%;
    --border: 0 0% 89.8%;
    --input: 0 0% 89.8%;
    --ring: 0 0% 3.9%;
    --chart-1: 12 76% 61%;
    --chart-2: 173 58% 39%;
    --chart-3: 197 37% 24%;
    --chart-4: 43 74% 66%;
    --chart-5: 27 87% 67%;
    --radius: 0.5rem;
  }
  .dark {
    --background: 0 0% 3.9%;
    --foreground: 0 0% 98%;
    --card: 0 0% 3.9%;
    --card-foreground: 0 0% 98%;
    --popover: 0 0% 3.9%;
    --popover-foreground: 0 0% 98%;
    --primary: 0 0% 98%;
    --primary-foreground: 0 0% 9%;
    --secondary: 0 0% 14.9%;
    --secondary-foreground: 0 0% 98%;
    --muted: 0 0% 14.9%;
    --muted-foreground: 0 0% 63.9%;
    --accent: 0 0% 14.9%;
    --accent-foreground: 0 0% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 0 0% 98%;
    --border: 0 0% 14.9%;
    --input: 0 0% 14.9%;
    --ring: 0 0% 83.1%;
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

.copilotKitMessage.copilotKitUserMessage {
  border-radius: 0.5rem 0.5rem 0 0.5rem;
}

.copilotKitMessage.copilotKitAssistantMessage {
  border-radius: 0.5rem 0.5rem 0.5rem 0;
}

.copilotKitMessage svg {
  color: white !important;
}

.copilotKitChat {
  background-color: #e0e9fd;
}

.copilotKitResponseButton {
  background-color: transparent;
  color: var(--copilot-kit-secondary-color);
  border: 0px;
}

.copilotKitInput {
  border-top: none;
  background-color: white;
  margin-bottom: 20px;
  margin-left: 30px;
  margin-right: 30px;
  border-radius: 10px;
}

.copilotKitInput > textarea {
  background-color: white;
  color: black;
}

.copilotKitInput > .copilotKitInputControls > button:not([disabled]) {
  color: var(--copilot-kit-secondary-color);
}

================
File: ui/src/app/layout.tsx
================
import type { Metadata } from "next";
import localFont from "next/font/local";
import "@copilotkit/react-ui/styles.css";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

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
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

================
File: ui/src/app/Main.tsx
================
import { ResearchCanvas } from "@/components/ResearchCanvas";
import { useModelSelectorContext } from "@/lib/model-selector-provider";
import { AgentState } from "@/lib/types";
import { useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotChatSuggestions } from "@copilotkit/react-ui";

export default function Main() {
  const { model, agent } = useModelSelectorContext();
  const { state, setState } = useCoAgent<AgentState>({
    name: agent,
    initialState: {
      model,
      research_question: "",
      resources: [],
      report: "",
      logs: [],
    },
  });

  useCopilotChatSuggestions({
    instructions: "Lifespan of penguins",
  });

  return (
    <>
      <h1 className="flex h-[60px] bg-[#0E103D] text-white items-center px-10 text-2xl font-medium">
        Research Helper
      </h1>

      <div
        className="flex flex-1 border"
        style={{ height: "calc(100vh - 60px)" }}
      >
        <div className="flex-1 overflow-hidden">
          <ResearchCanvas />
        </div>
        <div
          className="w-[500px] h-full flex-shrink-0"
          style={
            {
              "--copilot-kit-background-color": "#E0E9FD",
              "--copilot-kit-secondary-color": "#6766FC",
              "--copilot-kit-secondary-contrast-color": "#FFFFFF",
              "--copilot-kit-primary-color": "#FFFFFF",
              "--copilot-kit-contrast-color": "#000000",
            } as any
          }
        >
          <CopilotChat
            className="h-full"
            onSubmitMessage={async (message) => {
              // clear the logs before starting the new research
              setState({ ...state, logs: [] });
              await new Promise((resolve) => setTimeout(resolve, 30));
            }}
            labels={{
              initial: "Hi! How can I assist you with your research today?",
            }}
          />
        </div>
      </div>
    </>
  );
}

================
File: ui/src/app/page.tsx
================
"use client";

import { CopilotKit } from "@copilotkit/react-core";
import Main from "./Main";
import {
  ModelSelectorProvider,
  useModelSelectorContext,
} from "@/lib/model-selector-provider";
import { ModelSelector } from "@/components/ModelSelector";

export default function ModelSelectorWrapper() {
  return (
    <ModelSelectorProvider>
      <Home />
      <ModelSelector />
    </ModelSelectorProvider>
  );
}

function Home() {
  const { agent, lgcDeploymentUrl } = useModelSelectorContext();

  return (
    <CopilotKit
      runtimeUrl={`/api/copilotkit?lgcDeploymentUrl=${lgcDeploymentUrl ?? ""}`}
      showDevConsole={false}
      agent={agent}
    >
      <Main />
    </CopilotKit>
  );
}

================
File: ui/src/components/AddResourceDialog.tsx
================
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { PlusCircle, Plus } from "lucide-react";
import { Resource } from "@/lib/types";

type AddResourceDialogProps = {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
  newResource: Resource;
  setNewResource: (resource: Resource) => void;
  addResource: () => void;
};

export function AddResourceDialog({
  isOpen,
  onOpenChange,
  newResource,
  setNewResource,
  addResource,
}: AddResourceDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="link"
          size="sm"
          className="text-sm font-bold text-[#6766FC]"
        >
          Add Resource <PlusCircle className="w-6 h-6 ml-2" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Add New Resource</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <label htmlFor="new-url" className="text-sm font-bold">
            Resource URL
          </label>
          <Input
            id="new-url"
            placeholder="Resource URL"
            value={newResource.url || ""}
            onChange={(e) =>
              setNewResource({ ...newResource, url: e.target.value })
            }
            aria-label="New resource URL"
            className="bg-background"
          />
          <label htmlFor="new-title" className="text-sm font-bold">
            Resource Title
          </label>
          <Input
            id="new-title"
            placeholder="Resource Title"
            value={newResource.title || ""}
            onChange={(e) =>
              setNewResource({ ...newResource, title: e.target.value })
            }
            aria-label="New resource title"
            className="bg-background"
          />
          <label htmlFor="new-description" className="text-sm font-bold">
            Resource Description
          </label>
          <Textarea
            id="new-description"
            placeholder="Resource Description"
            value={newResource.description || ""}
            onChange={(e) =>
              setNewResource({
                ...newResource,
                description: e.target.value,
              })
            }
            aria-label="New resource description"
            className="bg-background"
          />
        </div>
        <Button
          onClick={addResource}
          className="w-full bg-[#6766FC] text-white"
          disabled={
            !newResource.url || !newResource.title || !newResource.description
          }
        >
          <Plus className="w-4 h-4 mr-2" /> Add Resource
        </Button>
      </DialogContent>
    </Dialog>
  );
}

================
File: ui/src/components/EditResourceDialog.tsx
================
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Resource } from "@/lib/types";

type EditResourceDialogProps = {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
  editResource: Resource | null;
  setEditResource: (
    resource: ((prev: Resource | null) => Resource | null) | Resource | null
  ) => void;
  updateResource: () => void;
};

export function EditResourceDialog({
  isOpen,
  onOpenChange,
  editResource,
  setEditResource,
  updateResource,
}: EditResourceDialogProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit Resource</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <label htmlFor="edit-url" className="text-sm font-bold">
            Resource URL
          </label>
          <Input
            id="edit-url"
            placeholder="Resource URL"
            value={editResource?.url || ""}
            onChange={(e) =>
              setEditResource((prev) =>
                prev ? { ...prev, url: e.target.value } : null
              )
            }
            aria-label="Edit resource URL"
            className="bg-background"
          />
          <label htmlFor="edit-title" className="text-sm font-bold">
            Resource Title
          </label>
          <Input
            id="edit-title"
            placeholder="Resource Title"
            value={editResource?.title || ""}
            onChange={(e) =>
              setEditResource((prev: any) =>
                prev ? { ...prev, title: e.target.value } : null
              )
            }
            aria-label="Edit resource title"
            className="bg-background"
          />
          <label htmlFor="edit-description" className="text-sm font-bold">
            Resource Description
          </label>
          <Textarea
            id="edit-description"
            placeholder="Resource Description"
            value={editResource?.description || ""}
            onChange={(e) =>
              setEditResource((prev) =>
                prev ? { ...prev, description: e.target.value } : null
              )
            }
            aria-label="Edit resource description"
            className="bg-background"
          />
        </div>
        <Button
          onClick={updateResource}
          className="w-full bg-[#6766FC] text-white"
          disabled={
            !editResource?.url ||
            !editResource?.title ||
            !editResource?.description
          }
        >
          Save Changes
        </Button>
      </DialogContent>
    </Dialog>
  );
}

================
File: ui/src/components/ModelSelector.tsx
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
File: ui/src/components/Progress.tsx
================
import { cn } from "@/lib/utils";
import { CheckIcon, LoaderCircle } from "lucide-react";
import { truncateUrl } from "@/lib/utils";

export function Progress({
  logs,
}: {
  logs: {
    message: string;
    done: boolean;
  }[];
}) {
  if (logs.length === 0) {
    return null;
  }

  return (
    <div data-test-id="progress-steps">
      <div className="border border-slate-200 bg-slate-100/30 shadow-md rounded-lg overflow-hidden text-sm py-2">
        {logs.map((log, index) => (
          <div
            key={index}
            data-test-id="progress-step-item"
            className={`flex ${
              log.done || index === logs.findIndex((log) => !log.done)
                ? ""
                : "opacity-50"
            }`}
          >
            <div className="w-8">
              <div
                  className="w-4 h-4 bg-slate-700 flex items-center justify-center rounded-full mt-[10px] ml-[12px]"
                  data-test-id={log.done ? 'progress-step-item_done' : 'progress-step-item_loading'}
              >
                {log.done ? (
                  <CheckIcon className="w-3 h-3 text-white" />
                ) : (
                  <LoaderCircle className="w-3 h-3 text-white animate-spin" />
                )}
              </div>
              {index < logs.length - 1 && (
                <div
                  className={cn("h-full w-[1px] bg-slate-200 ml-[20px]")}
                ></div>
              )}
            </div>
            <div className="flex-1 flex justify-center py-2 pl-2 pr-4">
              <div className="flex-1 flex items-center text-xs">
                {log.message.replace(
                  /https?:\/\/[^\s]+/g, // Regex to match URLs
                  (url) => truncateUrl(url) // Replace with truncated URL
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

================
File: ui/src/components/ResearchCanvas.tsx
================
"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  useCoAgent,
  useCoAgentStateRender,
  useCopilotAction,
} from "@copilotkit/react-core";
import { Progress } from "./Progress";
import { EditResourceDialog } from "./EditResourceDialog";
import { AddResourceDialog } from "./AddResourceDialog";
import { Resources } from "./Resources";
import { AgentState, Resource } from "@/lib/types";
import { useModelSelectorContext } from "@/lib/model-selector-provider";

export function ResearchCanvas() {
  const { model, agent } = useModelSelectorContext();

  const { state, setState } = useCoAgent<AgentState>({
    name: agent,
    initialState: {
      model,
    },
  });

  useCoAgentStateRender({
    name: agent,
    render: ({ state, nodeName, status }) => {
      if (!state.logs || state.logs.length === 0) {
        return null;
      }
      return <Progress logs={state.logs} />;
    },
  });

  useCopilotAction({
    name: "DeleteResources",
    description:
      "Prompt the user for resource delete confirmation, and then perform resource deletion",
    available: "remote",
    parameters: [
      {
        name: "urls",
        type: "string[]",
      },
    ],
    renderAndWait: ({ args, status, handler }) => {
      return (
        <div
          className=""
          data-test-id="delete-resource-generative-ui-container"
        >
          <div className="font-bold text-base mb-2">
            Delete these resources?
          </div>
          <Resources
            resources={resources.filter((resource) =>
              (args.urls || []).includes(resource.url)
            )}
            customWidth={200}
          />
          {status === "executing" && (
            <div className="mt-4 flex justify-start space-x-2">
              <button
                onClick={() => handler("NO")}
                className="px-4 py-2 text-[#6766FC] border border-[#6766FC] rounded text-sm font-bold"
              >
                Cancel
              </button>
              <button
                data-test-id="button-delete"
                onClick={() => handler("YES")}
                className="px-4 py-2 bg-[#6766FC] text-white rounded text-sm font-bold"
              >
                Delete
              </button>
            </div>
          )}
        </div>
      );
    },
  });

  const resources: Resource[] = state.resources || [];
  const setResources = (resources: Resource[]) => {
    setState({ ...state, resources });
  };

  // const [resources, setResources] = useState<Resource[]>(dummyResources);
  const [newResource, setNewResource] = useState<Resource>({
    url: "",
    title: "",
    description: "",
  });
  const [isAddResourceOpen, setIsAddResourceOpen] = useState(false);

  const addResource = () => {
    if (newResource.url) {
      setResources([...resources, { ...newResource }]);
      setNewResource({ url: "", title: "", description: "" });
      setIsAddResourceOpen(false);
    }
  };

  const removeResource = (url: string) => {
    setResources(
      resources.filter((resource: Resource) => resource.url !== url)
    );
  };

  const [editResource, setEditResource] = useState<Resource | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [isEditResourceOpen, setIsEditResourceOpen] = useState(false);

  const handleCardClick = (resource: Resource) => {
    setEditResource({ ...resource }); // Ensure a new object is created
    setOriginalUrl(resource.url); // Store the original URL
    setIsEditResourceOpen(true);
  };

  const updateResource = () => {
    if (editResource && originalUrl) {
      setResources(
        resources.map((resource) =>
          resource.url === originalUrl ? { ...editResource } : resource
        )
      );
      setEditResource(null);
      setOriginalUrl(null);
      setIsEditResourceOpen(false);
    }
  };

  return (
    <div className="w-full h-full p-10 bg-[#F5F8FF]">
      <div className="space-y-8">
        <div>
          <h2 className="text-lg font-medium mb-3 text-primary">
            Research Question
          </h2>
          <Input
            placeholder="Enter your research question"
            value={state.research_question || ""}
            onChange={(e) =>
              setState({ ...state, research_question: e.target.value })
            }
            aria-label="Research question"
            className="bg-background px-6 py-8 border-0 shadow-none rounded-xl text-md font-extralight focus-visible:ring-0 placeholder:text-slate-400"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-primary">Resources</h2>
            <EditResourceDialog
              isOpen={isEditResourceOpen}
              onOpenChange={setIsEditResourceOpen}
              editResource={editResource}
              setEditResource={setEditResource}
              updateResource={updateResource}
            />
            <AddResourceDialog
              isOpen={isAddResourceOpen}
              onOpenChange={setIsAddResourceOpen}
              newResource={newResource}
              setNewResource={setNewResource}
              addResource={addResource}
            />
          </div>
          {resources.length === 0 && (
            <div className="text-sm text-slate-400">
              Click the button above to add resources.
            </div>
          )}

          {resources.length !== 0 && (
            <Resources
              resources={resources}
              handleCardClick={handleCardClick}
              removeResource={removeResource}
            />
          )}
        </div>

        <div className="flex flex-col h-full">
          <h2 className="text-lg font-medium mb-3 text-primary">
            Research Draft
          </h2>
          <Textarea
            data-test-id="research-draft"
            placeholder="Write your research draft here"
            value={state.report || ""}
            onChange={(e) => setState({ ...state, report: e.target.value })}
            rows={10}
            aria-label="Research draft"
            className="bg-background px-6 py-8 border-0 shadow-none rounded-xl text-md font-extralight focus-visible:ring-0 placeholder:text-slate-400"
            style={{ minHeight: "200px" }}
          />
        </div>
      </div>
    </div>
  );
}

================
File: ui/src/components/Resources.tsx
================
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { Resource } from "@/lib/types";
import { truncateUrl } from "@/lib/utils";

type ResourcesProps = {
  resources: Resource[];
  customWidth?: number;
  handleCardClick?: (resource: Resource) => void;
  removeResource?: (url: string) => void;
};

export function Resources({
  resources,
  handleCardClick,
  removeResource,
  customWidth,
}: ResourcesProps) {
  return (
    <div data-test-id="resources" className="flex space-x-3 overflow-x-auto">
      {resources.map((resource, idx) => (
        <Card
          data-test-id={`resource`}
          key={idx}
          className={
            "bg-background border-0 shadow-none rounded-xl text-md font-extralight focus-visible:ring-0 flex-none" +
            (handleCardClick ? " cursor-pointer" : "")
          }
          style={{ width: customWidth + "px" || "320px" }}
          onClick={() => handleCardClick?.(resource)}
        >
          <CardContent className="px-6 py-6 relative">
            <div className="flex items-start space-x-3 text-sm">
              <div className="flex-grow">
                <h3
                  className="font-bold text-lg"
                  style={{
                    maxWidth: customWidth ? customWidth - 30 + "px" : "230px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {resource.title}
                </h3>
                <p
                  className="text-base mt-2"
                  style={{
                    maxWidth: customWidth ? customWidth - 30 + "px" : "250px",
                    overflowWrap: "break-word",
                  }}
                >
                  {resource.description?.length > 250
                    ? resource.description.slice(0, 250) + "..."
                    : resource.description}
                </p>
                <a
                  href={resource.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-primary hover:underline mt-3 text-slate-400 inline-block"
                  title={resource.url}
                  style={{
                    width: customWidth ? customWidth - 30 + "px" : "250px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {resource.description && (
                    <>
                      <img
                        src={`https://www.google.com/s2/favicons?domain=${resource.url}`}
                        alt="favicon"
                        className="inline-block mr-2"
                        style={{ width: "16px", height: "16px" }}
                      />
                      {truncateUrl(resource.url)}
                    </>
                  )}
                </a>
              </div>
              {removeResource && (
                <div className="flex items-start absolute top-4 right-4">
                  <Button
                    data-test-id="remove-resource"
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeResource?.(resource.url);
                    }}
                    aria-label={`Remove ${resource.url}`}
                  >
                    <Trash2 className="w-6 h-6 text-gray-400 hover:text-red-500" />
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

================
File: ui/src/components/ui/button.tsx
================
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90",
        outline:
          "border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
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
File: ui/src/components/ui/card.tsx
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
      "rounded-xl border bg-card text-card-foreground shadow",
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
    className={cn("font-semibold leading-none tracking-tight", className)}
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
File: ui/src/components/ui/dialog.tsx
================
"use client"

import * as React from "react"
import * as DialogPrimitive from "@radix-ui/react-dialog"
import { Cross2Icon } from "@radix-ui/react-icons"

import { cn } from "@/lib/utils"

const Dialog = DialogPrimitive.Root

const DialogTrigger = DialogPrimitive.Trigger

const DialogPortal = DialogPrimitive.Portal

const DialogClose = DialogPrimitive.Close

const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      "fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
  />
))
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        "fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg",
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
        <Cross2Icon className="h-4 w-4" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName

const DialogHeader = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col space-y-1.5 text-center sm:text-left",
      className
    )}
    {...props}
  />
)
DialogHeader.displayName = "DialogHeader"

const DialogFooter = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",
      className
    )}
    {...props}
  />
)
DialogFooter.displayName = "DialogFooter"

const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  />
))
DialogTitle.displayName = DialogPrimitive.Title.displayName

const DialogDescription = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Description>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Description>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Description
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  />
))
DialogDescription.displayName = DialogPrimitive.Description.displayName

export {
  Dialog,
  DialogPortal,
  DialogOverlay,
  DialogTrigger,
  DialogClose,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
}

================
File: ui/src/components/ui/input.tsx
================
import * as React from "react"

import { cn } from "@/lib/utils"

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
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
File: ui/src/components/ui/select.tsx
================
"use client"

import * as React from "react"
import {
  CaretSortIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from "@radix-ui/react-icons"
import * as SelectPrimitive from "@radix-ui/react-select"

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
      "flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
      className
    )}
    {...props}
  >
    {children}
    <SelectPrimitive.Icon asChild>
      <CaretSortIcon className="h-4 w-4 opacity-50" />
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
    <ChevronUpIcon />
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
    <ChevronDownIcon />
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
    className={cn("px-2 py-1.5 text-sm font-semibold", className)}
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
      "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-2 pr-8 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      className
    )}
    {...props}
  >
    <span className="absolute right-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectPrimitive.ItemIndicator>
        <CheckIcon className="h-4 w-4" />
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
File: ui/src/components/ui/textarea.tsx
================
import * as React from "react"

import { cn } from "@/lib/utils"

export type TextareaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[60px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
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
File: ui/src/lib/model-selector-provider.tsx
================
"use client";

import React from "react";
import { createContext, useContext, useState, ReactNode } from "react";

type ModelSelectorContextType = {
  model: string;
  setModel: (model: string) => void;
  agent: string;
  lgcDeploymentUrl?: string | null;
  hidden: boolean;
  setHidden: (hidden: boolean) => void;
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

  const agent =
    model === "google_genai" ? "research_agent_google_genai" : "research_agent";

  return (
    <ModelSelectorContext.Provider
      value={{
        model,
        agent,
        lgcDeploymentUrl,
        hidden,
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
File: ui/src/lib/types.ts
================
export type Resource = {
  url: string;
  title: string;
  description: string;
};

export type AgentState = {
  model: string;
  research_question: string;
  report: string;
  resources: any[];
  logs: any[];
}

================
File: ui/src/lib/utils.ts
================
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const truncateUrl = (url: string, maxLength: number = 40) => {
  if (!url) return "";
  if (url.length <= maxLength) return url;
  return url.substring(0, maxLength - 3) + "...";
};

================
File: ui/tailwind.config.ts
================
import type { Config } from "tailwindcss";

const config: Config = {
    darkMode: ["class"],
    content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
  	extend: {
  		colors: {
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			}
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
};
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
        "./src/*"
      ]
    },
    "target": "ES2017"
  },
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx",
    ".next/types/**/*.ts"
  ],
  "exclude": [
    "node_modules"
  ]
}
================================================================
End of Codebase
================================================================
