from groq import Groq
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

db_path = Path(__file__).parent / "db.sqlite"

client_sql = Groq()

sql_prompt = """
You are an expert in understanding a database schema and generating SQL queries
for questions that can be answered using the available data.

The schema is provided below.

<schema>

table: product

fields:
product_link - string (hyperlink to product)
title - string (name of the product)
brand - string (brand of the product)
price - integer (price of the product in Indian Rupees)
discount - float (discount on the product.
                  10 percent discount = 0.1,
                  20 percent discount = 0.2)
avg_rating - float (average rating of the product.
                    Range 0-5, where 5 is highest)
total_ratings - integer (total number of ratings)

</schema>

Rules:

1. Generate SQL ONLY if the question can be answered
   using the schema provided above.

2. If the question is unrelated to products, shopping,
   brands, prices, discounts, ratings, reviews, or any
   information available in the schema, return:

<SQL>
NOT_RELEVANT
</SQL>

3. Do NOT invent columns, tables, or values that do not
   exist in the schema.

4. Brand names can appear in any case.
   Always perform case-insensitive matching using:

   LOWER(column_name) LIKE LOWER('%value%')

   Never use ILIKE.

5. When filtering shoes or products by title,
   use case-insensitive matching:

   LOWER(title) LIKE LOWER('%shoe%')

6. For questions asking:
   - "How many" → use COUNT(*)
   - "Average" → use AVG(...)
   - "Maximum" → use MAX(...)
   - "Minimum" → use MIN(...)

7. For ranking questions:
   - "highest rated" → ORDER BY avg_rating DESC
   - "cheapest" → ORDER BY price ASC
   - "most expensive" → ORDER BY price DESC
   - "highest discount" → ORDER BY discount DESC

8. Use SELECT * whenever the user is asking for products.

9. Generate exactly ONE SQL query.

10. Return ONLY the SQL query wrapped inside:

<SQL>
...
</SQL>

No explanations.
No markdown.
No additional text.

11. Always add LIMIT 10 to queries that use SELECT *,
    unless the user specifies a different number.
"""

comprehension_prompt = """
You are an expert at converting database query results
into clear, natural language answers.

You will receive:

QUESTION: The user's question.
DATA: The query result.

Rules:

1. Answer ONLY using the information present in DATA.

2. Never make up facts, products, prices,
   ratings, discounts, or links.

3. If DATA is empty, reply:

   "I couldn't find any matching products."

4. Do not mention:
   - databases
   - SQL
   - tables
   - records
   - rows
   - technical details

5. Give a direct answer to the user's question.

6. When DATA contains product records,
   format the response as:

   1. Product Title: Rs. Price
      (X% off), Rating: Y
      Product Link

   2. Product Title: Rs. Price
      (X% off), Rating: Y
      Product Link

7. When DATA contains a single numeric result
   (COUNT, AVG, MAX, MIN, etc.),
   answer naturally.

Examples:

Question:
How many Puma shoes are available?

Data:
[{'count': 25}]

Answer:
There are 25 Puma shoes available.

Question:
What is the average rating of Nike shoes?

Data:
[{'avg_rating': 4.32}]

Answer:
The average rating of Nike shoes is 4.32.

Use only the provided data.
"""


def generate_sql_from_question(question: str) -> str:
    """
    Converts a natural language question into an SQL query
    using an LLM.

    Args:
        question (str): User question.

    Returns:
        str: Raw LLM response containing SQL inside
             <SQL></SQL> tags.
    """

    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sql_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        model=os.environ["GROQ_MODEL"],
        temperature=0.2,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def run_query(query: str) -> pd.DataFrame | None:
    """
    Executes a SELECT query on SQLite and returns
    the result as a Pandas DataFrame.

    Args:
        query (str): SQL query.

    Returns:
        pd.DataFrame | None
    """
    if query.strip().upper().startswith("SELECT"):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df


def generate_natural_language_answer(question: str, context: list[dict]) -> str:
    """
    Converts structured database results into a
    human-readable answer using an LLM.

    Args:
        question (str): Original user question.
        context (list[dict]): Query results.

    Returns:
        str: Natural language response.
    """
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": comprehension_prompt,
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}. DATA: {context}",
            },
        ],
        model=os.environ["GROQ_MODEL"],
        temperature=0.2,
        # max_tokens=1024
    )
    return chat_completion.choices[0].message.content


def sql_chain(question: str) -> dict:
    """
    End-to-end SQL Question Answering pipeline.

    Flow:
        1. Convert user question into SQL using LLM.
        2. Extract SQL query from LLM response.
        3. Execute query against SQLite database.
        4. Convert query results into structured records.
        5. Generate a natural language answer using LLM.
        6. Return answer along with metadata.

    Args:
        question (str):
            User's natural language question.

    Returns:
        dict:
            {
                "answer": str,
                "metadata": {
                    "sql_query": str
                }
            }
    """

    sql_query = generate_sql_from_question(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if len(matches) == 0:
        return {
            "answer": "Sorry, LLM is not able to generate a query for your question",
            "metadata": {},
        }

    sql_text = matches[0].strip()

    if sql_text == "NOT_RELEVANT":
        return {
            "answer": "This question cannot be answered using the product database.",
            "metadata": {},
        }

    print(sql_text)

    response = run_query(sql_text)
    if response is None:
        return {
            "answer": "Sorry, there was a problem executing SQL query",
            "metadata": {},
        }

    context = response.to_dict(orient="records")[:10]  # limit to 10 rows

    answer = generate_natural_language_answer(question, context)

    return {"answer": answer, "metadata": {"sql_query": sql_text}}


if __name__ == "__main__":

    question = "Show top 3 shoes in descending order of rating"

    print("\n=== Testing SQL Generation ===")
    llm_response = generate_sql_from_question(question)
    print(llm_response)

    print("\n=== Testing SQL Extraction ===")
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, llm_response, re.DOTALL)

    if len(matches) == 0:
        print("No SQL found in LLM response")
        exit()

    generated_sql = matches[0].strip()
    print(generated_sql)

    print("\n=== Testing Query Execution ===")
    df = run_query(generated_sql)

    if df is None:
        print("Query execution failed")
        exit()

    print(df.head())

    print("\n=== Testing DataFrame Info ===")
    print(f"Type: {type(df)}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("\n=== Testing Context Creation ===")
    context = df.to_dict(orient="records")
    print(context)

    print("\n=== Testing Answer Generation ===")
    answer = generate_natural_language_answer(question, context)
    print(answer)

    print("\n=== Testing Full SQL Pipeline ===")
    final_answer = sql_chain(question)
    print(final_answer)

    print("\n=== SQL Generation Calibration ===")

    test_questions = [
        # Basic retrieval
        "Show top 3 shoes by rating",
        "Show Nike shoes",
        "Show Puma shoes",
        "Show Adidas shoes",
        # Price filters
        "Show Puma shoes under 2000",
        "Show Nike shoes under 3000",
        "Show Adidas shoes above 5000",
        "Show shoes between 2000 and 4000 rupees",
        # Rating filters
        "Show shoes with rating above 4.5",
        "Show Nike shoes with rating above 4.3",
        # Discount filters
        "Show shoes with discount greater than 30%",
        "Show Puma shoes with discount above 40%",
        # Multiple conditions
        "Give me Puma shoes with rating higher than 4.5 and discount more than 30%",
        "Show Nike shoes under 4000 with rating above 4.2",
        "Show Adidas shoes with more than 1000 ratings and discount above 20%",
        "Show Puma shoes under 2500 with rating above 4.0 and discount above 25%",
        # Sorting
        "Show top 5 shoes by rating",
        "Show the cheapest 5 shoes",
        "Show the most expensive Nike shoes",
        "Show shoes with highest discount",
        # Aggregations
        "What is the average rating of Nike shoes?",
        "How many Puma shoes are available?",
        "What is the maximum discount offered on Adidas shoes?",
        "What is the cheapest Nike shoe?",
        # Edge cases
        "Show Reebok shoes under 1000 with rating above 4.5",
        "Show products with more than 5000 ratings",
        # Out of domain
        "Who won the cricket world cup?",
        "How do I become a pilot?",
        "Tell me today's weather",
    ]

    for q in test_questions:

        print("\n" + "-" * 50)
        print(f"QUESTION: {q}")

        llm_response = generate_sql_from_question(q)

        matches = re.findall(pattern, llm_response, re.DOTALL)

        if len(matches) > 0:
            print(matches[0].strip())
        else:
            print("No SQL generated")
