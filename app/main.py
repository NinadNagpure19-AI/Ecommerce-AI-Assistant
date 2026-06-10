import streamlit as st
from faq_2 import ingest_faq_data, faq_rag_pipeline
from sql import sql_chain
from pathlib import Path
from router import router

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
ingest_faq_data(faqs_path)


def ask(query):
    route = router(query).name
    if route == "faq":
        return faq_rag_pipeline(query)
    elif route == "sql":
        return sql_chain(query)
    else:
        return {
            "answer": "I can only help with product queries or store-related questions. Please ask something else.",
            "metadata": {},
        }


st.title("E-commerce Bot")

query = st.chat_input("Write your query")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        metadata = message.get("metadata")

        if metadata:
            if "distance" in metadata:
                st.caption(
                    f"Source: {metadata['source']} | Distance: {metadata['distance']:.3f}"
                )

            elif "sql_query" in metadata:
                st.caption(f"SQL: {metadata['sql_query']}")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = ask(query)
    answer = response["answer"]
    metadata = response["metadata"]

    with st.chat_message("assistant"):
        st.markdown(answer)
        if "distance" in metadata:
            st.caption(
                f"Source: {metadata['source']} | Distance: {metadata['distance']:.3f}"
            )

        elif "sql_query" in metadata:
            st.caption(f"SQL: {metadata['sql_query']}")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "metadata": metadata}
    )
