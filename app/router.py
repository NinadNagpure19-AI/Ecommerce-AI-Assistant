from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)

faq = Route(
    name='faq',
    score_threshold=0.2,
    utterances=[
        # return policy
        "What is the return policy of the products?",
        "How do I return a product?",
        "Can I return a product after 30 days?",
        "What is your return policy?",
        # defective / damaged
        "What should I do if I receive a damaged product?",
        "I received a defective product",
        "My product is broken, what do I do?",
        "I got the wrong item, how do I report it?",
        # refunds
        "How long does it take to process a refund?",
        "When will I get my money back?",
        "How do I get a refund?",
        # payments
        "What payment methods are accepted?",
        "Do I get discount with the HDFC credit card?",
        "Can I pay with UPI?",
        "Do you accept cash on delivery?",
        # tracking
        "How can I track my order?",
        "Where is my order?",
        "How do I check my order status?",
    ]
)

sql = Route(
    name='sql',
    score_threshold=0.3,
    utterances=[
        # brand queries
        "I want to buy nike shoes that have 50% discount.",
        "Are there any Puma shoes on sale?",
        "Show me Adidas shoes",
        "What is the price of puma running shoes?",
        # price filters
        "Are there any shoes under Rs. 3000?",
        "Show me shoes between 2000 and 5000 rupees",
        "What are the cheapest shoes available?",
        "Show me shoes above 4000 rupees",
        # size / type
        "Do you have formal shoes in size 9?",
        "Show me running shoes in size 8",
        "What kind of shoes do you have?",
        "Show me all available shoes",
        # rating / discount
        "Show me shoes with rating above 4.5",
        "Which shoes have the highest discount?",
        "Show me top rated Nike shoes",
        "Which Puma shoes have more than 40% discount?",
        # aggregations
        "How many Puma shoes are available?",
        "What is the average rating of Nike shoes?",
        "What is the cheapest Adidas shoe?",
    ]
)

routes = [faq, sql]

router = SemanticRouter(routes=routes, encoder=encoder, auto_sync='local')


if __name__ == "__main__":

    print("\n=== Smoke Test ===")
    print(router("What is your policy on defective product?").name)
    print(router("Pink Puma shoes in price range 5000 to 1000").name)

    print("\n=== Threshold Calibration ===")

    test_queries = [
        # expected: faq
        ("What is the return policy?", "faq"),
        ("I received a defective product", "faq"),
        ("When will I get my refund?", "faq"),
        ("How can I track my order?", "faq"),
        ("What payment options do you have?", "faq"),
        ("Can I pay with Google Pay?", "faq"),
        ("My order is damaged", "faq"),
        # expected: sql
        ("Show me Nike shoes under 3000", "sql"),
        ("Are there any Puma shoes on sale?", "sql"),
        ("What kind of shoes do you have?", "sql"),
        ("Cheapest shoes available", "sql"),
        ("Show shoes with rating above 4.5", "sql"),
        ("How many Adidas shoes are there?", "sql"),
        # expected: None (out of domain)
        ("Who won the cricket world cup?", None),
        ("How do I become a pilot?", None),
        ("What is the weather today?", None),
    ]

    correct = 0
    total = len(test_queries)

    for query, expected in test_queries:
        result = router(query)
        actual = result.name
        score = result.similarity_score
        score_str = f"{score:.3f}" if score is not None else "None"
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} [{expected} -> {actual}] score: {score_str:>6}  |  {query}")
        if actual == expected:
            correct += 1

    print(f"\nAccuracy: {correct}/{total} ({100 * correct // total}%)")