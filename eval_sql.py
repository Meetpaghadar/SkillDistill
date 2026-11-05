import sqlite3

conn = sqlite3.connect(":memory:")

conn.execute("""
CREATE TABLE orders (
    id INTEGER,
    restaurant TEXT,
    amount REAL,
    status TEXT
)
""")

conn.executemany(
    "INSERT INTO orders VALUES (?, ?, ?, ?)",
    [
        (1, "A", 100, "completed"),
        (2, "A", 200, "completed"),
        (3, "A", 50, "cancelled"),
        (4, "B", 150, "completed"),
        (5, "B", 80, "cancelled"),
    ],
)

tests = [
    {
        "question": "Total completed revenue for restaurant A?",
        "sql": """
            SELECT SUM(amount)
            FROM orders
            WHERE restaurant = 'A'
              AND status = 'completed'
        """,
        "expected": [(300.0,)],
    },
    {
        "question": "How many completed orders does B have?",
        "sql": """
            SELECT COUNT(*)
            FROM orders
            WHERE restaurant = 'B'
              AND status = 'completed'
        """,
        "expected": [(1,)],
    },
    {
        "question": "Incorrect query: revenue including cancellations",
        "sql": """
            SELECT SUM(amount)
            FROM orders
            WHERE restaurant = 'A'
        """,
        "expected": [(300.0,)],
    },
]

for i, test in enumerate(tests, start=1):
    actual = conn.execute(test["sql"]).fetchall()
    passed = actual == test["expected"]

    print(
        f"Test {i}: {'PASS' if passed else 'FAIL'} | "
        f"Expected: {test['expected']} | Actual: {actual}"
    )

conn.close()
