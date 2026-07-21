import pandas as pd

test_data = [
    {"Date": "2026-06-01", "Description": "Yerevan City Supermarket", "Amount": -12500},
    {"Date": "2026-06-02", "Description": "Yandex Go Taxi Yerevan", "Amount": -1800},
    {"Date": "2026-06-05", "Description": "Salary Transfer ACME LLC", "Amount": 500000},
    {"Date": "2026-06-07", "Description": "Tavern Yerevan Restaurant", "Amount": -28000},
    {"Date": "2026-06-10", "Description": "SAS Supermarket Komitas", "Amount": -9400},
    {"Date": "2026-06-12", "Description": "GG Taxi Yerevan", "Amount": -1500},
    {"Date": "2026-06-15", "Description": "VEON Armenia Mobile Pay", "Amount": -4500},
    
    {"Date": "2026-06-18", "Description": "BUYING NEW MACBOOK PRO UTILITY", "Amount": -950000},
    
    {"Date": "2026-06-20", "Description": "Carrefour Mall Supermarket", "Amount": -14200},
    {"Date": "2026-06-22", "Description": "Coffee House Cascade", "Amount": -3200},
    {"Date": "2026-06-25", "Description": "Wildberries Online Order", "Amount": -18500},
    {"Date": "2026-06-28", "Description": "CPS Petrol Station Yerevan", "Amount": -20000},

    {"Date": "2026-07-01", "Description": "Yerevan City Supermarket", "Amount": -15300},
    {"Date": "2026-07-03", "Description": "Yandex Go Taxi", "Amount": -2200},
    {"Date": "2026-07-05", "Description": "Salary Transfer ACME LLC", "Amount": 500000},
    {"Date": "2026-07-08", "Description": "Dargett Craft Beer Restaurant", "Amount": -19000},
    {"Date": "2026-07-12", "Description": "Zangak Bookstore", "Amount": -8500},
    {"Date": "2026-07-15", "Description": "JAZZVE Coffee Yerevan", "Amount": -2800},
]

df = pd.DataFrame(test_data)
df.to_csv("test_statement_auto.csv", index=False)
print("'test_statement_auto.csv' successfully!")
