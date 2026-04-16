import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")
print("\n📊 Dataset:\n", df)

# Create model
model = pulp.LpProblem("Profit_Maximization", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable.dicts("Prod", df['Product'], lowBound=0, cat='Integer')

# Objective function
model += pulp.lpSum(df.loc[i, 'Profit'] * x[df.loc[i, 'Product']]
                    for i in df.index)

# Constraints
material_limit = 200
labour_limit = 150

model += pulp.lpSum(df.loc[i, 'Material'] * x[df.loc[i, 'Product']]
                    for i in df.index) <= material_limit

model += pulp.lpSum(df.loc[i, 'Labour'] * x[df.loc[i, 'Product']]
                    for i in df.index) <= labour_limit

# Demand constraint
for i in df.index:
    model += x[df.loc[i, 'Product']] <= df.loc[i, 'Max_Demand']

# Solve
model.solve()

print("\n✅ Status:", pulp.LpStatus[model.status])

# Results
df["Production"] = [x[p].value() for p in df['Product']]
df["Total Profit"] = df["Production"] * df["Profit"]

print("\n📋 Result:\n", df)
print("\n💰 Total Profit =", pulp.value(model.objective))

# ---------------- VISUALS ---------------- #

# Production
plt.figure()
sns.barplot(x="Product", y="Production", data=df)
plt.title("Optimal Production Plan")
plt.show()

# Profit
plt.figure()
sns.barplot(x="Product", y="Total Profit", data=df)
plt.title("Profit Distribution")
plt.show()