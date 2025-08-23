import matplotlib.pyplot as plt

# Number of years to project
years = 30

# Initial values
rent = 1895*12
mortgage = 3000*12

# Growth rates
rent_growth_rate = 0.05
stock_market_rate = 0.07

# Lists to store values for each year
rent_cost = []
mortgage_cost = [mortgage] * years
savings = []

# Initial accumulated value
accum_value = 0

print("Initial values:")
print(f"Rent: {rent}")
print(f"Mortgage: {mortgage}")
for year in range(years):
    rent_cost.append(rent)
    
    saved = mortgage - rent
    invested = abs(saved*.6)
    print("saved: ", saved)
    accum_value = (accum_value + invested) * (1 + stock_market_rate)
    savings.append(accum_value)
    
    # Increase rent for next year
    rent = rent * (1 + rent_growth_rate)

# Plotting the values
plt.figure(figsize=(12, 6))

plt.plot(range(years), rent_cost, label='Rent (5% increase per year)')
plt.plot(range(years), mortgage_cost, label='Flat Value ($3000)')
plt.plot(range(years), savings, label='Accumulated Value (7% growth per year)')

plt.xlabel('Years')
plt.ylabel('Value ($)')
plt.title('Rent vs Flat Value and Accumulated Difference')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
