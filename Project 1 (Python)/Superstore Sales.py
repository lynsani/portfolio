# **Introduction**

This notebook explores the **"Superstore Sales"** dataset, a comprehensive collection of sales data from a large retail store offering a wide variety of products. The dataset includes information on **sales, customer segments, and product categories**, covering a period of 2015-2018.

The goal of this analysis is to:
- **Uncover sales patterns** across different regions, categories, and customer segments.
- **Assess business performance** by analyzing shipping efficiency, and customer behavior.
- **Provide actionable insights** to improve profitability and enhance customer satisfaction.

# Import All Dependencies
"""

# Libraries for data manipulation, visualization, and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with Latin-1 encoding to handle special characters
df = pd.read_csv('superstore_final_dataset (1).csv', encoding='latin-1')

"""# Data Exploring and Understanding"""

# Initial Check
df.head(20)

# Check all columns info
df.info()
print(f'Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}')

# Check statistic value of dataset
df.describe()

# Distribution of Sales with Mean and Median
plt.figure(figsize=(10, 6))
plt.hist(df['Sales'], bins=30, color='skyblue', edgecolor='black', log=True)
plt.axvline(df['Sales'].mean(), color='red', linestyle='--', label=f"Mean: ${df['Sales'].mean():.2f}")
plt.axvline(df['Sales'].median(), color='green', linestyle='--', label=f"Median: ${df['Sales'].median():.2f}")
plt.title('Distribution of Sales (Log Scale)', fontsize=16)
plt.xlabel('Sales (USD)', fontsize=12)
plt.ylabel('Frequency (Log Scale)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=1)
plt.show()

# Calculate skewness coefficient
sales_skewness = df['Sales'].skew()
print(f'Skewness coefficient : {sales_skewness:.2f}')

"""The sales data is right-skewed, with a skewness coefficient of 12.98. This indicates that the distribution has a longer tail on the right, with a mean of 230.76 being significantly higher than the median of 57.49. This suggests that a few high-value transactions are pulling the mean upwards."""

# Identified the duplicates
print(f'There are {df.duplicated().sum()} duplicates in the dataset')

# Check for missing values in the dataset
missing_value = df.isna().sum().sort_values(ascending=False)
print(missing_value)

# Assess the significance of missing values
print(f'The percentage of missing value in Postal Code is : {round((df["Postal_Code"].isna().sum() / len(df)) * 100, 2)}% of data set')

"""The Postal Code column has 11 missing values, which represents only 0.11% of the dataset. Since this is a small percentage and Postal Code is not critical for our analysis, we can safely ignore these missing values."""

# Unique Value each columns
for column in df.columns:
    unique_values = df[column].nunique()
    print(f"Unique values in column '{column}': {unique_values}")
    print()

"""**Key Takeaways**

- The dataset consists of **18 features**, including **15 categorical** and **3 numerical** variables.
- There are **no duplicated records** in the dataset, ensuring data integrity.
- The **Order_Date** and **Ship_Date** columns need to be converted into datetime format for time-based analysis.
- The **Postal_Code** column contains **11 missing values**, which represent only **0.11%** of the dataset and are insignificant for our analysis.
- Some **Product_ID** values share the same **Product_Name**, indicating potential data entry issues or product variants.
- The Sales data exhibits a right-skewed distribution, as indicated by:
  * The mean being greater than the median.
  * The maximum value being significantly larger than the mean.
  * The data having high variability based on the standard deviation.

**Irrelevant Columns**
The following columns are irrelevant for analysis and can be dropped:
- **Row_ID**: This is just an index and provides no analytical value.
- **Customer_Name**: Contains too many unique values and is not useful for aggregate analysis.
- **Postal_Code**: Not critical for regional analysis, as we already have **City** and **State** columns.
- **Country**: Contains only **1 unique value** (United States), making it redundant for analysis.
- **Product_ID**: While useful for unique identification, it is not necessary for aggregate analysis.

# Data Cleaning
"""

# Dropping columns that are irrelevant for analysis
df.drop(['Country', 'Row_ID', 'Customer_Name', 'Postal_Code', 'Product_ID'], axis=1, inplace=True)

# Convert order_date and ship_date to datetime format
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d/%m/%Y')
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], format='%d/%m/%Y')

# Create new columns for time-based analysis
# - Month_Year_OD: Combines year and month from Order_Date for monthly analysis.
# - Quarter: Extracts the quarter from Order_Date for quarterly analysis.
# - Year_OD: Extracts the year from Order_Date for yearly analysis.
# - Year_Quarter: Combines year and quarter for easier time-series grouping.
df['Month_Year_OD'] = df['Order_Date'].dt.to_period('M')
df['Quarter'] = df['Order_Date'].dt.quarter
df['Year_OD'] = df['Order_Date'].dt.year
df['Year_Quarter'] = df['Year_OD'].astype(str) + '-Q' + df['Quarter'].astype(str)

# Rename columns to lowercase for consistency
df = df.rename(columns=str.lower)

"""# Exploratory Data Analysis (EDA)

### Customer Behaviour

##### Customer Retention
"""

# Count orders per customer
customer_orders = df.groupby(['customer_id',])['order_date'].nunique()

# Identify repeat customers
repeat_customers = customer_orders[customer_orders > 1].count()
total_customers = customer_orders.count()

# Calculate customer retention rate
customer_retention_rate = (repeat_customers / total_customers) * 100
print(f"Customer Retention Rate: {customer_retention_rate:.2f}%")

# Count repeat orders over time
repeat_orders = df.groupby(['customer_id', 'month_year_od'])['order_id'].nunique().reset_index()
repeat_orders = repeat_orders[repeat_orders['order_id'] > 1]

# Aggregate by month
repeat_orders_over_time = repeat_orders.groupby('month_year_od').size()


plt.figure(figsize=(10, 6))
repeat_orders_over_time.plot(kind='line', color='blue')
plt.title('Repeat Orders Trend')
plt.xlabel('Month-Year')
plt.ylabel('Number of Repeat Orders')
plt.grid(True)
plt.show()

# Repeat orders date difference
customer_order_date = df[['customer_id', 'order_date']].sort_values(by=['customer_id', 'order_date'])
customer_order_date['date_diff'] = customer_order_date.groupby('customer_id')['order_date'].diff().dt.days
print(f'The average time between repeat orders for customers is {customer_order_date["date_diff"].mean():.2f} days')

# Calculate average time between repeat orders for each customer segment
segment_time_diff = df.groupby('segment').apply(
    lambda x: x.sort_values('order_date').groupby('customer_id')['order_date'].diff().dropna().dt.days.mean()
).reset_index(name='avg_time_between_orders')

plt.figure(figsize=(10, 6))
sns.barplot(data=segment_time_diff, x='segment', y='avg_time_between_orders', palette='viridis', hue='segment', legend=False)
plt.title('Average Time Between Orders by Customer Segment', fontsize=16)
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Average Time Between Orders (Days)', fontsize=12)

for i, avg_time in enumerate(segment_time_diff['avg_time_between_orders']):
    plt.text(i, avg_time, f"{avg_time:,.2f}", ha='center', fontsize=10)

plt.show()

# Number of customers
first_purchase = df.groupby('customer_id')['order_date'].min().reset_index()
first_purchase['first_purchase_year'] = first_purchase['order_date'].dt.year

new_customers = first_purchase.groupby('first_purchase_year')['customer_id'].nunique().reset_index()
new_customers.columns = ['year', 'new_customers']

total_customers = df.groupby('year_od')['customer_id'].nunique().reset_index()
total_customers.columns = ['year', 'total_customers']

customers = pd.merge(new_customers, total_customers, on='year')
customers

# Customer Growth Rate
customer_growth_rate = ((customers[customers['year'] == 2018]['total_customers'].values[0] - customers[customers['year'] == 2015]['total_customers'].values[0]) / customers[customers['year'] == 2015]['new_customers'].values[0]*100)
print(f'Customer Growth Rate: {customer_growth_rate:.2f}%')

"""Key Insights:

* The customer retention rate demonstrates exceptional performance, maintaining a high value of 98.36%.

 - The high retention rate suggests that the majority of customers make repeat purchases, indicating strong customer loyalty.
 - However, new customer acquisition is minimal, with total customers growing only 17% from 2015–2018. This reliance on retained customers limits growth potential.

* The repeat order graph reveals an overall upward trend in repeat orders over the years, indicates a noticeable trend, with a significant increase in repeat orders observed during the second half of each year.
* The average time between repeat orders for customers is 87.45 days or approximately 3 months.

  - Corporate customers have the shortest average time between orders (86.16 days), indicating higher engagement and potential for upselling.

  - Home Office customers have the longest average time between orders (89.70 days), suggesting less frequent purchases and an opportunity to improve engagement.


Actionable Strategies :
* Leverage High Retention for Growth:
  - Introduce loyalty programs or subscription models to further engage retained customers.
  - Upsell/cross-sell higher-margin products to loyal customers, especially Corporate customers.

* Increase New Customer Acquisition:
  - Invest in marketing channels (e.g., social media ads, influencer partnerships) to attract new customers.
  - Launch referral programs to incentivize existing customers to bring in new ones.

* Target Customer Segments Differently:
  - Corporate Customers: Introduce subscription plans and exclusive perks to increase engagement and shorten reorder cycles
  - Consumer Customers :  Launch a loyalty program and personalized recommendations to drive repeat purchases.
  - Home Office Customers: Improve engagement with targeted campaigns to reduce the time between orders.

* Capitalize on Seasonal Trends: Plan promotions and campaigns around the second half of the year when repeat orders spike.

##### Average Order Value (AOV)
"""

# Calculate Average Order Value per Year
total_sales_orders = df.groupby('year_od').agg(
    total_sales=('sales', 'sum'),
    total_orders=('order_id', 'nunique'),
).reset_index()

total_sales_orders['AOV'] = total_sales_orders['total_sales'] / total_sales_orders['total_orders']
total_sales_orders

# Calculate AOV by customer segment
aov_by_segment = df.groupby('segment').agg(
    total_sales=('sales', 'sum'),
    total_orders=('order_id', 'nunique')
).reset_index()
aov_by_segment['AOV'] = aov_by_segment['total_sales'] / aov_by_segment['total_orders']

plt.figure(figsize=(12, 6))
sns.barplot(data=aov_by_segment, x='segment', y='AOV')
plt.title('Average Order Value (AOV) by Customer Segment', fontsize=16)
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Average Order Value (AOV)', fontsize=12)
plt.show()

"""Key Insights:

* The most significant jump in sales occurred between 2016 and 2017, with an increase of approximately 140,756.55.
* Total orders steadily increased every year, suggesting higher order frequency.
* The AOV (calculated as total sales divided by total orders) shows a decreasing trend from 2015 to 2018.
This suggests that while more orders are being placed, the average amount spent per order has been declining. This could be due to discounts, smaller order sizes, or changes in product pricing.
* Business Growth with Reduced Per-Order Value:
The business is growing in terms of total sales and orders, but the reduced AOV might require attention.
* The AOV for Consumer, Corporate, and Home Office segments are relatively close, with Home Office having the highest AOV. This suggests that while all segments contribute similarly, there may be untapped opportunities for increasing AOV in some segments.

Actionable Strategies :
  - Focus on value-added incentives like free shipping thresholds or product bundles.
  - Targeted Promotions for Home Office Segment:
    - Introduce high-value product bundles or loyalty incentives for repeat purchases.
    - Offer personalized marketing campaigns focusing on premium or productivity-enhancing products.
    
  - Corporate Bulk Purchase Incentives:
    - Provide dedicated account managers for personalized assistance and upselling opportunities.
    - Encourage bulk purchases with tiered discounts and bundled deals to maximize order value.

  - Consumer Segment Upsell Strategy:
    -  Increase cart size with cross-selling, free shipping thresholds, and limited-time bundles.

##### Customer Segmentation
"""

cust_segment = df.groupby(['segment']).agg(
    sales=('sales', 'sum'),
    number_of_cust=('customer_id', 'nunique')
).reset_index()
cust_segment['average_sales_cust'] = (cust_segment['sales'] / cust_segment['number_of_cust'])

# Plot sales, number of customers, and average sales per customer by segment
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
colors = ['skyblue', 'lightgreen', 'orange']

# Segment Contribution in Sales
ax1.bar(cust_segment['segment'], cust_segment['sales'], color=colors, alpha=0.6)
ax1.set_title('Segment Contribution in Sales', fontsize=14)
ax1.set_xlabel('Segment', fontsize=12)
ax1.set_ylabel('Sales', fontsize=12)

# Number of Customers by Segment
ax2.bar(cust_segment['segment'], cust_segment['number_of_cust'], color=colors, alpha=0.6)
ax2.set_title('Number of Customers by Segment', fontsize=14)
ax2.set_xlabel('Segment', fontsize=12)
ax2.set_ylabel('Number of Customers', fontsize=12)

# Average Sales per Customer by Segment
ax3.bar(cust_segment['segment'], cust_segment['average_sales_cust'], color=colors, alpha=0.6)
ax3.set_title('Average Sales per Customer by Segment', fontsize=14)
ax3.set_xlabel('Segment', fontsize=12)
ax3.set_ylabel('Average Sales per Customer', fontsize=12)

plt.subplots_adjust(wspace=0.3)
plt.show()

region_segment = df.groupby(['region', 'segment'])['sales'].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=region_segment, x='region', y='sales', hue='segment', ax=ax, palette='viridis')
ax.set_title('Total Sales by Region and Segment', fontsize=16)
ax.set_xlabel('Region', fontsize=12)
ax.set_ylabel('Total Sales', fontsize=12)
plt.title('Total Sales by Region and Segment')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.legend(title='Segment')
plt.show()

"""Key Insights
* Consumer Segment Dominance:
  - The Consumer segment has the highest number of customers and contributes the most to total sales, confirming that the business's primary market is individual consumers.
  - Despite its dominance, the Consumer segment shows the lowest average sales per customer, highlighting an opportunity for growth through upselling or cross-selling.

* Corporate Segment Value:
  - The Corporate segment has significantly fewer customers compared to the Consumer segment but contributes relatively high total sales.
  - With one of the highest average sales per customer, Corporate customers tend to place larger orders, making this a valuable segment despite its smaller customer base.

* Home Office Segment Potential:
  - The Home Office segment has the smallest customer base and the lowest total sales contribution. However, its average sales per customer are comparable to the Corporate segment, indicating that Home Office customers place high-value orders.
  - This suggests untapped potential if more customers can be attracted to this segment.

* Regional Sales Distribution:
  - The Consumer segment consistently generates the highest sales across all regions, particularly in the East and West, where sales significantly outperform other segments.
  - The South region shows relatively lower sales across all segments, indicating a potential growth opportunity.

Actionable Strategies:
* Consumer Segment:
  - Focus on increasing the average sales per customer by implementing personalized upselling and cross-selling strategies.
  - Introduce loyalty programs to incentivize repeat purchases and boost order values.
* Corporate Segment:
  - Invest in customer acquisition to expand this high-value segment.
  - Strengthen customer relationship management (CRM) strategies by offering personalized services and exclusive business packages to retain and engage corporate clients.
* Home Office Segment:
  - Attract more Home Office customers through targeted marketing campaigns.
  - Develop tailored product bundles and offer value-added services like setup assistance to maintain high average order values.

### Sales Trends

##### Sales Trends
"""

# Sales Trends by Year
sales_year = df.groupby('year_od')['sales'].sum().reset_index()
sales_year['year_od'] = sales_year['year_od'].astype(str)

plt.figure(figsize=(10, 6))
plt.plot(sales_year['year_od'], sales_year['sales'], marker='o')

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales Trends by Year')

plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Sales Trends by Monthly
sales_month_year = df.groupby('month_year_od')['sales'].sum().reset_index()
sales_month_year['month_year_od'] = sales_month_year['month_year_od'].astype(str)

plt.figure(figsize=(20, 6))
plt.plot(sales_month_year['month_year_od'], sales_month_year['sales'], marker='o', label='Monthly Sales')

for i, date in enumerate(sales_month_year['month_year_od']):
    if date.endswith('-01'):  # Year divider
        plt.axvline(x=i, color='red', linestyle='--', linewidth=1, label='Year Divider' if i == 0 else None)
    elif date[-2:] in ['04', '07', '10']:  # Quarter divider
        plt.axvline(x=i, color='blue', linestyle='--', linewidth=1, label='Quarter Divider' if i == 3 else None)

xticks = [i for i, date in enumerate(sales_month_year['month_year_od']) if date.endswith('-01')]
xtick_labels = [date for date in sales_month_year['month_year_od'] if date.endswith('-01')]
plt.xticks(xticks, xtick_labels, rotation=45)

plt.title('Monthly Sales Trends with Yearly and Quarterly Dividers', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Sales Trends by Year-Quarter
sales_year_quarter = df.groupby('year_quarter')['sales'].sum().reset_index()

plt.figure(figsize=(20, 6))
plt.plot(sales_year_quarter['year_quarter'], sales_year_quarter['sales'], marker='o', label='Quarterly Sales')

for i, row in sales_year_quarter.iterrows():
    if row['year_quarter'].endswith('Q4'):
        plt.annotate(f"Q4 Peak: ${row['sales']:,.2f}",
                     (i, row['sales']),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     fontsize=10,
                     color='green')

plt.title('Quarterly Sales Trends', fontsize=16)
plt.xlabel('Year-Quarter', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

"""Key Insights:
* Sales rebounded significantly after a dip in 2016, demonstrating strong recovery in 2017 and continued growth in 2018.
* The upward trend from 2016 to 2018 reflects effective business strategies and market adaptation.
* Q4 consistently records the highest sales each year, with Q4 2018 achieving the peak performance.
* The second half of the year (Q3–Q4) contributes the majority of annual revenue, driven by strong Q4 results.
* Q1 consistently experiences the lowest sales, indicating a recurring post-holiday slowdown.
* Sales demonstrate a steady recovery from Q1, improving through Q2 and Q3 before peaking in Q4.
* Seasonal sales volatility is observed, with fluctuations occurring before stabilization in Q3.

Actionable Strategies:
* Address mid-year fluctuations to enhance revenue predictability and operational planning.
* Ramp up marketing efforts in Q3 to maximize Q4 performance.
Optimize inventory management to align with seasonal demand trends.
* Implement targeted post-holiday promotions and product launches in Q1 to mitigate early-year declines.
* Develop customer retention programs to maintain engagement and boost sales in historically weaker months.
* Focus on sustaining a predictable growth trajectory by optimizing Q4 performance, addressing Q1 slowdowns, and minimizing mid-year volatility.

##### Product Insight
"""

# Product Category Analysis
product_category = df.groupby('category').agg(
    sales=('sales', 'sum'),
    number_of_order=('order_id', 'nunique')
).reset_index()
product_category = product_category.sort_values(by='sales', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Sales by Category
ax1.bar(product_category['category'], product_category['sales'], color='skyblue', alpha=0.6)
ax1.set_title('Sales by Product Category', fontsize=14)
ax1.set_xlabel('Product Category', fontsize=12)
ax1.set_ylabel('Total Sales', fontsize=12)

for i, sales in enumerate(product_category['sales']):
    ax1.text(i, sales, f"${sales:,.2f}", ha='center', va='bottom', fontsize=10)

# Number of Orders by Category
ax2.bar(product_category['category'], product_category['number_of_order'], color='lightgreen', alpha=0.6)
ax2.set_title('Number of Orders by Product Category', fontsize=14)
ax2.set_xlabel('Product Category', fontsize=12)
ax2.set_ylabel('Number of Orders', fontsize=12)

for i, orders in enumerate(product_category['number_of_order']):
    ax2.text(i, orders, f"{orders:,}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Product Sub-Category Analysis
product_sub_category = df.groupby('sub_category')['sales'].sum().reset_index()
product_sub_category = product_sub_category.sort_values(by='sales', ascending=True)
product_sub_category_mean = product_sub_category['sales'].mean()

plt.figure(figsize=(12, 8))
plt.barh(product_sub_category['sub_category'], product_sub_category['sales'], color='skyblue', alpha=0.6)
plt.axvline(x=product_sub_category_mean, color='red', linestyle='--', linewidth=1, label='Average Sales by Sub-Category')
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Product Sub-Category', fontsize=12)
plt.title('Sales by Product Sub-Category', fontsize=16)

for i, sales in enumerate(product_sub_category['sales']):
    if sales > product_sub_category_mean:
        plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=10, color='black')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Top Selling Products
top_selling_product = df.groupby('product_name')['sales'].sum().nlargest(10).iloc[::-1].reset_index()

plt.figure(figsize=(20, 6))
plt.barh(top_selling_product['product_name'], top_selling_product['sales'], color='skyblue', alpha=0.6)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.title('Top 10 Performing Products by Sales', fontsize=16)

for i, sales in enumerate(top_selling_product['sales']):
    plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Least Selling Products
least_selling_product = df.groupby('product_name')['sales'].sum().nsmallest(10).iloc[::-1].reset_index()

plt.figure(figsize=(20, 6))
plt.barh(least_selling_product['product_name'], least_selling_product['sales'], color='lightcoral', alpha=0.6)
plt.xlabel('Total Sales', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.title('Bottom 10 Performing Products by Sales', fontsize=16)

for i, sales in enumerate(least_selling_product['sales']):
    plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

"""Key Insight:
* The Technology category has the highest sales but the lowest number of orders, indicating higher-priced items.
* The Office Supplies segment has the highest number of orders but contributes less to total sales, suggesting lower-priced items.
* Increasing the order volume in Technology or the average price per order in Office Supplies could significantly boost revenue.
- The least-selling products are primarily from the **Office Supplies** category, indicating low demand. Consider discontinuing or rebranding these products to improve sales performance.

##### Regional Sales
"""

# Sales by Region
region_sales = df.groupby('region')['sales'].sum().reset_index()
region_sales = region_sales.sort_values(by='sales', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(region_sales['region'], region_sales['sales'], color='skyblue', alpha=0.6)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales by Region', fontsize=16)

# Add annotations
for i, sales in enumerate(region_sales['sales']):
    plt.text(i, sales, f"${sales:,.2f}", ha='center', va='bottom', fontsize=10)

plt.show()

# Sales by State
state_sales = df.groupby('state')['sales'].sum().reset_index()
state_sales = state_sales.sort_values(by='sales', ascending=True)
state_sales_mean = state_sales['sales'].mean()

plt.figure(figsize=(8, 10))
plt.barh(state_sales['state'], state_sales['sales'], color='lightgreen', alpha=0.6)
plt.axvline(x=state_sales_mean, color='red', linestyle='--', linewidth=1, label='Average Sales by State')
plt.xlabel('Sales', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.title('Sales by State', fontsize=16)
plt.yticks(fontsize=7)

# Add annotations
for i, sales in enumerate(state_sales['sales']):
  if sales > state_sales_mean:  # Only annotate above-average values
    plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=7)

plt.legend()
plt.show()

# Top 10 Cities by Sales
top_city_sales = df.groupby('city')['sales'].sum().nlargest(10).iloc[::-1].reset_index()

plt.figure(figsize=(10, 6))
plt.barh(top_city_sales['city'], top_city_sales['sales'], color='skyblue', alpha=0.6)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.title('Top 10 Performing Cities by Sales', fontsize=16)

for i, sales in enumerate(top_city_sales['sales']):
    plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=10)

plt.show()

# Bottom 10 Cities by Sales
least_city_sales = df.groupby('city')['sales'].sum().nsmallest(10).iloc[::-1].reset_index()

plt.figure(figsize=(10, 6))
plt.barh(least_city_sales['city'], least_city_sales['sales'], color='lightcoral', alpha=0.6)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('City', fontsize=12)
plt.title('Bottom 10 Performing Cities by Sales', fontsize=16)

for i, sales in enumerate(least_city_sales['sales']):
    plt.text(sales, i, f"${sales:,.2f}", va='center', fontsize=10)

plt.show()

"""### Operational Performance



"""

df['date_diff'] = (df['ship_date'] - df['order_date']).dt.days
avg_delivery_time = df['date_diff'].mean()

print(f'Average delivery time is {avg_delivery_time:.2f} days')

# Average Delivery Time by Region
region_delivery_time = df.groupby('region')['date_diff'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(region_delivery_time['region'], region_delivery_time['date_diff'], color='skyblue', alpha=0.6)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Average Delivery Time (Days)', fontsize=12)
plt.title('Average Delivery Time by Region', fontsize=16)

# Add annotations
for i, days in enumerate(region_delivery_time['date_diff']):
    plt.text(i, days, f"{days:.2f} days", ha='center', va='bottom', fontsize=10)

plt.show()

# Shipping Mode Analysis
ship_mode_delivery = df.groupby('ship_mode').agg(
    avg_delivery_time=('date_diff', 'mean'),
    sales=('sales', 'sum')
).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Average Delivery Time by Shipping Mode
ax1.bar(ship_mode_delivery['ship_mode'], ship_mode_delivery['avg_delivery_time'], color='skyblue', alpha=0.6)
ax1.set_title('Average Delivery Time by Shipping Mode', fontsize=14)
ax1.set_xlabel('Shipping Mode', fontsize=12)
ax1.set_ylabel('Average Delivery Time (Days)', fontsize=12)

for i, days in enumerate(ship_mode_delivery['avg_delivery_time']):
    ax1.text(i, days, f"{days:.2f} days", ha='center', va='bottom', fontsize=10)

# Total Sales by Shipping Mode
ax2.bar(ship_mode_delivery['ship_mode'], ship_mode_delivery['sales'], color='lightgreen', alpha=0.6)
ax2.set_title('Total Sales by Shipping Mode', fontsize=14)
ax2.set_xlabel('Shipping Mode', fontsize=12)
ax2.set_ylabel('Total Sales', fontsize=12)

for i, sales in enumerate(ship_mode_delivery['sales']):
    ax2.text(i, sales, f"${sales:,.2f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Shipping Mode Preferences by Region
region_ship_mode = df.groupby(['region', 'ship_mode'])['order_id'].nunique().reset_index()
region_ship_mode = region_ship_mode.rename(columns={'order_id': 'number_of_orders'})

plt.figure(figsize=(12, 6))
sns.barplot(data=region_ship_mode, x='region', y='number_of_orders', hue='ship_mode', palette='viridis')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.title('Shipping Mode Preferences by Region', fontsize=16)
plt.legend(title='Shipping Mode')
plt.show()

# Shipping Mode Preferences by Segment
segment_ship_mode = df.groupby(['segment', 'ship_mode'])['order_id'].nunique().reset_index()
segment_ship_mode = segment_ship_mode.rename(columns={'order_id': 'number_of_orders'})

plt.figure(figsize=(12, 6))
sns.barplot(data=segment_ship_mode, x='segment', y='number_of_orders', hue='ship_mode', palette='viridis')
plt.xlabel('Customer Segment', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.title('Shipping Mode Preferences by Customer Segment', fontsize=16)
plt.legend(title='Shipping Mode')
plt.show()

"""Key Insights:
* The overall average delivery time is 3.96 days, indicating relatively stable logistics performance.
* Delivery Performance Across Regions
  - Central region has the longest delivery time (4.07 days), suggesting possible inefficiencies or delays in fulfillment.
  - West region has the shortest delivery time (3.93 days), but the difference between regions is minimal (~0.14 days), indicating operational consistency.
* Shipping Mode Preferences
  - Standard Class dominates in both order volume and sales, showing that most customers prioritize affordability over speed.
  - Same-Day shipping has the fastest delivery time (0.04 days) but very low adoption, suggesting either lack of awareness, high costs, or limited availability.
  - First Class (2.18 days) and Second Class (3.25 days) offer faster delivery than Standard Class but are not as widely adopted, indicating price sensitivity.
* Regional Order Distribution
  - West and East regions generate the highest order volumes, suggesting these areas are key revenue drivers.
  - South has the lowest order volume, presenting a potential growth opportunity.
* Customer Segmentation
  - The "Consumer" segment is the primary driver of all shipping modes, especially Standard Class, indicating strong B2C demand.
  - Coorporate customers have low adoption of premium shipping options, potentially due to alternative logistics solutions or negotiated shipping terms.
* There is no clear correlation between faster shipping and higher order volumes, suggesting that speed alone may not be a primary factor in customer purchasing decisions.

Actionable Strategies:
* Optimize Standard Class logistics to maintain its affordability while reducing delivery time further.
* Explore pricing strategies for premium shipping options (First & Second Class) to improve adoption without significantly impacting margins.
* Develop Coorporate-specific shipping incentives to encourage enterprises to adopt faster shipping options.

"""

