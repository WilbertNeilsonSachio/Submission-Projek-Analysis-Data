import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Fungsi Memanggil all_data


@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'all_data.csv')
    data = pd.read_csv(file_path, parse_dates=['dteday'])
    return data


data = load_data()

# Sidebar
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", data['dteday'].min())
end_date = st.sidebar.date_input("End Date", data['dteday'].max())

filtered_data = data[(data['dteday'] >= pd.to_datetime(start_date)) & (
    data['dteday'] <= pd.to_datetime(end_date))]

st.title("📊 Bike Rentals Dashboard")

st.subheader("Average Bike Rentals by Season and Weather")
season_weather_group = filtered_data.groupby(['season', 'weather_condition']).agg({'total': 'mean'}).reset_index()
season_weather_pivot = season_weather_group.pivot(index='season', columns='weather_condition', values='total')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(season_weather_pivot, annot=True, cmap='coolwarm', fmt='.1f')
# plt.title('Average Bike Rentals by Season and Weather')
plt.xlabel('Weather Situation')
plt.ylabel('Season')
st.pyplot(fig)

st.subheader("Average Bike Rentals by Month")
monthly_rentals = filtered_data.groupby(['year', 'month']).agg({'total': 'mean'}).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='month', y='total', hue='year', data=monthly_rentals, marker='o', palette='coolwarm')
# plt.title('Average Bike Rentals by Month')
plt.xlabel('Month')
plt.ylabel('Average Rentals')
plt.xticks(range(1, 13))
plt.grid(True)
st.pyplot(fig)

st.subheader("Average Bike Rentals by Hour")
hourly_rentals = filtered_data.groupby('hour')['total'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='hour', y='total', data=hourly_rentals, marker='o')
# plt.title('Average Bike Rentals by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Rentals')
plt.xticks(range(0, 24))
plt.grid(True)
st.pyplot(fig)

st.write("**Explore different date ranges using the sidebar filter!**")
