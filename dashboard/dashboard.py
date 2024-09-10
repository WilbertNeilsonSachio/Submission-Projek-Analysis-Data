import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi Memanggil cleaned_data
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'cleaned_data.csv')
    data = pd.read_csv(file_path)
    return data

data_cleaned = load_data()

# Sidebar
st.sidebar.header('Pengaturan Dashboard')
option = st.sidebar.selectbox(
    'Pilih visualisasi:',
    ['Visualisasi Pengaruh Cuaca dan Musim', 'Tren Penyewaan Sepeda per Jam', 'Analisis Regresi']
)

# Visualization: Pengaruh Cuaca dan Musim
if option == 'Visualisasi Pengaruh Cuaca dan Musim':
    st.title('Pengaruh Cuaca dan Musim terhadap Penyewaan Sepeda')
    season_weather_group = data_cleaned.groupby(['season', 'weathersit'], observed=True).agg({'cnt': 'mean'}).reset_index()
    season_weather_pivot = season_weather_group.pivot(index='season', columns='weathersit', values='cnt')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(season_weather_pivot, annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Rata-rata Penyewaan Sepeda berdasarkan Musim dan Cuaca')
    plt.xlabel('Situasi Cuaca')
    plt.ylabel('Musim')
    st.pyplot(plt)

# Visualization: Tren Penyewaan Sepeda per Jam
elif option == 'Tren Penyewaan Sepeda per Jam':
    st.title('Tren Penyewaan Sepeda Berdasarkan Jam dalam Sehari')
    hourly_rentals = data_cleaned.groupby('hr').agg({'cnt': 'mean'}).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hr', y='cnt', data=hourly_rentals, marker='o')
    plt.title('Rata-rata Penyewaan Sepeda berdasarkan Jam dalam Sehari')
    plt.xlabel('Jam dalam Sehari')
    plt.ylabel('Rata-rata Penyewaan Sepeda')
    plt.xticks(range(0, 24))
    plt.grid(True)
    st.pyplot(plt)

# Analisis regresi
elif option == 'Analisis Regresi':
    st.title('Analisis Regresi Penyewaan Sepeda')
    
    X = data_cleaned[['season', 'weathersit', 'temp', 'hum', 'windspeed']]
    y = data_cleaned['cnt']
    
    X = pd.get_dummies(X, columns=['season', 'weathersit'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R Squared: {r2:.4f}")
