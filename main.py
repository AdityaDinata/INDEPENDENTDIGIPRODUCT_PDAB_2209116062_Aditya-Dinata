import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split

# Nonaktifkan peringatan
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    url = "Data Cleaning.csv"
    df = pd.read_csv(url)
    return df

def main():
    st.title('Energy Consumption Prediction')

    st.markdown("---")

    # Load data
    df = load_data()
    # Konversi kolom 'Timestamp' menjadi tipe datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Pilihan halaman di sidebar menggunakan option_menu
    with st.sidebar:
        st.title("Selamat Datang Di Website Prediksi Konsumsi Energi")
        st.title("Pilih Halaman")
        page_option = option_menu("", ["Informasi Dasar Dataset Yang Digunakan", "Visualisasi", "Predict Energy Consumption"])

    if page_option == "Informasi Dasar Dataset Yang Digunakan":
        st.markdown('---')
        st.markdown('# Informasi Dasar Dataset Yang Digunakan')
        st.write(df)
        st.write("""
        - **Timestamp**: Menunjukkan tanggal dan waktu ketika data direkam.
        - **Temperature**: Mewakili suhu di lokasi pengumpulan data, dalam derajat Celsius.
        - **Humidity**: Menunjukkan kelembaban relatif di lokasi pengumpulan data.
        - **Occupancy**: Menunjukkan jumlah penghuni di lokasi selama periode waktu yang direkam.
        - **HVACUsage**: Mengindikasikan penggunaan atau status operasi sistem HVAC selama periode waktu yang direkam, sering menunjukkan apakah sistem tersebut aktif atau tidak.
        - **LightingUsage**: Menunjukkan penggunaan atau status operasi sistem pencahayaan selama periode waktu yang direkam, biasanya menunjukkan apakah lampu menyala atau mati.
        - **RenewableEnergy**: Mewakili jumlah energi yang dihasilkan dari sumber energi terbarukan, seperti surya atau angin, selama periode waktu yang direkam.
        - **DayOfWeek**: Menunjukkan hari dalam seminggu yang sesuai dengan timestamp yang direkam, biasanya direpresentasikan sebagai bilangan bulat (misalnya, 1 untuk Minggu, 2 untuk Senin, dst).
        - **Holiday**: Indikator biner (0 atau 1) yang menunjukkan apakah timestamp yang direkam jatuh pada hari libur atau tidak.
        - **EnergyConsumption**: Mencerminkan total konsumsi energi selama periode waktu yang direkam, diukur dalam kilowatt-hour (kWh).
        """)


    elif page_option == "Visualisasi":
        st.markdown('---')
        st.markdown('# Visualisasi')
        # Tambahkan pilihan visualisasi di sini

        pilihan_visualisasi = st.selectbox('Pilih Visualisasi', ['Scatter Plot Suhu vs Konsumsi Energi', 'Heatmap Korelasi', 'Line Plot Konsumsi Energi per Jam', 'Line Plot Trend Konsumsi Energi', 'Grafik Temperatur Harian dan Rata-rata'])

        if pilihan_visualisasi == 'Scatter Plot Suhu vs Konsumsi Energi':
            st.subheader('Scatter Plot Suhu vs Konsumsi Energi')
            fig = px.scatter(df, x='Temperature', y='EnergyConsumption', title='Scatter Plot Suhu vs Konsumsi Energi')
            st.plotly_chart(fig)
            st.write ('Diagram Scatter Plot menunjukkan hubungan positif antara suhu dan konsumsi energi untuk skuter. Seiring bertambahnya suhu, konsumsi energi juga meningkat.')

        elif pilihan_visualisasi == 'Heatmap Korelasi':
            st.subheader('Heatmap Korelasi')
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot()
            st.write('''Dalam grafik ini, semua koefisien korelasi relatif kecil, menunjukkan tidak adanya hubungan yang kuat antara variabel mana pun. Namun, ada beberapa tren penting.
            Suhu dan konsumsi energi memiliki korelasi positif sebesar 0,7. Ini berarti bahwa ketika suhu meningkat, konsumsi energi juga meningkat. Hal ini kemungkinan karena orang menggunakan lebih banyak energi untuk mendinginkan rumah mereka pada saat cuaca panas.
            ''')

        elif pilihan_visualisasi == 'Line Plot Konsumsi Energi per Jam':
            st.subheader('Line Plot Konsumsi Energi per Jam')
            df['Hour'] = df['Timestamp'].dt.hour
            df_hourly_energy = df.groupby('Hour')['EnergyConsumption'].mean()
            plt.figure(figsize=(10, 6))
            plt.plot(df_hourly_energy.index, df_hourly_energy.values, marker='o', linestyle='-')
            plt.xlabel('Jam')
            plt.ylabel('Konsumsi Energi (kWh)')
            plt.title('Konsumsi Energi per Jam')
            st.pyplot()
            st.write(''' Garis plot menunjukkan tren konsumsi energi yang meningkat dari jam 0 hingga jam 15, kemudian menurun hingga jam 20. Nilai konsumsi energi terendah adalah 75 kWh pada jam 0, dan nilai tertinggi adalah 79 kWh pada jam 15.''')

        elif pilihan_visualisasi == 'Line Plot Trend Konsumsi Energi':
            st.subheader('Line Plot Trend Konsumsi Energi')
            df_daily_energy = df.resample('D', on='Timestamp')['EnergyConsumption'].mean()
            plt.figure(figsize=(10, 6))
            plt.plot(df_daily_energy.index, df_daily_energy.values, marker='o', linestyle='-')
            plt.xlabel('Tanggal')
            plt.ylabel('Rata-rata Konsumsi Energi (kWh)')
            plt.title('Trend Konsumsi Energi')
            st.pyplot()
            st.write('Grafik garis menunjukkan tren konsumsi energi yang meningkat selama periode pengukuran. Rata rata konsumsi energi harian tertinggi terdapat pada tanggal 2022-02-05 dan rata rata konsumsi energi harian terendah terdapat pada tanggal 2022-01-03.')

        elif pilihan_visualisasi == 'Grafik Temperatur Harian dan Rata-rata':
            st.subheader('Grafik Temperatur Harian dan Rata-rata')
            # Kelompokkan data berdasarkan hari dan hitung rata-rata temperatur per hari
            df_daily_temperature = df.resample('D', on='Timestamp')['Temperature'].mean()

            # Hitung rata-rata temperatur secara keseluruhan
            average_temperature = df['Temperature'].mean()

            # Buat grafik line plot
            plt.figure(figsize=(12, 6))
            plt.plot(df_daily_temperature.index, df_daily_temperature.values, marker='o', linestyle='-', label='Temperatur Harian')
            plt.axhline(average_temperature, color='r', linestyle='--', label='Rata-rata Temperatur')
            plt.title('Grafik Temperatur Harian dan Rata-rata')
            plt.xlabel('Tanggal')
            plt.ylabel('Temperatur')
            plt.legend()
            plt.grid(True)
            st.pyplot()
            st.write('Bisa dilihat untuk grafik Line Plot diatas bahwa rata rata temperatur harian dari tanggal 2022-01-01 sampai 2022-02-11 adalah 25°C.')






    elif page_option == "Predict Energy Consumption":
        st.markdown('---')
        st.subheader("Prediksi Konsumsi Energi Berdasarkan Suhu")

        # Menggunakan kolom Temperature sebagai fitur dan EnergyConsumption sebagai target
        X = df[['Temperature']]
        y = df['EnergyConsumption']

        # Pisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Membuat pipeline dengan normalisasi menggunakan MinMaxScaler dan model regresi linear
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),  # Normalisasi
            ('regressor', LinearRegression())  # Model Regresi Linear
        ])

        # Latih model menggunakan data latih
        pipeline.fit(X_train, y_train)

        # Meminta pengguna untuk memasukkan nilai suhu
        temperature_input = st.slider("Masukkan suhu (dalam derajat Celsius): ", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

        # Peringatan jika nilai suhu melebihi 50 derajat Celsius
        if temperature_input > 50:
            st.warning("Nilai suhu melebihi 50 derajat Celsius.")

        # Fungsi untuk melakukan prediksi berdasarkan suhu
        def predict_energy_consumption(temperature):
            # Membuat DataFrame dengan satu baris yang berisi nilai suhu yang dimasukkan pengguna
            input_data = pd.DataFrame({'Temperature': [temperature]})

            # Melakukan prediksi menggunakan pipeline yang telah kita buat
            energy_prediction = pipeline.predict(input_data)

            # Mengembalikan prediksi konsumsi energi
            return energy_prediction[0]

        # Melakukan prediksi berdasarkan suhu yang dimasukkan pengguna
        predicted_energy = predict_energy_consumption(temperature_input)

        # Menampilkan prediksi konsumsi energi kepada pengguna
        st.write(f"Prediksi konsumsi energi untuk suhu {temperature_input} °C adalah {predicted_energy:.2f} kWh")

        # Visualisasi konsumsi energi berdasarkan suhu
        st.write("Bar chart menunjukkan konsumsi energi berdasarkan suhu yang dimasukkan pengguna.")
        
        # Memperbarui data latih dengan nilai suhu yang dimasukkan pengguna
        X_train_updated = pd.DataFrame({'Temperature': [temperature_input]})
        y_train_updated = predict_energy_consumption(temperature_input)
        
        # Buat peta warna dari biru ke merah (coolwarm), dengan transisi menjadi ungu jika melebihi 50 derajat Celsius
        colors = plt.cm.coolwarm(np.linspace(0, 1, 101))
        # Tentukan pembagian warna untuk batas 50 derajat Celsius
        num_colors_below_50 = int(50 * (len(colors) - 1) / 100)
        num_colors_above_50 = len(colors) - 1 - num_colors_below_50
        # Jika nilai suhu di bawah 50, gunakan pembagian warna dari biru ke merah
        if temperature_input <= 50:
            color_index = int(temperature_input * num_colors_below_50 / 50)
        # Jika nilai suhu di atas 50, gunakan transisi warna menjadi ungu
        else:
            color_index = num_colors_below_50 + int((temperature_input - 50) * num_colors_above_50 / 50)

        # Buat bar chart dengan warna batang yang sesuai
        fig, ax = plt.subplots()
        bar = ax.bar(temperature_input, predicted_energy, color=colors[color_index])  # Warna berdasarkan suhu
        ax.set_xlabel('Suhu (°C)')
        ax.set_ylabel('Konsumsi Energi (kWh)')
        ax.set_title('Konsumsi Energi Berdasarkan Suhu')

        st.pyplot(fig)



# Pemanggilan fungsi main()
if __name__ == "__main__":
    main()
