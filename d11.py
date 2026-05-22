import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Настройка страницы
st.set_page_config(page_title="AI City Sustainability Dashboard", layout="wide")

# Базовый словарь координат крупных городов для стабильной работы карты без задержек API
CITY_COORDS = {
    'Москва': [55.7558, 37.6173], 'Санкт-Петербург': [59.9343, 30.3351],
    'Екатеринбург': [56.8389, 60.6057], 'Новосибирск': [55.0084, 82.9357],
    'Казань': [55.7887, 49.1221], 'Нижний Новгород': [56.3287, 44.0020],
    'Челябинск': [55.1644, 61.4368], 'Самара': [53.2001, 50.1500],
    'Омск': [54.9885, 73.3242], 'Ростов-на-Дону': [47.2357, 39.7015],
    'Ростов-На-Дону': [47.2357, 39.7015], 'Уфа': [54.7388, 55.9721],
    'Красноярск': [56.0153, 92.8932], 'Пермь': [58.0105, 56.2502],
    'Воронеж': [51.6608, 39.2003], 'Волгоград': [48.7080, 44.5133],
    'Балашиха': [55.7963, 37.9381], 'Барнаул': [53.3548, 83.7698],
    'Тольятти': [53.5078, 49.4111], 'Томск': [56.4977, 84.9744],
    'Рязань': [54.6095, 39.7126], 'Абакан': [53.7156, 91.4292],
    'Азов': [47.1114, 39.4236]
}

# --- 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
@st.cache_data
def load_data():
    # Загрузка исторического датасета
    df = pd.read_csv('datasetitog.csv', sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Чистка названий городов
    df['city'] = df['city'].astype(str).str.strip().str.replace('г.', '', regex=False).str.replace('г ', '', regex=False).str.strip()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['data_type'] = 'История'
    
    # Принудительная конвертация всех характеристик в числа (защита от пустых значений и текста)
    exclude_cols = ['city', 'year', 'data_type', 'pred_lr', 'pred_rf']
    for col in df.columns:
        if col not in exclude_cols and not col.startswith('unnamed'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Создаем интегральный индекс устойчивости
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    service_cols = ['year', 'pred_lr', 'pred_rf']
    metric_cols = [c for c in numeric_cols if c not in service_cols]

    if metric_cols:
        normalized = df[metric_cols].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        df['env_score'] = normalized.mean(axis=1)
    else:
        df['env_score'] = 0.5
        
    # Интеграция прогнозных значений 2025, если файл существует
    if os.path.exists('forecast_2025.csv'):
        df_forecast = pd.read_csv('forecast_2025.csv', encoding='utf-8')
        df_forecast.columns = df_forecast.columns.str.strip().str.lower().str.replace(' ', '_')
        df_forecast['city'] = df_forecast['city'].astype(str).str.strip().str.replace('г.', '', regex=False).str.replace('г ', '', regex=False).str.strip()
        df_forecast['year'] = pd.to_numeric(df_forecast['year'], errors='coerce')
        df_forecast['data_type'] = 'Прогноз'
        
        if 'env_score' not in df_forecast.columns and 'pred_rf' in df_forecast.columns:
            df_forecast['env_score'] = df_forecast['pred_rf']
            
        for col in df_forecast.columns:
            if col not in exclude_cols and not col.startswith('unnamed'):
                df_forecast[col] = pd.to_numeric(df_forecast[col], errors='coerce')
        
        df = pd.concat([df, df_forecast], ignore_index=True)
        
    return df

df = load_data()

# Доступные для анализа числовые колонки
exclude_cols = ['city', 'year', 'data_type', 'pred_lr', 'pred_rf']
feature_columns = [c for c in df.columns if c not in exclude_cols and not c.startswith('unnamed')]

# Словарь для красивого отображения названий в интерфейсе
readable_metrics = {
    'env_score': '🌍 Общий индекс устойчивости',
    'people': '👥 Население',
    'net_salary': '💵 Чистая зарплата',
    'housing_price': '🏠 Цена за жилье',
    'poverty_level': '📉 Уровень бедности',
    'air_general_level': '🌬️ Индекс загрязнения воздуха',
    'avg_age': '⏳ Средний возраст',
    'birth': '👶 Рождаемость',
    'death': '⚰️ Смертность',
    'crimes': '🚨 Количество преступлений',
    'criminals': '🚔 Число преступников',
    'pens': '👴 Пенсионеры',
    'preschool_child': '🧒 Дошкольники',
    'ilm': '🏭 Промышленность',
    'wage': '💰 Средняя зарплата',
    'workers': '👷 Работники',
    'length_of_roads': '🛣️ Дороги',
    'migration_net': '🧳 Миграционный прирост',
    'housing_stock': '🏢 Жилой фонд',
    'energy_consumption': '⚡ Потребление энергии'
}

# --- ИНТЕРФЕЙС И ТАБЫ ---
st.title("🏙️ Комплексный анализ и прогнозирование устойчивости городов РФ")
st.markdown("---")

t1, t2, t3 = st.tabs(["🗺️ Геоинформационная карта", "📊 Аналитика и Рейтинги", "🔮 Интерактивный прогноз 2025"])

# ==================== ТАБ 1: КАРТА С ВЫБОРОМ ХАРАКТЕРИСТИК ====================
with t1:
    st.subheader("📍 Интерактивная ГИС-карта показателей")
    
    map_metric = st.selectbox(
        "Выберите характеристику для отображения на карте:", 
        options=feature_columns, 
        format_func=lambda x: readable_metrics.get(x, x)
    )
    
    # Фильтруем пустые значения лет, чтобы slider работал корректно
    valid_years = df['year'].dropna().unique()
    map_year = st.selectbox("Выберите год для карты:", sorted(valid_years, reverse=True), key='map_year_box')
    
    df_map_year = df[df['year'] == map_year].copy()
    
    map_data = []
    for _, row in df_map_year.iterrows():
        city_name = row['city']
        if city_name in CITY_COORDS:
            val = row[map_metric]
            if pd.notna(val) and val > 0:
                map_data.append({
                    'City': city_name,
                    'Lat': CITY_COORDS[city_name][0],
                    'Lon': CITY_COORDS[city_name][1],
                    'Значение': val
                })
                
    if map_data:
        df_plot_map = pd.DataFrame(map_data)
        fig_map = px.scatter_mapbox(
            df_plot_map,
            lat="Lat",
            lon="Lon",
            hover_name="City",
            hover_data={"Значение": True, "Lat": False, "Lon": False},
            size="Значение",
            color="Значение",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=35,
            zoom=3,
            mapbox_style="open-street-map"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=600)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Нет доступных географических данных для выбранного года и метрики.")

# ==================== ТАБ 2: АНАЛИТИКА, РЕЙТИНГИ И ДИНАМИКА ====================
with t2:
    st.subheader("📈 Аналитический разрез и структура факторов")
    
    st.markdown("#### 🎯 Важность признаков в математической модели (Random Forest)")
    feature_importances = pd.DataFrame({
        'Фактор': [
            'Чистая зарплата (net_salary)', 'Цена за жилье (housing_price)', 
            'Уровень бедности (poverty_level)', 'Население (people)', 
            'Миграционный прирост (migration_net)', 'Жилой фонд (housing_stock)',
            'Энергопотребление (energy_consumption)', 'Прочие социальные факторы'
        ],
        'Вес влияния': [0.294, 0.218, 0.154, 0.121, 0.087, 0.063, 0.041, 0.022]
    }).sort_values(by='Вес влияния', ascending=True)
    
    fig_imp = px.bar(
        feature_importances, 
        x='Вес влияния', 
        y='Фактор',
        orientation='h',
        color='Вес влияния',
        color_continuous_scale='Blues',
        template='plotly_white'
    )
    fig_imp.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 🏆 Сравнительные рейтинги городов")
    rating_metric = st.selectbox(
        "Выберите метрику для расчета ТОП/Анти-ТОП списков:", 
        options=feature_columns, 
        format_func=lambda x: readable_metrics.get(x, x),
        key='rating_metric_box'
    )
    
    rating_year = st.selectbox("Выберите год для формирования рейтингов:", sorted(valid_years, reverse=True), key='rating_year_box')
    
    df_rating_filtered = df[(df['year'] == rating_year) & (df[rating_metric].notna()) & (df[rating_metric] > 0)]
    df_unique_cities = df_rating_filtered.sort_values(by=rating_metric, ascending=False).drop_duplicates(subset=['city'])
    
    col_top, col_anti = st.columns(2)
    
    with col_top:
        st.success(f"📈 Лидеры: ТОП-15 городов ({int(rating_year)} г.)")
        top_15 = df_unique_cities.head(15)[['city', rating_metric]].reset_index(drop=True)
        top_15.index += 1
        st.dataframe(top_15.rename(columns={'city': 'Город', rating_metric: 'Значение'}), use_container_width=True)
        
    with col_anti:
        st.error(f"📉 Аутсайдеры: Анти-ТОП-15 городов ({int(rating_year)} г.)")
        anti_15 = df_unique_cities.tail(15).iloc[::-1][['city', rating_metric]].reset_index(drop=True)
        anti_15.index += 1
        st.dataframe(anti_15.rename(columns={'city': 'Город', rating_metric: 'Значение'}), use_container_width=True)

    st.markdown("---")
    
    st.markdown("#### 📊 Сравнительная динамика изменений во времени")
    
    all_cities_list = sorted(df['city'].unique())
    selected_cities = st.multiselect(
        "Выберите города для одновременного сравнения трендов:", 
        options=all_cities_list, 
        default=['Москва', 'Санкт-Петербург', 'Екатеринбург'] if 'Екатеринбург' in all_cities_list else all_cities_list[:2]
    )
    
    dynamic_metrics = st.multiselect(
        "Выберите показатели для вывода графиков (будет построено несколько окон):",
        options=feature_columns,
        default=['net_salary', 'housing_price'] if 'net_salary' in feature_columns else feature_columns[:2],
        format_func=lambda x: readable_metrics.get(x, x)
    )
    
    if selected_cities and dynamic_metrics:
        df_dynamic_filtered = df[df['city'].isin(selected_cities)].sort_values(by='year')
        
        for metric in dynamic_metrics:
            fig_trend = px.line(
                df_dynamic_filtered,
                x='year',
                y=metric,
                color='city',
                line_dash='data_type',
                markers=True,
                title=f"Временной тренд: {readable_metrics.get(metric, metric)}",
                labels={'year': 'Год', metric: 'Значение', 'city': 'Город'},
                template='plotly_white'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Выберите хотя бы один город и одну метрику для отображения графиков временных рядов.")

# ==================== ТАБ 3: СОХРАНЕННЫЙ ИНТЕРАКТИВНЫЙ ПРОГНОЗ ====================
with t3:
    st.subheader("🔮 Интерактивный прогноз 2025")

    latest_year_num = df[df['data_type'] == 'История']['year'].max()
    latest_df = df[(df['year'] == latest_year_num) & (df['data_type'] == 'История')]

    col_e, col_f = st.columns([1, 2])

    with col_e:

        st.markdown("### 🌍 Отслеживается общий индекс устойчивости среды")

        # Сначала выбираем показатель
        sim_feat = st.selectbox(
            "Изменить показатель:",
            feature_columns,
            format_func=lambda x: readable_metrics.get(x, x),
            key='sim_feat'
        )

        # Фильтруем только города где есть реальные данные
        valid_sim_cities = []

        for city in sorted(latest_df['city'].unique()):

            city_rows = latest_df[latest_df['city'] == city]

            if sim_feat in city_rows.columns:

                vals = city_rows[sim_feat].dropna()

                valid_vals = []

                for v in vals:
                    try:
                        if float(v) > 0:
                            valid_vals.append(float(v))
                    except:
                        pass

                if len(valid_vals) > 0:
                    valid_sim_cities.append(city)

        if len(valid_sim_cities) == 0:
            st.warning("Нет городов с данными для выбранного показателя.")
            st.stop()

        # Потом выбор города
        sim_city = st.selectbox(
            "Выберите город для моделирования:",
            valid_sim_cities,
            key='sim_city'
        )

        city_rows = latest_df[latest_df['city'] == sim_city]

        curr_val = None

        if not city_rows.empty and sim_feat in city_rows.columns:

            vals = city_rows[sim_feat].dropna()

            valid_vals = []

            for v in vals:
                try:
                    if float(v) > 0:
                        valid_vals.append(float(v))
                except:
                    pass

            if len(valid_vals) > 0:
                curr_val = valid_vals[0]

        if curr_val is None:
            st.warning("Для выбранного показателя нет данных по этому городу.")
            st.stop()

        st.write(
            f"Текущее значение ({int(latest_year_num)} г.): **{curr_val:,.2f}**".replace(",", " ")
        )

        change = st.slider(
            "Сценарий изменения показателя (%):",
            -50,
            50,
            0,
            key='slider_sim'
        )

        new_val = curr_val * (1 + change / 100)

        st.write(
            f"Моделируемое значение на 2025 г.: **{new_val:,.2f}**".replace(",", " ")
        )

    with col_f:

        st.markdown("##### Результат влияния изменения на интегральный индекс стабильности")

        city_data = city_rows.iloc[0]

        base_score = float(city_data['env_score']) if pd.notna(city_data['env_score']) else 0.5

        weight = 1 / len(feature_columns) if feature_columns else 0.1

        simulated_score = base_score + (change / 100.0) * weight
        simulated_score = max(0.0, min(1.0, simulated_score))

        fig_indicator = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=simulated_score,
            delta={'reference': base_score},
            title={'text': "Прогнозный индекс (2025)"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.33], 'color': '#ffcccc'},
                    {'range': [0.33, 0.66], 'color': '#fff0b3'},
                    {'range': [0.66, 1], 'color': '#ccffcc'}
                ]
            }
        ))

        fig_indicator.update_layout(
            height=350,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        st.plotly_chart(fig_indicator, use_container_width=True)
