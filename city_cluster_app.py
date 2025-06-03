import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import folium
from openrouteservice import Client
import time
from streamlit_folium import st_folium

# ------------------- App Title -------------------
st.set_page_config(layout="wide")
st.title("City Clustering and Route Mapping")

# ------------------- File Location -------------------
file_path = 'ClusterLatlong.xlsx'


try:
    df_cities = pd.read_excel(file_path, sheet_name='Sheet1')
    df_fos = pd.read_excel(file_path, sheet_name='Sheet2')
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# ------------------- Clean and Prepare -------------------
df_cities.columns = df_cities.columns.str.strip()
df_fos.columns = df_fos.columns.str.strip()

coords = df_cities[['Latitude', 'Longitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_cities['Cluster'] = kmeans.fit_predict(coords_scaled)
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

# ------------------- Radius Check -------------------
st.subheader("🔍 Cluster Radius Validation (150 km max)")

def cluster_radius_check(df, centers):
    warnings = []
    for cluster_id in df['Cluster'].unique():
        subset = df[df['Cluster'] == cluster_id]
        center_coord = (centers[cluster_id][0], centers[cluster_id][1])
        for _, row in subset.iterrows():
            dist_km = geodesic((row['Latitude'], row['Longitude']), center_coord).kilometers
            if dist_km > 150:
                warnings.append(f"⚠️ Cluster {cluster_id} exceeds 150 km: {row['City']} - {dist_km:.2f} km")
    return warnings

for warning in cluster_radius_check(df_cities, centers):
    st.warning(warning)

# ------------------- Closest Cities to Center -------------------
st.subheader("📍 Closest 3 Cities to Each Cluster Center")

for cluster_id in sorted(df_cities['Cluster'].unique()):
    city_distances = []
    for _, row in df_cities[df_cities['Cluster'] == cluster_id].iterrows():
        dist_km = geodesic(
            (row['Latitude'], row['Longitude']),
            (centers[cluster_id][0], centers[cluster_id][1])
        ).kilometers
        city_distances.append((row['City'], dist_km))
    top3 = sorted(city_distances, key=lambda x: x[1])[:3]
    st.markdown(f"**Cluster {cluster_id + 1}**: " + ", ".join([f"{city} ({dist:.1f} km)" for city, dist in top3]))

# ------------------- Map Creation -------------------
st.subheader("🗺️ Cluster Route Map")

map_center = [df_cities['Latitude'].mean(), df_cities['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=6)
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'black']

# Initialize ORS client
client = Client(key='5b3ce3597851110001cf624858f617b1d4974c6d8c8f1f050d5e245f')  # Replace with your OpenRouteService API key

for cluster_id in sorted(df_cities['Cluster'].unique()):
    cluster_data = df_cities[df_cities['Cluster'] == cluster_id].reset_index(drop=True)
    
    # Sort cities by proximity to the first city
    base_city = (cluster_data.loc[0, 'Latitude'], cluster_data.loc[0, 'Longitude'])
    cluster_data['Distance'] = cluster_data.apply(
        lambda row: geodesic(base_city, (row['Latitude'], row['Longitude'])).km, axis=1
    )
    cluster_data = cluster_data.sort_values(by='Distance').reset_index(drop=True)

    # Draw routes between consecutive cities
    for i in range(len(cluster_data) - 1):
        start = (cluster_data.loc[i, 'Longitude'], cluster_data.loc[i, 'Latitude'])
        end = (cluster_data.loc[i + 1, 'Longitude'], cluster_data.loc[i + 1, 'Latitude'])
        try:
            route = client.directions(
                coordinates=[start, end],
                profile='driving-car',
                format='geojson'
            )
            folium.GeoJson(
                route,
                name=f"Route Cluster {cluster_id + 1}",
                style_function=lambda x, color=cluster_colors[cluster_id % len(cluster_colors)]: {
                    'color': color,
                    'weight': 3,
                    'opacity': 0.8
                }
            ).add_to(m)
            time.sleep(1)
        except Exception as e:
            st.error(f"Routing error in cluster {cluster_id}: {e}")

    # City Markers
    for _, row in cluster_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['City']} (Cluster {cluster_id + 1})",
            icon=folium.Icon(color=cluster_colors[cluster_id % len(cluster_colors)])
        ).add_to(m)

# ------------------- Show Map -------------------
st_data = st_folium(m, width=1000, height=600)
