import re
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import sklearn
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Explorador de clusters científicos",
    page_icon="🧠",
    layout="wide"
)


def find_file(filename: str) -> Path:
    """Search common locations for exported CSV files."""
    candidates = [
        Path("data") / filename,
        Path(filename),
        Path(".") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No se encontró '{filename}'. Busqué en: "
        + ", ".join(str(p) for p in candidates)
    )


@st.cache_data
def load_data():
    docs_path = find_file("clustered_docs.csv")
    stats_path = find_file("cluster_stats.csv")

    docs = pd.read_csv(docs_path)
    stats = pd.read_csv(stats_path)

    # Standardize expected columns from the notebook
    rename_map = {}
    if "prediction" in docs.columns and "cluster" not in docs.columns:
        rename_map["prediction"] = "cluster"
    if rename_map:
        docs = docs.rename(columns=rename_map)

    required_cols = {"title", "abstract_clean", "cluster"}
    missing = required_cols - set(docs.columns)
    if missing:
        raise ValueError(
            "Faltan columnas necesarias en clustered_docs.csv: "
            + ", ".join(sorted(missing))
        )

    # Ensure cluster is numeric when possible
    docs["cluster"] = pd.to_numeric(docs["cluster"], errors="coerce")
    docs = docs.dropna(subset=["cluster"]).copy()
    docs["cluster"] = docs["cluster"].astype(int)

    if "abstract_length" in docs.columns:
        docs["abstract_length"] = pd.to_numeric(docs["abstract_length"], errors="coerce")

    return docs, stats


@st.cache_data
def build_cluster_top_words(docs: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Compute top words per cluster directly from abstract_clean."""
    stopwords = {
        "the", "and", "for", "with", "that", "from", "this", "were", "have", "has",
        "had", "into", "than", "then", "they", "their", "there", "after", "before",
        "using", "used", "use", "our", "can", "may", "also", "such", "these", "those",
        "are", "was", "been", "being", "but", "not", "all", "any", "each", "other",
        "more", "most", "some", "many", "few", "via", "per", "due", "new", "two",
        "one", "among", "over", "under", "between", "during", "within", "without",
        "covid", "19", "sars", "cov", "coronavirus", "study", "results", "conclusions",
        "background", "methods", "patients", "patient", "disease", "analysis",
        "data", "clinical", "infected", "infection"
    }

    rows = []
    for cluster_id in sorted(docs["cluster"].dropna().unique()):
        text_series = docs.loc[docs["cluster"] == cluster_id, "abstract_clean"].fillna("")
        tokens = []
        for text in text_series:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            words = [w for w in words if w not in stopwords]
            tokens.extend(words)

        counts = Counter(tokens).most_common(top_n)
        for word, count in counts:
            rows.append({"cluster": cluster_id, "word": word, "count": count})

    return pd.DataFrame(rows)


@st.cache_data
def build_yearly_counts(docs: pd.DataFrame) -> pd.DataFrame:
    """Create yearly counts if publish_year exists."""
    if "publish_year" not in docs.columns:
        return pd.DataFrame(columns=["cluster", "publish_year", "doc_count"])

    yearly = docs.dropna(subset=["publish_year"]).copy()
    yearly["publish_year"] = pd.to_numeric(yearly["publish_year"], errors="coerce")
    yearly = yearly.dropna(subset=["publish_year"]).copy()
    yearly["publish_year"] = yearly["publish_year"].astype(int)

    return (
        yearly.groupby(["cluster", "publish_year"])
        .size()
        .reset_index(name="doc_count")
        .sort_values(["cluster", "publish_year"])
    )


def main():
    try:
        docs, stats = load_data()
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")
        st.stop()

    words = build_cluster_top_words(docs)
    yearly_counts = build_yearly_counts(docs)

    st.title("🧠 Explorador interactivo de clusters científicos")
    st.write(
        "Aplicación para explorar clusters temáticos obtenidos a partir de abstracts "
        "científicos usando PySpark, TF-IDF y KMeans."
    )

    st.sidebar.header("Filtros")
    clusters = sorted(docs["cluster"].dropna().unique().tolist())
    selected_cluster = st.sidebar.selectbox("Seleccionar cluster", clusters)
    n_examples = st.sidebar.slider("Cantidad de ejemplos", 3, 15, 5)
    search_term = st.sidebar.text_input("Buscar palabra en abstracts")

    cluster_docs = docs[docs["cluster"] == selected_cluster].copy()

    if search_term:
        cluster_docs = cluster_docs[
            cluster_docs["abstract_clean"].fillna("").str.contains(search_term, case=False, na=False)
        ]

    cluster_words = words[words["cluster"] == selected_cluster].sort_values("count", ascending=False)
    cluster_stats = stats[stats["cluster"] == selected_cluster]

    st.subheader(f"Cluster {selected_cluster}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cantidad de documentos", len(cluster_docs))

    with col2:
        if not cluster_stats.empty and "avg_abstract_length" in cluster_stats.columns:
            st.metric(
                "Longitud promedio del abstract",
                round(float(cluster_stats["avg_abstract_length"].iloc[0]), 2)
            )
        else:
            st.metric("Longitud promedio del abstract", "N/D")

    with col3:
        if not cluster_words.empty:
            top_signal = ", ".join(cluster_words["word"].head(3).tolist())
            st.metric("Top términos", top_signal)
        else:
            st.metric("Top términos", "N/D")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Palabras clave",
        "Ejemplos",
        "Journals",
        "Tendencia temporal"
    ])

    with tab1:
        st.markdown("### Palabras más frecuentes")
        if cluster_words.empty:
            st.info("No hay palabras para mostrar en este cluster.")
        else:
            st.dataframe(cluster_words.head(15), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_df = cluster_words.head(10).sort_values("count", ascending=True)
            ax.barh(plot_df["word"], plot_df["count"])
            ax.set_title(f"Top palabras del cluster {selected_cluster}")
            ax.set_xlabel("Frecuencia")
            ax.set_ylabel("Palabra")
            st.pyplot(fig)

    with tab2:
        st.markdown("### Ejemplos de documentos")
        cols_to_show = [c for c in ["cord_uid", "title", "journal", "abstract_clean", "abstract_length"] if c in cluster_docs.columns]
        st.dataframe(cluster_docs[cols_to_show].head(n_examples), use_container_width=True)

    with tab3:
        st.markdown("### Journals más frecuentes")
        if "journal" in cluster_docs.columns:
            journal_counts = (
                cluster_docs["journal"]
                .fillna("NULL")
                .value_counts()
                .reset_index()
            )
            journal_counts.columns = ["journal", "count"]
            st.dataframe(journal_counts.head(10), use_container_width=True)
        else:
            st.info("No hay columna 'journal' en los datos exportados.")

    with tab4:
        st.markdown("### Publicaciones por año")
        if yearly_counts.empty:
            st.info("El notebook no exportó 'publish_year', así que no se puede mostrar la tendencia temporal.")
        else:
            cluster_year = yearly_counts[yearly_counts["cluster"] == selected_cluster]
            if cluster_year.empty:
                st.info("No hay datos temporales para este cluster.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(cluster_year["publish_year"], cluster_year["doc_count"], marker="o")
                ax.set_title(f"Publicaciones por año - cluster {selected_cluster}")
                ax.set_xlabel("Año")
                ax.set_ylabel("Cantidad de documentos")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                st.dataframe(cluster_year, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "La app usa directamente los resultados exportados por el notebook: "
        "`clustered_docs.csv` y `cluster_stats.csv`."
    )


if __name__ == "__main__":
    main()
