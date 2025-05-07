import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def load_metrics():
    files = [
        "stats_SingleNode.json",
        "stats_SingleNode_tuned.json",
        "stats_MultiNode.json",
        "stats_MultiNode_tuned.json"
    ]
    
    data = []
    for f in files:
        path = os.path.join("results", f)
        if os.path.exists(path):
            with open(path, "r") as fp:
                metrics = json.load(fp)
                is_optimized = "tuned" in f
                nodes = "SingleNode" if "SingleNode" in metrics.get("setup", "") else "MultiNode"
                
                data.append({
                    "Configuration": f"{nodes} {'(Opt)' if is_optimized else ''}",
                    "Nodes": nodes,
                    "Optimized": is_optimized,
                    "Runtime (s)": metrics["duration"],
                    "Memory (MiB)": float(metrics["memory"].replace("MiB", "").strip())
                })
    
    return pd.DataFrame(data)


def plot_matplotlib_bar(x, y, xlabel, ylabel, title, colors=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, y, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.title("Анализ производительности Spark Application")
    
    df = load_metrics()
    
    st.subheader("Обзор результатов")
    st.table(df[["Configuration", "Runtime (s)", "Memory (MiB)"]])
    
    runtime_colors = ['#1a5f7a', '#ff9933', '#1a5f7a', '#ff9933'] 
    memory_colors = ['#e66c2c', '#1a5f7a', '#e66c2c', '#1a5f7a']   
    
    st.subheader("Сравнение времени выполнения")
    plot_matplotlib_bar(
        x=df["Configuration"],
        y=df["Runtime (s)"],
        xlabel="Конфигурация эксперимента",
        ylabel="Время (секунды)",
        title="Сравнение времени выполнения",
        colors=runtime_colors
    )
    
    st.subheader("Использование памяти")
    plot_matplotlib_bar(
        x=df["Configuration"],
        y=df["Memory (MiB)"],
        xlabel="Конфигурация эксперимента",
        ylabel="Память (МиБ)",
        title="Сравнение использования памяти",
        colors=memory_colors
    )

if __name__ == "__main__":
    main()
