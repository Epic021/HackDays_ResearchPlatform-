import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Unified Research Assistant", page_icon="🧭", layout="centered")

st.title("🧭 Unified Research Assistant")
st.caption("Navigate through different stages of your research journey — from ideation to review to conferences.")

st.markdown("""
### Choose your current stage:
Each stage uses a specialized AI agent and interface:
- 💡 **Ideation** → Generate and refine innovative project ideas.
- 📚 **Literature Review** → Fetch and summarize research papers.
- 🎓 **Conference Finder** → Discover upcoming conferences in your area.
""")

stage = st.selectbox(
    "Select your research stage:",
    ["-- Choose --", "💡 Ideation", "📚 Literature Review", "🎓 Conference Finder"],
    index=0
)

st.markdown("---")

if stage == "-- Choose --":
    st.info("Select a stage above to continue.")

elif stage == "💡 Ideation":
    st.success("Launching Ideation Assistant...")
    st.markdown("Click below to open the **Ideation Assistant** in a new tab.")
    st.link_button("Open 💡 Ideation Assistant", "http://10.95.25.34:8502")

elif stage == "📚 Literature Review":
    st.success("Launching Literature Review Assistant...")
    st.markdown("Click below to open the **Literature Review Assistant** in a new tab.")
    st.link_button("Open 📚 Literature Review Assistant", "http://10.95.25.34:8501")

elif stage == "🎓 Conference Finder":
    st.success("Launching Conference Finder...")
    st.markdown("Click below to open the **Conference Finder** in a new tab.")
    st.link_button("Open 🎓 Conference Finder", "http://10.95.25.34:8503")

st.markdown("---")
st.caption("Built with ❤️ using LangGraph, Arxiv, Tavily, and Streamlit.")

