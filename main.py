import streamlit as st
from streamlit_option_menu import option_menu

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

from collections import Counter

import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import neattext.functions as nfx
from wordcloud import WordCloud

import pdfplumber
import docx2txt

from rouge import Rouge

from goose3 import Goose
from requests import get

import validators

import nltk
nltk.download('punkt')

def sumy_summarizer(docx, num=5):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, num)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def plot_worldcloud(docx):
    myworldcloud = WordCloud().generate(docx)
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(myworldcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

def plot_word_freq(docx, num=10):
    word_freq = Counter(docx.split())
    most_common_tokens = dict(word_freq.most_common(num))
    word_freq_df = pd.DataFrame({'Tokens': most_common_tokens.keys(), 'Counts': most_common_tokens.values()})
    c = alt.Chart(word_freq_df).mark_bar().encode(
        x='Tokens', y='Counts')
    st.altair_chart(c, use_container_width=True)


def read_pdf2(file):
    all_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            all_text += '\n' + text
        return all_text

def evaluate_summary(summary,reference):
    r = Rouge()
    eval_score = r.get_scores(summary, reference)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df

def main():
    st.set_page_config(
        layout="wide",
        page_title="Text Article Analyzer",
    )

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    with st.sidebar.image("/logos/logos_black.png", use_column_width=True):
        selected = option_menu("Methods", ["Raw_Text", 'Folder', 'URL', 'Evaluate_Summary'],
                               icons=['pencil', 'folder', 'link','book'], menu_icon="cast", default_index=0)

# _______________________________________________________________________________________________________________________________
    if selected == "Raw_Text":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)

        st.subheader("Raw Text")
        raw_text = st.text_area("Enter Text Here", height=120)

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        if col8.button("Summarize"):
            try:
                processed_text = nfx.remove_stopwords(raw_text)
                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("LexRank"):
                    my_summary = sumy_summarizer(raw_text)
                    st.write(my_summary)

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Word Cloud"):
                        try:
                            plot_worldcloud(processed_text)
                        except:
                            st.warning("Insufficient Data")
                with col2:
                    with st.expander("Word Frequency"):
                        plot_word_freq(processed_text)
            except:
                st.warning("Insufficient Data")

# _______________________________________________________________________________________________________________________________
    elif selected == "Folder":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)
        st.subheader("Folder")

        text_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

        if text_file is not None:
            if text_file.type == 'application/pdf':
                raw_text = read_pdf2(text_file)
            elif text_file.type == 'text/plain':
                raw_text = str(text_file.read(), "utf-8")
            else:
                raw_text = docx2txt.process(text_file)

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        if col8.button("Summarize"):
            try:
                processed_text = nfx.remove_stopwords(raw_text)
                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("LexRank"):
                    my_summary = sumy_summarizer(raw_text)
                    st.write(my_summary)

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Word Cloud"):
                        try:
                            plot_worldcloud(processed_text)
                        except:
                            st.warning("Insufficient Data")
                with col2:
                    with st.expander("Word Frequency"):
                        plot_word_freq(processed_text)
            except:
                st.warning("Insufficient Data")

# _______________________________________________________________________________________________________________________________
    elif selected == "URL":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)
        st.subheader("Uniform Resource Locator (URL)")

        url = st.text_input('Enter URL here:')
        col1, col2, col3, col4, col5, col6, col7, col8= st.columns(8)

        if col8.button("Summarize"):
            valid = validators.url(url)
            if valid == True:
                response = get(url)
                extractor = Goose()
                article = extractor.extract(raw_html=response.content)
                raw_text = article.cleaned_text

                processed_text = nfx.remove_stopwords(raw_text)
                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("LexRank"):
                    my_summary = sumy_summarizer(raw_text)
                    st.write(my_summary)

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Word Cloud"):
                        try:
                            plot_worldcloud(processed_text)
                        except:
                            st.warning("Insufficient Data")
                with col2:
                    with st.expander("Word Frequency"):
                        plot_word_freq(processed_text)
            else:
                st.warning("Not a Valid URL")

# _______________________________________________________________________________________________________________________________
    elif selected == "Evaluate_Summary":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)
        st.subheader("Rouge Score Metrics")

        col1, col2 = st.columns(2)

        with col1:
            raw_text = st.text_area("Enter Human Generated Summary Here:", height=120)

        with col2:
            my_summary = st.text_area("Enter Computer Generated Summary Here:", height=120)

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        if col8.button("Evaluate"):
            try:
                with st.expander("Rouge Score"):
                    score = evaluate_summary(my_summary, raw_text)
                    st.write(score)
            except:
                st.warning("Please Check Inputs")

if __name__ == '__main__':
    main()





