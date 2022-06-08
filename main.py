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

def sumy_summarizer(docx, num):
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


    # hide_st_style = """
    #             <style>
    #             #MainMenu {visibility: hidden;}
    #             footer {visibility: hidden;}
    #             header {visibility: hidden;}
    #             </style>
    #             """
    # st.markdown(hide_st_style, unsafe_allow_html=True)

    with st.sidebar:
        st.image("logos/logos_white.png", use_column_width='auto')
        selected = option_menu("Methods", ["Raw_Text", 'File', 'URL', 'Upgrade to Text Article Analyzer PRO'],
                               icons=['pencil', 'folder', 'link','book', 'diamond'], menu_icon="cast", default_index=0)

# _______________________________________________________________________________________________________________________________
    if selected == "Raw_Text":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)

        st.subheader("Raw Text")

        num = st.slider("Pick the number of sentences you want to extract: ", 1, 20, 5)

        raw_text = st.text_area("Enter Text Here", height=120)
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        count = len(raw_text.split())
        col1.caption("Words at present:  {} /1000".format(count))
        if count < 1000:
            if col7.button("Summarize"):
                try:
                    processed_text = nfx.remove_stopwords(raw_text)
                    with st.expander("Original Text"):
                        st.write(raw_text)

                    with st.expander("LexRank"):
                        my_summary = sumy_summarizer(raw_text, num)
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
        else:
            st.error("Words are more than 1000. Subscribe to Text Article Analyzer PRO")

# _______________________________________________________________________________________________________________________________
    elif selected == "File":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)
        st.subheader("File")

        num = st.slider("Pick the number of sentences you want to extract: ", 1, 20, 5)

        text_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        if text_file is not None:

            if text_file.type == 'application/pdf':
                raw_text = read_pdf2(text_file)
                count = len(raw_text.split())
                col1.caption("Words at present:  {} /1000".format(count))
                if count < 1000:
                    if col7.button("Summarize"):
                        try:
                            processed_text = nfx.remove_stopwords(raw_text)
                            with st.expander("Original Text"):
                                st.write(raw_text)

                            with st.expander("LexRank"):
                                my_summary = sumy_summarizer(raw_text, num)
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
                else:
                    st.warning("Words are more than 1000. Subscribe to Text Article Analyzer PRO")
            elif text_file.type == 'text/plain':
                raw_text = str(text_file.read(), "utf-8")
                count = len(raw_text.split())
                col1.caption("Words at present:  {} /1000".format(count))
                if count < 1000:
                    if col7.button("Summarize"):
                        try:
                            processed_text = nfx.remove_stopwords(raw_text)
                            with st.expander("Original Text"):
                                st.write(raw_text)

                            with st.expander("LexRank"):
                                my_summary = sumy_summarizer(raw_text, num)
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
                else:
                    st.warning("Words are more than 1000. Subscribe to Text Article Analyzer PRO")
            else:
                raw_text = docx2txt.process(text_file)
                count = len(raw_text.split())
                col1.caption("Words at present:  {} /1000".format(count))
                if count < 1000:
                    if col7.button("Summarize"):
                        try:
                            processed_text = nfx.remove_stopwords(raw_text)
                            with st.expander("Original Text"):
                                st.write(raw_text)

                            with st.expander("LexRank"):
                                my_summary = sumy_summarizer(raw_text, num)
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
                else:
                    st.warning("Words are more than 1000. Subscribe to Text Article Analyzer PRO")
# _______________________________________________________________________________________________________________________________
    elif selected == "URL":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)
        st.subheader("Uniform Resource Locator (URL)")

        num = st.slider("Pick the number of sentences you want to extract: ", 1, 20, 5)

        url = st.text_input('Enter URL here:')
        col1, col2, col3, col4, col5, col6, col7= st.columns(7)

        if col7.button("Summarize"):
            valid = validators.url(url)
            if valid == True:
                response = get(url)
                extractor = Goose()
                article = extractor.extract(raw_html=response.content)
                raw_text = article.cleaned_text

                count = len(raw_text.split())
                col1.caption("Words at present:  {} /1000".format(count))
                if count < 1000:
                    processed_text = nfx.remove_stopwords(raw_text)
                    with st.expander("Original Text"):
                        st.write(raw_text)

                    with st.expander("LexRank"):
                        my_summary = sumy_summarizer(raw_text,num)
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
                    st.warning("Words are more than 1000. Subscribe to Text Article Analyzer PRO")
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
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Rouge Score"):
                        score = evaluate_summary(my_summary, raw_text)
                        st.write(score.T)
                with col2:
                    with st.expander("Rouge Score Graph"):
                        score = evaluate_summary(my_summary, raw_text)
                        score['metrics'] = score.index
                        c = alt.Chart(score).mark_bar().encode(
                            x= 'rouge-1', y='metrics'
                        )

                        d = alt.Chart(score).mark_bar().encode(
                            x= 'rouge-2', y='metrics'

                        )

                        e = alt.Chart(score).mark_bar().encode(
                            x='rouge-l', y='metrics'

                        )

                        st.altair_chart(c)
                        st.altair_chart(d)
                        st.altair_chart(e)

            except:
                st.warning("Please Check Inputs")
    elif selected == "Upgrade to Text Article Analyzer PRO":
        st.markdown("<h1 style='text-align: center;'>Text Article Analyzer</h1>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()





