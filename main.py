import random

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk.sentiment.vader as vd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from wordcloud import WordCloud
from PIL import Image
import matplotlib.colors as mcolors
# import textblob
from textblob import TextBlob
# import plotly.express as px

st.set_page_config(
    page_title='Topic Modeling',
    page_icon='data/favicon.png',
    layout='wide'
)

DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001
DEFAULT_NUM_TOPICS = 6

nltk.download("stopwords")
nltk.download('vader_lexicon')
nltk.download('punkt')
# textblob.download_corpora()
DATASETS = {
    'Escuela de Verano 2021': {
        'path': 'data/EV2021.csv.zip',
        'column': 'Q05_pregunta',
        'url': '#',
        'description': (
            'A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from '
            'February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, '
            'followed by categorizing negative reasons (such as "late flight" or "rude service").'
        )
    },
    'Escuela de Invierno 2021': {
        'path': 'data/EI2021.csv.zip',
        'column': 'Q05_pregunta',
        'url': '#',
        'description': (
            'Opinions about post-graduated UCI 2021 Winter School course.'
        )
    }
}


def lda_options():
    return {
        'num_topics': st.number_input('Numero de Tópicos', min_value=1, value=9,
                                      help='El número de temas latentes solicitados para ser extraídos del corpus '
                                           'de entrenamiento.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Número de documentos que se utilizarán en cada fragmento de formación.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Número de iteraciones por el corpus durante el entrenamiento.'),
        'update_every': st.number_input('Update Every', min_value=1, value=1,
                                        help='Número de documentos que se van a iterar para cada actualización. '
                                             'Establézcalo en 0 para el aprendizaje por lotes,'
                                             ' 1 para el aprendizaje iterativo en línea.'),
        'alpha': st.selectbox('𝛼', ('symmetric', 'asymmetric', 'auto'),
                              help='Creencia a priori en la distribución documento-tema.'),
        'eta': st.selectbox('𝜂', (None, 'symmetric', 'auto'), help='Creencia a priori sobre la distribución tema-palabra.'),
        'decay': st.number_input('𝜅', min_value=0.5, max_value=1.0, value=0.5,
                                 help='Un número entre (0,5, 1) para ponderar qué porcentaje del valor lambda '
                                      'anterior se olvida cuando se examina cada nuevo documento.'),
        'offset': st.number_input('𝜏_0', value=1.0,
                                  help='Hiperparámetro que controla cuánto ralentizaremos los primeros pasos '
                                       'las primeras iteraciones.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='La perplejidad de registro se estima cada tantas actualizaciones.'),
        'iterations': st.number_input('Iteraciones', min_value=1, value=50,
                                      help='Número máximo de iteraciones a través del corpus al inferir la '
                                           'distribución de temas de un corpus.'),
        'gamma_threshold': st.number_input('𝛾', min_value=0.0, value=0.001,
                                           help='Cambio mínimo en el valor de los parámetros gamma para '
                                                'continuar iterando.'),
        'minimum_probability': st.number_input('Probability Minima', min_value=0.0, max_value=1.0, value=0.01,
                                               help='Se filtrarán los temas con una probabilidad inferior a '
                                                    'este umbral.'),
        'minimum_phi_value': st.number_input('𝜑', min_value=0.0, value=0.01,
                                             help='si per_word_topics es True, esto representa un límite inferior '
                                                  'en las probabilidades del término.'),
        'per_word_topics': st.checkbox('Per Word Topics',
                                       help='Si es Verdadero, el modelo también calcula una lista de temas, '
                                            'ordenados en orden descendente de los temas más probables para cada '
                                            'palabra, junto con sus valores de phi multiplicados por la longitud de '
                                            'la función (es decir, el recuento de palabras).')
    }


def nmf_options():
    return {
        'num_topics': st.number_input('Numbero of Topicos', min_value=1, value=9, help='Numbero de topicos a extraer.'),
        'chunksize': st.number_input('Chunk Size', min_value=1, value=2000,
                                     help='Número de documentos que se utilizarán en cada fragmento de formación.'),
        'passes': st.number_input('Passes', min_value=1, value=1,
                                  help='Número de pasadas completas sobre el corpus de entrenamiento.'),
        'kappa': st.number_input('𝜅', min_value=0.0, value=1.0, help='Gradient descent step size.'),
        'minimum_probability': st.number_input('Minimum Probability', min_value=0.0, max_value=1.0, value=0.01,
                                               help='Si normalizar es Verdadero, se filtran los temas con '
                                                    'probabilidades más pequeñas. Si normalize es False, se filtran '
                                                    'los temas con factores más pequeños. Si se establece en Ninguno,'
                                                    ' se utiliza un valor de 1e-8 para evitar los 0.'),
        'w_max_iter': st.number_input('W max iter', min_value=1, value=200,
                                      help='Número máximo de iteraciones para entrenar W por cada lote.'),
        'w_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.0001,
                                            help='Si la diferencia de error es menor que eso, el entrenamiento de '
                                                 'W se detiene para el lote actual.'),
        'h_max_iter': st.number_input('H max iter', min_value=1, value=50,
                                      help='Número máximo de iteraciones para entrenar h por cada lote.'),
        'h_stop_condition': st.number_input('W stop cond', min_value=0.0, value=0.001,
                                            help='Si la diferencia de error es menor que eso, el entrenamiento de h '
                                                 'se detiene para el lote actual.'),
        'eval_every': st.number_input('Evaluate Every', min_value=1, value=10,
                                      help='Número de lotes después de los cuales se calcula la norma l2 de (v - Wh).'),
        'normalize': st.selectbox('Normalize', (True, False, None), help='Ya sea para normalizar el resultado.')
    }


MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': gensim.models.LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    },
    'Non-Negative Matrix Factorization': {
        'options': nmf_options,
        'class': gensim.models.Nmf,
        'help': 'https://radimrehurek.com/gensim/models/nmf.html'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

WORDCLOUD_FONT_PATH = r'data/Inkfree.ttf'

EMAIL_REGEX_STR = r'\S*@\S*'
MENTION_REGEX_STR = r'@\S*'
HASHTAG_REGEX_STR = r'#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


@st.cache_data()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')


@st.cache_data()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', str(text)) for text in texts]
    docs = [[w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('spanish')] for doc in texts]
    return docs


@st.cache_data()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs


@st.cache_data()
def create_trigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[docs])
    trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs


@st.cache_data()
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)
    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    if ngrams == 'trigrams':
        docs = create_trigrams(docs)
    return docs


@st.cache_data()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    word_cloud = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                           background_color='white', collocations=collocations).generate(wordcloud_text)
    return word_cloud


@st.cache_data()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


@st.cache_data()
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model


def clear_session_state():
    for key in ('model_kwargs', 'id2word', 'corpus', 'model', 'previous_perplexity', 'previous_coherence_model_value'):
        if key in st.session_state:
            del st.session_state[key]


def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))


def calculate_coherence(model, corpus, coherence):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()


@st.cache_data()
def white_or_black_text(background_color):
    # https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'


def perplexity_section():
    with st.spinner('Calculando valor de Perplexity ...'):
        perplexity = calculate_perplexity(st.session_state.model, st.session_state.corpus)
    key = 'previous_perplexity'
    delta = f'{perplexity - st.session_state[key]:.4}' if key in st.session_state else None
    st.metric(label='Perplexity', value=f'{perplexity:.4f}', delta=delta, delta_color='inverse')
    st.session_state[key] = perplexity
    st.markdown('Viz., https://en.wikipedia.org/wiki/Perplexity')
    st.latex(r'Perplexity = \exp\left(-\frac{\sum_d \log(p(w_d|\Phi, \alpha))}{N}\right)')


def coherence_section():
    with st.spinner('Cálculo de la puntuación de coherencia ...'):
        coherence = calculate_coherence(st.session_state.model, st.session_state.corpus, 'u_mass')
    key = 'previous_coherence_model_value'
    delta = f'{coherence - st.session_state[key]:.4f}' if key in st.session_state else None
    st.metric(label='Coherence Score', value=f'{coherence:.4f}', delta=delta)
    st.session_state[key] = coherence
    st.markdown('Viz., http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf')
    st.latex(
        r'C_{UMass} = \frac{2}{N \cdot (N - 1)}\sum_{i=2}^N\sum_{j=1}^{i-1}\log\frac{P(w_i, w_j) + \epsilon}{P(w_j)}')


@st.cache_data()
def train_projection(projection, n_components, df):
    if projection == 'PCA':
        projection_model = PCA(n_components=n_components)
    elif projection == 'T-SNE':
        projection_model = TSNE(n_components=n_components)
    elif projection == 'UMAP':
        projection_model = UMAP(n_components=n_components)
    data = projection_model.fit_transform(df.drop(columns=['dominant_topic']))
    return data


if __name__ == '__main__':
    # preprocessing_options = st.sidebar.form('preprocessing-options')
    # with preprocessing_options:
    #     st.header('Opciones de preprocesamiento')
    #     ngrams = st.selectbox('N-grams', [None, 'bigrams', 'trigams'], help='TODO ...')  # TODO ...
    #     st.form_submit_button('Preprocess')

    visualization_options = st.sidebar.form('visualization-options')
    with visualization_options:
        st.header('Opciones de visualización')
        collocations = st.checkbox('Habilitar colocaciones de WordCloud',
                                   help='Las colocaciones en nubes de palabras permiten la visualización de frases.')
        highlight_probability_minimum = st.select_slider('Resalte la probabilidad mínima',
                                                         options=[10 ** exponent for exponent in range(-10, 1)],
                                                         value=DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
                                                         help='Probabilidad mínima de tema para resaltar en color '
                                                              'una palabra en la visualización de oraciones '
                                                              'resaltadas por tema.')
        st.form_submit_button('Apply')

    st.title('Escuela Internacional de Postgrado')

    st.header('Datasets')
    st.markdown('Precargado de un par de pequeños conjuntos de datos.')
    selected_dataset = st.selectbox('Dataset', [None, *sorted(list(DATASETS.keys()))], on_change=clear_session_state)
    if not selected_dataset:
        st.write('Seleccione un conjunto de Datos ...')
        st.stop()

    with st.expander('Descripción del Dataset'):
        st.markdown(DATASETS[selected_dataset]['description'])
        st.markdown(DATASETS[selected_dataset]['url'])

    text_column = DATASETS[selected_dataset]['column']
    texts_df = generate_texts_df(selected_dataset)
    docs = generate_docs(texts_df, text_column, ngrams=None)

    st.header('Información estadística general')
    df = generate_texts_df('Escuela de Invierno 2021 Opiniones')
    ## Change the reviews type to string
    texts_df['Q05_pregunta'] = texts_df['Q05_pregunta'].astype(str)
    ## Lowercase all reviews
    texts_df['text'] = texts_df['Q05_pregunta'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    ## remove punctuation
    texts_df['text'] = texts_df['text'].str.replace('[^ws]', '')
    ## remove stopwords
    stop = stopwords.words('spanish')
    texts_df['text'] = texts_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #df2 = df.iloc[1:15, 2:16]

    with st.expander('Cursos ofrecidos'):
        cursos = texts_df.groupby("Curso")["Curso"].value_counts()
        st.table(cursos)

    with st.expander('Cantidad de estudiantes por cursos'):
        fig, ax = plt.subplots()
        result = texts_df.groupby(['Curso']).size()
        sns.barplot(x=result.values, y=result.index)
        st.pyplot(fig)

    with st.expander('Correlacion entre indicadores'):
        matrix = texts_df.iloc[:, 2:17].corr().round(1)
        fig, ax = plt.subplots()
        result = texts_df.groupby(['Curso']).size()
        sns.heatmap(matrix, annot=True)
        st.pyplot(fig)

    with st.expander('Equipo  de Conexión'):
        st.table(texts_df.iloc[:, 17:21].sum())

    with st.expander('Tipo de Conexión'):
        st.table(texts_df.iloc[:, 21:27].sum())

    swords = set().union(stopwords.words('spanish'))
    texts_df['Q05_pregunta'].drop_duplicates()
    # df.drop_duplicates(subset='text', inplace=True)

    texts_df['processed_text'] = texts_df['Q05_pregunta'].str.lower() \
        .str.replace('(@[a-z0-9]+)\w+', ' ') \
        .str.replace('(http\S+)', ' ') \
        .str.replace('([^0-9a-z \t])', ' ') \
        .str.replace(' +', ' ') \
        .apply(lambda x: [i for i in x.split() if not i in swords])
    sia = SentimentIntensityAnalyzer()

    def senti(x):
        analysis = TextBlob(x)
        # analysis = analysis.translate(from_lang='es', to='en')
        print(analysis.sentiment)
        return analysis.sentiment.polarity

    def sentiVader(x):
        vs = sia.polarity_scores(x)
        print(vs)
        return vs['compound']


    # st.table(df['processed_text'].head())
    # df['sentiment_score'] = df['processed_text'].apply(
    #      lambda x: sum([sia.polarity_scores(i)['compound'] for i in word_tokenize(' '.join(x))]))
    st.title('Analisis de sentimientos')
    st.header('¿Qué es el análisis de sentimiento?')
    st.markdown(
        "Análisis de sentimiento (también conocido como minería de opinión) se refiere al uso de procesamiento de "
        "lenguaje natural, análisis de texto y lingüística computacional para identificar y extraer información "
        "subjetiva de los recursos. Desde el punto de vista de la minería de textos, el análisis de sentimientos "
        "es una tarea de clasificación masiva de documentos de manera automática, en función de la connotación "
        "positiva o negativa del lenguaje empleado en el documento. Es importante mencionar que estos "
        "tratamientos generalmente \"se basan en relaciones estadísticas y de asociación, no en análisis "
        "lingüístico\". En términos generales, el análisis de sentimiento intenta determinar la actitud de un "
        "interlocutor o usuario con respecto a algún tema o la polaridad contextual general de un documento. "
        "La actitud puede ser su juicio o evaluación, estado afectivo (o sea, el estado emocional del autor al "
        "momento de escribir), o la intención comunicativa emocional (o sea, el efecto emocional que el autor "
        "intenta causar en el lector)."
    )
    texts_df = texts_df[texts_df['text'].notna()]
    # df['text'].drop_duplicates()
    texts_df['sentiment_score'] = texts_df['text'].apply(senti)
    # st.table(df['sentiment_score'].apply(lambda x: round(x, )).value_counts())
    with st.expander('Ejemplo de opiniones'):
        st.table(texts_df['Q05_pregunta'].head())
    with st.expander('Histograma de la polaridad de las opiniones'):
        fig, ax = plt.subplots()
        texts_df['sentiment_score'].hist()
    # df.groupby(['Curso']).size().plot(kind="barh")
        st.pyplot(fig)
    with st.expander('Calificación de las opiniones'):
        st.dataframe(texts_df[['Q05_pregunta', 'sentiment_score']])

    st.title('Topic Modeling')
    st.header('¿Qué es el modelado de temas?')
    with st.expander('Hero Image'):
        img = Image.open('data/is-this-a-topic-modeling.jpg')
        st.image(img, caption='No ... no it\'s not ...', use_column_width=True)
    st.markdown(
        'El modelado de temas es un término amplio. Abarca una serie de métodos específicos de aprendizaje estadístico.'
        ' Estos métodos hacen lo siguiente: explican documentos en términos de un conjunto de temas y esos temas '
        'en términos de un conjunto de palabras. Dos métodos muy utilizados son'
        ' la asignación latente de Dirichlet (LDA) y la factorización de matriz no negativa (NMF), por ejemplo. '
        'Si se utiliza sin calificativos adicionales, se suele suponer que el enfoque es no supervisado, '
        'aunque existen variantes supervisadas y semi-supervisadas.'
    )

    with st.expander('Detalles adicionales'):
        st.markdown('El objetivo puede verse como una factorización matricial.')
        img = Image.open('data/mf.png')
        st.image(img, use_column_width=True)
        st.markdown('Esta factorización hace que los métodos sean mucho más eficientes que la caracterización '
                    'directa de documentos en términos de palabras.')
        st.markdown('Puede encontrar más información sobre LDA y NMF en:'
                    'https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation y '
                    'https://en.wikipedia.org/wiki/Non-negative_matrix_factorization, respectivamente.')

    with st.expander('Ejemplos'):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    with st.expander('Nube de palabras del corpus'):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(), caption='Nube de palabras del Dataset', use_column_width=True)
        st.markdown('Estas son las palabras restantes después del preprocesamiento del documento.')

    with st.expander('Distribución del número de palabras del documento'):
        len_docs = [len(doc) for doc in docs]
        fig, ax = plt.subplots()
        sns.histplot(data=pd.DataFrame(len_docs, columns=['Palabras por Documento']), discrete=True, ax=ax)
        st.pyplot(fig)

    model_key = st.sidebar.selectbox('Model', [None, *list(MODELS.keys())], on_change=clear_session_state)
    model_options = st.sidebar.form('model-options')
    if not model_key:
        with st.sidebar:
            st.write('Elija un modelo para continuar ...')
        st.stop()
    with model_options:
        st.header('Opciones')
        model_kwargs = MODELS[model_key]['options']()
        st.session_state['model_kwargs'] = model_kwargs
        train_model_clicked = st.form_submit_button('Entrenar Modelo')

    if train_model_clicked:
        with st.spinner('Entrenando el Modelo ...'):
            id2word, corpus, model = train_model(docs, MODELS[model_key]['class'], **st.session_state.model_kwargs)
        st.session_state.id2word = id2word
        st.session_state.corpus = corpus
        st.session_state.model = model

    if 'model' not in st.session_state:
        st.stop()

    st.header('Modelo')
    st.write(type(st.session_state.model).__name__)
    st.write(st.session_state.model_kwargs)

    st.header('Resultados del Modelo')

    topics = st.session_state.model.show_topics(formatted=False, num_words=50,
                                                num_topics=st.session_state.model_kwargs['num_topics'], log=False)
    with st.expander('Resúmenes ponderados por palabras del tema'):
        topic_summaries = {}
        for topic in topics:
            topic_index = topic[0]
            topic_word_weights = topic[1]
            topic_summaries[topic_index] = ' + '.join(
                f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10])
        for topic_index, topic_summary in topic_summaries.items():
            st.markdown(f'**Tema {topic_index}**: _{topic_summary}_')

    colors = random.sample(COLORS, k=model_kwargs['num_topics'])
    with st.expander('Nubes de palabras de las palabras clave por tópico'):
        cols = st.columns(3)
        for index, topic in enumerate(topics):
            wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                           background_color='white', collocations=collocations, prefer_horizontal=1.0,
                           color_func=lambda *args, **kwargs: colors[index])
            with cols[index % 3]:
                wc.generate_from_frequencies(dict(topic[1]))
                st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

    with st.expander('Oraciones destacadas del tema'):
        sample = texts_df.sample(10)
        for index, row in sample.iterrows():
            html_elements = []
            for token in row[text_column].__str__().split():
                if st.session_state.id2word.token2id.get(token) is None:
                    html_elements.append(f'<span style="text-decoration:line-through;">{token}</span>')
                else:
                    term_topics = st.session_state.model.get_term_topics(token, minimum_probability=0)
                    topic_probabilities = [term_topic[1] for term_topic in term_topics]
                    max_topic_probability = max(topic_probabilities) if topic_probabilities else 0
                    if max_topic_probability < highlight_probability_minimum:
                        html_elements.append(token)
                    else:
                        max_topic_index = topic_probabilities.index(max_topic_probability)
                        max_topic = term_topics[max_topic_index]
                        background_color = colors[max_topic[0]]
                        # color = 'white'
                        color = white_or_black_text(background_color)
                        html_elements.append(
                            f'<span style="background-color: {background_color}; color: {color}; opacity: 0.5;">{token}</span>')
            st.markdown(f'Document #{index}: {" ".join(html_elements)}', unsafe_allow_html=True)

    has_log_perplexity = hasattr(st.session_state.model, 'log_perplexity')
    with st.expander('Métricas'):
        if has_log_perplexity:
            left_column, right_column = st.columns(2)
            with left_column:
                perplexity_section()
            with right_column:
                coherence_section()
        else:
            coherence_section()

    # with st.expander('Low Dimensional Projections'):
    #     with st.form('projections-form'):
    #         left_column, right_column = st.columns(2)
    #         projection = left_column.selectbox('Projection', ['PCA', 'T-SNE', 'UMAP'], help='TODO ...')
    #         plot_type = right_column.selectbox('Plot', ['2D', '3D'], help='TODO ...')
    #         n_components = 3
    #         columns = [f'proj{i}' for i in range(1, 4)]
    #         generate_projection_clicked = st.form_submit_button('Generate Projection')
    #
    #     if generate_projection_clicked:
    #         topic_weights = []
    #         for index, topic_weight in enumerate(st.session_state.model[st.session_state.corpus]):
    #             weight_vector = [0] * int(st.session_state.model_kwargs['num_topics'])
    #             for topic, weight in topic_weight:
    #                 weight_vector[topic] = weight
    #             topic_weights.append(weight_vector)
    #         df = pd.DataFrame(topic_weights)
    #         dominant_topic = df.idxmax(axis='columns').astype('string')
    #         dominant_topic_percentage = df.max(axis='columns')
    #         df = df.assign(dominant_topic=dominant_topic, dominant_topic_percentage=dominant_topic_percentage,
    #                        text=texts_df[text_column])
    #         with st.spinner('Training Projection'):
    #             projections = train_projection(projection, n_components, df.drop(columns=['text']))
    #         data = pd.concat([df, pd.DataFrame(projections, columns=columns)], axis=1)
    #
    #         px_options = {'color': 'dominant_topic', 'size': 'dominant_topic_percentage',
    #                       'hover_data': ['dominant_topic', 'dominant_topic_percentage', 'text']}
    #         if plot_type == '2D':
    #             fig = px.scatter(data, x='proj1', y='proj2', **px_options)
    #             st.plotly_chart(fig)
    #             fig = px.scatter(data, x='proj1', y='proj3', **px_options)
    #             st.plotly_chart(fig)
    #             fig = px.scatter(data, x='proj2', y='proj3', **px_options)
    #             st.plotly_chart(fig)
    #         elif plot_type == '3D':
    #             fig = px.scatter_3d(data, x='proj1', y='proj2', z='proj3', **px_options)
    #             st.plotly_chart(fig)

    if hasattr(st.session_state.model, 'inference'):  # gensim Nmf has no 'inference' attribute so pyLDAvis fails
        if st.button('Generar visualizacion de modelo'):
            with st.spinner('Creando visualizacion ...'):
                py_lda_vis_data = pyLDAvis.gensim_models.prepare(st.session_state.model, st.session_state.corpus,
                                                                 st.session_state.id2word)
                py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
            with st.expander('pyLDAvis', expanded=True):
                st.markdown('pyLDAvis está diseñado para ayudar a los usuarios a interpretar los temas en un modelo'
                            ' de tema que se ha ajustado a un corpus de datos de texto. El paquete extrae '
                            'información de un modelo de tema LDA ajustado para informar una visualización '
                            'interactiva basada en la web.')
                st.markdown('https://github.com/bmabey/pyLDAvis')
                components.html(py_lda_vis_html, width=1300, height=800)
