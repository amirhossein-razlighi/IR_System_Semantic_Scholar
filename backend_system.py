from typing import Dict, List
import networkx as nx
import json
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from math import log10

with open("Kasaei.json", "r") as f:
    kasaei_data = json.load(f)

with open("Rohban.json", "r") as f:
    rohban_data = json.load(f)

with open("Sharifi.json", "r") as f:
    sharifi_data = json.load(f)

with open("Soleymani.json", "r") as f:
    soleymani_data = json.load(f)

with open("Rabiee.json", "r") as f:
    rabiee_data = json.load(f)


def pagerank(
    graph: Dict[str, List[str]], user_prefs: Dict[str, float]
) -> Dict[str, float]:
    """
    Returns the personalized PageRank scores for the nodes in the graph, given the user's preferences.

    Parameters:
    graph (Dict[str, List[str]]): The graph represented as a dictionary of node IDs and their outgoing edges.
    user_prefs (Dict[str, float]): The user's preferences represented as a dictionary of node IDs and their scores.

    Returns:
    Dict[str, float]: A dictionary of node IDs and their personalized PageRank scores.
    """
    damping_factor = 0.85
    max_iterations = 100
    convergence_threshold = 0.0001

    num_nodes = len(graph)
    initial_score = 1 / num_nodes
    page_ranks = {node: initial_score for node in graph}

    jumping_probs = {}
    sum_prefs = sum(user_prefs.values())

    if sum_prefs != 0:
        for node, score in user_prefs.items():
            jumping_probs[node] = score / sum_prefs

    in_graph = {}
    for node, out_nodes in graph.items():
        for out_node in out_nodes:
            if out_node not in in_graph:
                in_graph[out_node] = []
            in_graph[out_node].append(node)

    for _ in range(max_iterations):
        new_page_ranks = {}
        for node in graph:
            new_page_rank = (1 - damping_factor) / num_nodes

            if node in in_graph:
                for in_node in in_graph[node]:
                    new_page_rank += (
                        damping_factor * page_ranks[in_node] / len(graph[in_node])
                    )

            if node in jumping_probs:
                new_page_rank += damping_factor * jumping_probs[node]

            new_page_ranks[node] = new_page_rank

        diff = 0
        for node in graph:
            diff += abs(new_page_ranks[node] - page_ranks[node])

        page_ranks = new_page_ranks

        if diff < convergence_threshold:
            break

    return page_ranks


papers_by_id = {}

for data in [kasaei_data, rabiee_data, rohban_data, sharifi_data, soleymani_data]:
    for paper in data:
        papers_by_id[paper["ID"]] = paper

papers_by_prof = {}

profs = ["kasaei", "Rabiee", "Rohban", "Sharifi", "Soleymani"]

for prof in profs:
    with open(f"{prof}.json", "r") as f:
        data = json.load(f)
        papers_by_prof[prof] = data


papers_dataset = list(papers_by_id.values())


def important_articles(Professor: str) -> List[str]:
    """
    Returns the most important articles in the field of given professor, based on the personalized PageRank scores.

    Parameters:
    Professor (str): Professor's name.

    Returns:
    List[str]: A list of article IDs representing the most important articles in the field of given professor.
    """

    graph = {}
    for paper in papers_dataset:
        graph[paper["ID"]] = []
        for ref in paper["References"]:
            if ref in papers_by_id:
                graph[paper["ID"]].append(ref)

    user_prefs = {paper["ID"]: 1.0 for paper in papers_by_prof[Professor]}

    # Calculate the personalized PageRank scores for the nodes in the graph
    node_scores = pagerank(graph, user_prefs)

    # Sort the nodes based on their scores
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top 10 nodes
    return [node[0] for node in sorted_nodes[:10]]


def clean_data(text: str):
    """Preprocesses the text with tokenization, case folding, stemming and lemmatization, and punctuations

    Parameters
    ----------
    text : str
        The title or abstract of an article

    Returns
    -------
    list
        A list of tokens
    """

    # TODO: tokenize, case_folding, stem, lemmatize, punctuations
    tokenized = nltk.word_tokenize(text)
    case_folded = [word.lower() for word in tokenized]
    stemmer = nltk.stem.PorterStemmer()
    stemmed = [stemmer.stem(word) for word in case_folded]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    punctuations = [word for word in lemmatized if word.isalpha()]
    return punctuations


def get_references(prof_name: str) -> List[str]:
    """Gets the references of the articles of a professor

    Parameters
    ----------
    prof_name : str
        The name of the professor

    Returns
    -------
    list
        A list of references
    """

    global papers_by_prof

    references = []
    for paper in papers_by_prof[prof_name]:
        for ref in paper["References"]:
            references.append(ref)

    return references


index = {}

for paper in papers_dataset:
    for word in clean_data(paper["Title"]):
        if word not in index:
            index[word] = []
        index[word].append(paper["ID"])

    for word in clean_data(paper["Abstract"]):
        if word not in index:
            index[word] = []
        index[word].append(paper["ID"])

N = len(papers_dataset)


def search(
    title_query: str,
    abstract_query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weight: float = 0.5,
    should_print=False,
    preferred_field: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn-lnn' or 'ltc-lnc' or 'okapi25'

    preferred_field: A list containing preference rate to Dr. Rabiee, Dr. Soleymani, Dr. Rohban,
                     Dr. Kasaei, and Dr. Sharifi's papers, respectively.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    # TODO: return top 'max_result_count' documents for your searched query
    # result = ["a72d6bd0b6d9b7aee66e91253bc6c6de37fa4e6a"]

    # Retrieve relevant documents based on the query terms
    title_terms = title_query.lower().split()
    abstract_terms = abstract_query.lower().split()

    title_terms = clean_data(title_query)
    abstract_terms = clean_data(abstract_query)
    query_terms = title_terms + abstract_terms

    # Calculate the IDF for each term in the query
    doc_freqs = defaultdict(int)
    for term in query_terms:
        for doc_id in index[term]:
            doc_freqs[doc_id[0]] += 1

    idfs = defaultdict(float)
    for term in query_terms:
        # Consider the 0 division problem
        if len(index[term]) == 0:
            idfs[term] = 0
        else:
            idfs[term] = log10(N / len(index[term]))

    # Retrieve the relevant documents for each term in the query
    relevant_docs = defaultdict(list)
    for term in query_terms:
        for doc in index[term]:
            relevant_docs[term].append(doc)

    # Calculate the score for each document. the final score is the weighted sum of title and abstract scores
    title_scores = defaultdict(float)
    abstract_scores = defaultdict(float)

    # Calculate tf for each term in the query in each document and store it
    doc_tf_s = {}
    query_tf_s = defaultdict(int)
    for term in query_terms:
        for doc in relevant_docs[term]:
            if doc not in doc_tf_s:
                doc_tf_s[doc] = defaultdict(int)
            doc_tf_s[doc][term] += 1
        query_tf_s[term] += 1

    avgdl = 0
    for paper in papers_dataset:
        avgdl += len(paper["Title"]) + len(paper["Abstract"])
    avgdl /= N

    if method == "ltc-lnc":
        query_vector = np.zeros(len(title_terms))
        query_norm_factor = 0

        for i in range(len(title_terms)):
            tf = query_tf_s[title_terms[i]]
            if tf == 0:
                query_vector[i] = 0
            else:
                query_vector[i] = 1 + log10(tf)
                query_norm_factor += query_vector[i] ** 2

        query_norm_factor = np.sqrt(query_norm_factor)

        for term in title_terms:
            for doc in relevant_docs[term]:
                document_vector = np.zeros(len(title_terms))
                document_norm_factor = 0

                for i in range(len(title_terms)):
                    tf = doc_tf_s[doc][title_terms[i]]
                    if tf == 0:
                        document_vector[i] = 0
                    else:
                        document_vector[i] = (1 + log10(tf)) * idfs[title_terms[i]]
                        document_norm_factor += document_vector[i] ** 2

                document_norm_factor = np.sqrt(document_norm_factor)
                title_scores[doc] = np.dot(
                    query_vector / query_norm_factor,
                    document_vector / document_norm_factor,
                )

        query_vector = np.zeros(len(abstract_terms))
        query_norm_factor = 0
        for i in range(len(abstract_terms)):
            tf = query_tf_s[abstract_terms[i]]
            if tf == 0:
                query_vector[i] = 0
            else:
                query_vector[i] = 1 + log10(tf)
                query_norm_factor += query_vector[i] ** 2

        query_norm_factor = np.sqrt(query_norm_factor)

        for term in abstract_terms:
            for doc in relevant_docs[term]:
                document_vector = np.zeros(len(abstract_terms))
                document_norm_factor = 0
                for i in range(len(abstract_terms)):
                    tf = doc_tf_s[doc][abstract_terms[i]]
                    if tf == 0:
                        document_vector[i] = 0
                    else:
                        document_vector[i] = (1 + log10(tf)) * idfs[abstract_terms[i]]
                        document_norm_factor += document_vector[i] ** 2

                document_norm_factor = np.sqrt(document_norm_factor)
                abstract_scores[doc] = np.dot(
                    query_vector / query_norm_factor,
                    document_vector / document_norm_factor,
                )

    elif method == "ltn-lnn":
        query_vector = np.zeros(len(title_terms))
        for i in range(len(title_terms)):
            query_vector[i] = 1 + log10(query_tf_s[title_terms[i]] + 1)

        for term in title_terms:
            for doc in relevant_docs[term]:
                document_vector = np.zeros(len(title_terms))
                for i in range(len(title_terms)):
                    document_vector[i] = (
                        1 + log10(doc_tf_s[doc][title_terms[i]] + 1)
                    ) * idfs[title_terms[i]]
                title_scores[doc] = np.dot(query_vector, document_vector)

        query_vector = np.zeros(len(abstract_terms))
        for i in range(len(abstract_terms)):
            query_vector[i] = 1 + log10(query_tf_s[abstract_terms[i]] + 1)

        for term in abstract_terms:
            for doc in relevant_docs[term]:
                document_vector = np.zeros(len(abstract_terms))
                for i in range(len(abstract_terms)):
                    document_vector[i] = (
                        1 + log10(doc_tf_s[doc][abstract_terms[i]] + 1)
                    ) * idfs[abstract_terms[i]]
                abstract_scores[doc] = np.dot(query_vector, document_vector)

    elif method == "okapi25":
        # Setting Hyper parameters (you can change this if you desire!)
        k1 = 1.2
        b = 0.75
        for term in query_terms:
            for doc in relevant_docs[term]:
                for term in title_terms:
                    title_scores[doc] += (
                        log10(N / doc_freqs[doc])
                        * (doc_tf_s[doc][term] * (k1 + 1))
                        / (
                            doc_tf_s[doc][term]
                            + k1 * (1 - b + b * (len(title_terms) / avgdl))
                        )
                    )
                for term in abstract_query:
                    abstract_scores[doc] += (
                        log10(N / doc_freqs[doc])
                        * (doc_tf_s[doc][term] * (k1 + 1))
                        / (
                            doc_tf_s[doc][term]
                            + k1 * (1 - b + b * (len(abstract_terms) / avgdl))
                        )
                    )

    scores = defaultdict(float)
    for doc in title_scores:
        scores[doc] = weight * title_scores[doc] + (1 - weight) * abstract_scores[doc]

    for doc in abstract_scores:
        if doc not in scores:
            scores[doc] = (
                weight * title_scores[doc] + (1 - weight) * abstract_scores[doc]
            )

    # Personalize the search based on preferred_field
    important_articles_lst = [
        important_articles("Rabiee"),
        important_articles("Soleymani"),
        important_articles("Rohban"),
        important_articles("kasaei"),
        important_articles("Sharifi"),
    ]

    references = [
        get_references("Rabiee"),
        get_references("Soleymani"),
        get_references("Rohban"),
        get_references("kasaei"),
        get_references("Sharifi"),
    ]

    if preferred_field is not None:
        s = sum(preferred_field)
        for i in range(len(preferred_field)):
            preferred_field[i] /= s

        for doc in scores:
            for i in range(len(preferred_field)):
                if doc in important_articles_lst[i]:
                    scores[doc] *= preferred_field[i]
                elif doc in references[i]:
                    scores[doc] *= preferred_field[i] * 0.5

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(result) > max_result_count:
        return result[:max_result_count]

    return result


def get_paper_by_id(paper_id: str, papers_ds):
    for paper in papers_ds:
        if paper["ID"] == paper_id:
            return paper
    return None


def hit_algorithm(papers, n):
    """
    Implementing the HITS algorithm to score authors based on their papers and co-authors.

    Parameters
    ---------------------------------------------------------------------------------------------------
    papers: A list of paper dictionaries with the following keys:
            "id": A unique ID for the paper
            "title": The title of the paper
            "abstract": The abstract of the paper
            "date": The year in which the paper was published
            "authors": A list of the names of the authors of the paper
            "related_topics": A list of IDs for related topics (optional)
            "citation_count": The number of times the paper has been cited (optional)
            "reference_count": The number of references in the paper (optional)
            "references": A list of IDs for papers that are cited in the paper (optional)
    n: An integer representing the number of top authors to return.

    Returns
    ---------------------------------------------------------------------------------------------------
    List
    list of the top n authors based on their hub scores.
    """
    # Create a graph of authors and papers (all of the authors and papers represented as nodes, and all of the authors who wrote each paper connected to the corresponding paper node by an edge)
    G = nx.Graph()

    for paper in papers:
        G.add_node(paper["ID"])
        for author in paper["Authors"]:
            G.add_node(author)

    for paper in papers:
        for author in paper["Authors"]:
            G.add_edge(author, paper["ID"])

    for paper in papers:
        for reference in paper["References"]:
            if get_paper_by_id(reference, papers) is not None:
                for author in paper["Authors"]:
                    for second_author in get_paper_by_id(reference, papers)["Authors"]:
                        if author != second_author:
                            G.add_edge(author, second_author)

    # Run the HITS algorithm
    hubs, authorities = nx.hits(G)

    # Create a list of top n authors based on their hub scores
    top_authors = sorted(hubs, key=hubs.get, reverse=True)[:n]
    return top_authors


# call the hit_algorithm function
top_authors = hit_algorithm(papers_dataset, 10)

# print the top authors
print(top_authors)

with open("recommended_papers.json", "r") as fp:
    recommended_papers = json.load(fp)

fields_set = set(
    field
    for user in recommended_papers
    for paper in user["positive_papers"]
    for field in paper["fieldsOfStudy"] or []
)
fields = list(fields_set)

M = len(fields)

train_data, test_data = train_test_split(
    recommended_papers, test_size=0.2, random_state=42
)


def generate_user_vectors(user_positive_papers):
    user_indices = list(range(len(user_positive_papers)))
    user_fields = {"user_index": user_indices}

    for field in fields:
        field_counts = []
        for positive_papers in user_positive_papers:
            count = 0
            for paper in positive_papers:
                if (
                    paper["fieldsOfStudy"] is not None
                    and field in paper["fieldsOfStudy"]
                ):
                    count += 1
            field_counts.append(count)

        user_fields[field] = field_counts

    return pd.DataFrame(user_fields).set_index("user_index")


class CollaborativeFiltering:
    def __init__(self, data: dict, n=10):
        self.data = data
        self.knn = NearestNeighbors(n_neighbors=n, metric="cosine")
        self.n = n

    def fit(self):
        user_vectors = generate_user_vectors(
            [user["positive_papers"] or [] for user in self.data]
        )
        self.knn.fit(user_vectors)
        return self

    def predict(self, user_positive_papers: List[Dict[str, Any]]):
        user_vectors = generate_user_vectors([user_positive_papers])
        distances, indices = self.knn.kneighbors(user_vectors)

        # get recommended papers from the k nearest neighbors
        result = [
            paper["paperId"]
            for similar_user_id in indices[0]
            for paper in self.data[similar_user_id]["recommendedPapers"]
        ]
        # get the most common recommended papers (10)
        result = [paper_id for paper_id, _ in Counter(result).most_common(self.n)]

        return result[: self.n]


collaborative_recommends = CollaborativeFiltering(train_data).fit()


class ContentBasedRecommendation:
    def __init__(self, all_recommended_papers: List[Dict[str, Any]]):
        self.data = all_recommended_papers
        self.tf_idf = TfidfVectorizer()
        self.recommended_vecors = None
        self.titles = [paper["title"] for paper in self.data]

    def fit(self):
        self.recommended_vecors = self.tf_idf.fit_transform(self.titles)
        return self

    def predict(self, user_positive_papers: List[Dict[str, Any]]):
        titles = [paper["title"] for paper in user_positive_papers]
        titles_vector = self.tf_idf.transform(titles).mean(axis=0)

        # calculate the similarities between the user's positive papers and all the recommended papers
        similarities: np.ndarray = np.asarray(
            titles_vector @ self.recommended_vecors.T
        ).flatten()

        # get the top 10 most similar papers
        result = [self.data[i]["paperId"] for i in similarities.argsort()[-10:][::-1]]
        return result


content_based_recommendation = ContentBasedRecommendation(
    [
        paper
        for user in recommended_papers
        for paper in (user["recommendedPapers"] or [])
    ]
).fit()


from typing import List, Dict


def create_bigram_index(texts: List[str]) -> Dict[str, List[str]]:
    """
    Creates a bigram index for the spell correction

    Parameters
    ----------
    texts: List[str]
        The titles and abstracts of articles

    Returns
    -------
    dict
        A dictionary of bigrams and their occurence
    """
    bigram: Dict[str, List[str]] = {}

    # TODO: Create the bigram index here

    for text in texts:
        for i in range(len(text) - 1):
            bigram[text[i : i + 2]] = bigram.get(text[i : i + 2], []) + [text]

    return bigram


def get_bigrams(text: str) -> List[str]:
    """
    Generate a list of bigrams from a given text

    Parameters
    ----------
    text : str
        The input text

    Returns
    -------
    List[str]
        A list of bigrams
    """
    return [text[i] + text[i + 1] for i in range(len(text) - 1)]


def jacard_similarity(str1: str, str2: str) -> float:
    """Computes the Jaccard similarity between two strings"""
    set1 = set(get_bigrams(str1))
    set2 = set(get_bigrams(str2))
    return len(set1.intersection(set2)) / len(set1.union(set2))


def min_edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]


from collections import Counter


def correct_text(
    text: str, bigram_index: Dict[str, List[str]], similar_words_limit: int = 20
) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text

    Returns
    str
        The corrected form of the given text
    """
    corrected_text = ""
    words = text.split()

    for i in range(len(words)):
        bigrams = get_bigrams(words[i])
        print(words[i], bigrams)
        candidates = []
        for bigram in bigrams:
            if bigram in bigram_index:
                candidates += bigram_index[bigram]

        if len(candidates) == 0:
            corrected_text += words[i] + " "
            continue

        candidates = Counter(candidates).most_common(similar_words_limit)
        candidates = [candidate[0] for candidate in candidates]
        candidates = sorted(
            candidates, key=lambda x: jacard_similarity(words[i], x), reverse=True
        )
        # find the most similar word from candidates using minimum edit distance
        min_distance = float("inf")
        corrected_word = words[i]
        for candidate in candidates:
            distance = min_edit_distance(words[i], candidate)
            if distance < min_distance:
                min_distance = distance
                corrected_word = candidate
        corrected_text += corrected_word + " "

    return corrected_text.strip()


bigram_index = create_bigram_index(
    [word for paper in papers_dataset for word in paper["Title"].split()]
    + [word for paper in papers_dataset for word in paper["Abstract"].split()]
)
