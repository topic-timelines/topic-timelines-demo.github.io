"""Clusters .txt files into topics using Sentence Transformers.

The .txt files are clustered, according to configuration specification provided by the user. 
The embedding vectors encoded by the Sentence Transformers chosen by the user,
are clustered with the clusterings method, also chosen by the user. 
The result are saved in a format that can be used by the plot_timeline_from_files.py

Typical usage example:

def get_label(doc_path):
    base_name = os.path.basename(doc_path)
    date_str = base_name[:10]
    return date_str
    
cluster_transformers.encode_sentences(
    main_path = "corpus_path,
    output_dir="diabetes_svt",
    n_clusters=0.17,
    transform_filename_method=get_label,
    nr_of_words_to_show=50,
    min_occ_in_corpus_for_keyword=100,
    min_nr_of_text_in_cluster=20,
    put_small_clusters_in_outlier_category=False,
    exclude_outlier_label=False,
    recalculate_centroids_after_merge=False,
    get_clustering_model = cluster_transformers.get_agglomerative_distance_threshold,
    high_level_cluster_threshold=0.28)
"""
import pickle
from collections import Counter
import glob
import os
import sys
import json

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt



#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.centroid.html


DEAULT_TRANSFORMER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_NR_OF_WORDS_TO_SHOW = 20
OUTLIER_NUMBER = -1

################################
# Custrom models for clustering
###############################
def default_get_cluster_model(n_clusters, min_nr_of_text_in_cluster):
    model = AgglomerativeClustering(linkage="average",
                                n_clusters=n_clusters,
                                compute_distances=False,
                                metric="cosine")
    return model
    
# n_clusters is used as distance_threshold here
def get_agglomerative_distance_threshold(n_clusters, min_nr_of_text_in_cluster):
    model = AgglomerativeClustering(linkage="average",
                                distance_threshold=n_clusters,
                                compute_distances=False,
                                metric="cosine",
                                compute_full_tree = True,
                                n_clusters=None)
    return model
    
def get_dbscan(n_clusters, min_nr_of_text_in_cluster):
    db = DBSCAN(eps=n_clusters, min_samples=min_nr_of_text_in_cluster, metric="cosine")
    return db
    
def get_hdbscan(n_clusters, min_nr_of_text_in_cluster):
    #https://hdbscan.readthedocs.io/en/latest/faq.html#q-most-of-data-is-classified-as-noise-why
    db = HDBSCAN(min_cluster_size=min_nr_of_text_in_cluster, metric="cosine", cluster_selection_method="eom", cluster_selection_epsilon=n_clusters, max_cluster_size=500) # #"eom"
    return db
    
def get_kmeans(n_clusters, min_nr_of_text_in_cluster):
    km = KMeans(n_clusters=n_clusters, random_state=1, n_init=10, max_iter=500)
    return km
    
############################################
# Helper functions for creating the clusters
############################################

def merge_one_label_category(merged_labels, min_nr_of_text_in_cluster, centroids_dict, put_small_clusters_in_outlier_category):
    count_labels = Counter(merged_labels)
    len_items = len(count_labels.items())
    if len_items % 100 == 0:
        print(f"Sorting to find too small topics among {len_items} topics")
    sorted_cluster_results = sorted((value, key) for key, value in count_labels.items() if key != OUTLIER_NUMBER)
    if sorted_cluster_results: #if there are other categories than the outlier one, find the one with smallest number of items
        min_value, min_cluster_label = sorted_cluster_results[0]
        
    if not sorted_cluster_results or min_value >= min_nr_of_text_in_cluster: # smallest cluster is large enough
        return merged_labels, centroids_dict, None # no replace labels
        
    #print(f"{min_value}, ", end='')

    # Remove too small cluster from cluster dict
    centroid_for_cluster_to_remove = centroids_dict[min_cluster_label]
    centroids_dict.pop(min_cluster_label)
        
    if put_small_clusters_in_outlier_category:
        label_to_replace_with = OUTLIER_NUMBER
    else: # Compute the cosine similarity from the centroid of the cluster to remove, to all other centroids
        cosine_similarity_values = []
        for label_class, centroid in centroids_dict.items():
            # Wikipedia explanation for cosine similarity:
            # For example, two proportional vectors have a cosine similarity of +1, two orthogonal vectors have a similarity of 0,
            # and two opposite vectors have a similarity of −1. In some contexts, the component values of the vectors cannot be
            # negative, in which case the cosine similarity is bounded in [0,1]
            cos_sim = cosine_similarity([centroid_for_cluster_to_remove], [centroid])[0][0]
            cosine_similarity_values.append((cos_sim, label_class))
        sorted_cos_values = sorted(cosine_similarity_values, reverse=True) # Sort, with the largest similarity first
        label_to_replace_with = sorted_cos_values[0][1] # Replace with the label that is most similar
    
    new_merged_labels = []
    for el in merged_labels:
        if el == min_cluster_label:
            new_merged_labels.append(label_to_replace_with) # Replace with new label
        else:
            new_merged_labels.append(el) # Keep the original label
            
    #print("Replaced ", min_cluster_label, "with", label_to_replace_with)
    return new_merged_labels, centroids_dict, min_cluster_label

def get_centroid_outlier(labels, embeddings):
    vector_list = []
    for label, embedding in zip(labels, embeddings):
        if label == OUTLIER_NUMBER:
            vector_list.append(embedding)
    assert(len(vector_list) > 0)
    c = np.mean(vector_list, axis=0)
        
    return c
    
def get_centroids_dict(labels, embeddings):
    vector_dict = {}
    for label, embedding in zip(labels, embeddings):
        if label not in vector_dict:
            vector_dict[label] = []
        vector_dict[label].append(embedding)
        
    centroids_dict = {}
    for key in sorted(vector_dict.keys()):
        c = np.mean(vector_dict[key], axis=0)
        centroids_dict[key] = c
        
    return centroids_dict
  

def get_tsne_for_centroids(centroids_dimensionallity_reduced_embeddings):
    tsne_model = TSNE(n_components=1, random_state=0, perplexity=min(5, len(centroids_dimensionallity_reduced_embeddings)), metric="cosine")
    tsne_reduced = tsne_model.fit_transform(centroids_dimensionallity_reduced_embeddings)
    min_tsne_reduced = float(min(tsne_reduced))
    tsne_reduced_positive = [float(x) - min_tsne_reduced for x in tsne_reduced]
    max_tsne_reduced = max(tsne_reduced_positive)
    
    tsne_reduced_normalised = sorted([(x/max_tsne_reduced, i+1) for (i, x) in enumerate(tsne_reduced_positive)])
    tsne_ordered = [x[1] for x in tsne_reduced_normalised]
    viridis = plt.cm.viridis
    colour_values = viridis([x[0] for x in tsne_reduced_normalised])
    
    return tsne_ordered, colour_values
    
def get_high_level_clusters(high_level_cluster_threshold, centroids_dict, n_reduced_dimensions, sort_high_level_clusters):
   # Meta high-level clustering
    db = AgglomerativeClustering(metric="cosine", distance_threshold=high_level_cluster_threshold, n_clusters=None, linkage="average")
    centroid_labels_external_nrs = []
    centorid_embeddings = []
    for external_cluster_nr_zero_based, centroid_label in enumerate(sorted(centroids_dict.keys())):
        if centroid_label != OUTLIER_NUMBER: # Don't include the outlier cluster in the high level clustering
            external_cluster_nr = external_cluster_nr_zero_based + 1
            centroid_labels_external_nrs.append(external_cluster_nr)
            centorid_embeddings.append(centroids_dict[centroid_label])
        
    if len(centorid_embeddings) > 1: # Otherwise it will not work if only one cluster is created
    
        if n_reduced_dimensions:
            pca = PCA(n_components=min(n_reduced_dimensions, len(centorid_embeddings)), svd_solver="full") # Reduce the dimension to the number of topics extracted, or smaller
            centroids_dimensionallity_reduced_embeddings = pca.fit_transform(centorid_embeddings)
            
        else:
            centroids_dimensionallity_reduced_embeddings = centorid_embeddings
    
        # Do the high-level clustering
        centroid_clusters = db.fit(centroids_dimensionallity_reduced_embeddings)
        centroid_labels_external_clusters = centroid_clusters.labels_
        
        # Also make a tsne projektion (not using the high-level clusters)
        tsne_ordered, colour_values = get_tsne_for_centroids(np.array(centroids_dimensionallity_reduced_embeddings))
        
        
    else:
        centroid_labels_external_clusters = [1]
        tsne_ordered = [1]
        
    
    timelines_external_cluster_format_dict = {}
    for external_cluster_nr, high_level_cluster_nr in zip(centroid_labels_external_nrs, centroid_labels_external_clusters):
        if high_level_cluster_nr not in timelines_external_cluster_format_dict:
            timelines_external_cluster_format_dict[high_level_cluster_nr] = []
        timelines_external_cluster_format_dict[high_level_cluster_nr].append(external_cluster_nr)
        
    if OUTLIER_NUMBER in centroids_dict:
        timelines_external_cluster_format_dict[OUTLIER_NUMBER] = [1]
    
    # Make a tsne projektion for the high-level clusters, to order these
    centroids_for_high_level_clusters = get_centroids_dict(centroid_labels_external_clusters, centroids_dimensionallity_reduced_embeddings)
    high_level_centroid_embeddings = []
    for high_level_centroid_label in sorted(centroids_for_high_level_clusters.keys()):
        high_level_centroid_embeddings.append(centroids_for_high_level_clusters[high_level_centroid_label])
            
    high_level_centroid_embeddings = np.array(high_level_centroid_embeddings)
    print(len(high_level_centroid_embeddings))
    tsne_ordered_high_level_clusters, colour_values_high_level_clusters = get_tsne_for_centroids(high_level_centroid_embeddings)
    print("timelines_external_cluster_format_dict", timelines_external_cluster_format_dict)
    print("tsne_ordered_high_level_clusters", tsne_ordered_high_level_clusters)
    
    timelines_external_cluster_format_list = []
    if sort_high_level_clusters:
        for high_level_cluster_nr in tsne_ordered_high_level_clusters:
            label_list = timelines_external_cluster_format_dict[high_level_cluster_nr-1] #Because Labels from clustering starts with 1?
            timelines_external_cluster_format_list.append(label_list)
            
    else: # TODO: This option is saved for backward compatability. Perhaps remove later
        outlier_list = []
     # Clusters with only one member is counted as an outlier, and placed last
        for label_list in timelines_external_cluster_format_dict.values():
            if len(label_list) > 1:
                timelines_external_cluster_format_list.append(label_list)
            else:
                outlier_list.extend(label_list)
        timelines_external_cluster_format_list.extend(outlier_list)
        
        
    return timelines_external_cluster_format_list, tsne_ordered, colour_values, colour_values_high_level_clusters


#############################################################
# Helper functions for reading and pre-processing the texts
#############################################################

def remove_words(text, words_to_remove_before_clustering):
    if not words_to_remove_before_clustering:
        return text
    tokens = word_tokenize(text)
    new_text = " ".join([word for word in tokens if word not in words_to_remove_before_clustering and word.lower() not in words_to_remove_before_clustering])
    return new_text


# Will add content to to_cluster
def read_files(main_path, to_cluster, to_cluster_raw_texts, preprocess_method, words_to_remove_before_clustering, transform_filename_method, exclude_file_method):

    sub_folders = os.path.join(os.path.join(main_path,"*"))
    files = sorted(glob.glob(os.path.join(sub_folders, "*.txt")))
    if len(files) == 0:
        print("Found no subfolders when searching as: ", sub_folders)
        print("Instead, searching for txt files directly in main path: ", main_path)
        files = sorted(glob.glob(os.path.join(main_path, "*.txt")))
        if len(files) == 0:
            print("Still no txt files found. Exiting")
            sys.exit()
            
    before_len = len(files)
    if exclude_file_method:
        files = [f for f in files if not exclude_file_method(f)]
        if len(files) < before_len:
            print("Files excluded: ", before_len-len(files))
    
    # Just to check for errors in transform_filename_method
    if transform_filename_method:
        for file in files:
            str_date = transform_filename_method(file)
            np.datetime64(str_date)
            
    if len(files) < 1:
        print(main_path, "not found. Exiting")
        sys.exit()

    print("Will cluster ", len(files), " nr of files")
    print("Reading files")
    
    for file_nr, file_name in enumerate(files):
        with open(file_name) as fn:
            text = fn.read()
            to_cluster_raw_texts.append(text.replace("\n", " "))
            if preprocess_method:
                text = preprocess_method(text)
            text = remove_words(text, words_to_remove_before_clustering)
            to_cluster.append((os.path.basename(file_name), text, main_path))
            
        if file_nr % 1000 == 0:
            print(int(100*file_nr/len(files)), "%")
          
          
###################################################################################################
# Helper functions for constructing typical sentences for each topic and and labels for each topic
####################################################################################################

def get_typical_sentences_for_topics(merged_labels, to_cluster, corpus_path, cos_texts, to_cluster_raw_texts):

    # Collect all texts for a topic, for the specific path, to be able to extract keywords
    texts_m_paths = [(filename, m_path, raw_text, text) for ((filename, text, m_path), raw_text) in zip(to_cluster, to_cluster_raw_texts)]
    text_dict = {}
    for label, cos, (filename, m_path, raw_text, text) in zip(merged_labels, cos_texts, texts_m_paths):
        
        if label not in text_dict:
            text_dict[label] = []
        if m_path == corpus_path: # Here the sorting depending on the paths is carried out
            text_dict[label].append((cos, raw_text, text))
            
    most_typical_texts_dict = {}
    for label, items in text_dict.items():
        typical_texts_for_label = "\t".join([raw_text for (cos, raw_text, text) in sorted(items, reverse=True)][:20])
        most_typical_texts_dict[label] = typical_texts_for_label
        
        
    return most_typical_texts_dict
        
def get_labels_for_topics(merged_labels, to_cluster, stop_words, min_occ_in_corpus_for_keyword, nr_of_words_to_show, corpus_path, min_nr_of_text_in_cluster):
    
    # Collect all texts for a topic, for the specific path, to be able to extract keywords
    texts_m_paths = [(text, m_path) for (filename, text, m_path) in to_cluster]
    text_dict = {}
    for label, (text, m_path) in zip(merged_labels, texts_m_paths):
        if label not in text_dict:
            text_dict[label] = []
        if m_path == corpus_path: # Here the sorting depending on the paths is carried out
            text_dict[label].append(text)

    text_list = []
    for key in sorted(text_dict.keys()):
        text_list.append(" ".join(text_dict[key]))
    
    # Create a set with frequent words in the entire corpus (for the path)
    frequent_words = set()
    all_texts = " ".join(text_list)
    count_vectorizer = CountVectorizer(stop_words = stop_words, ngram_range=(1, 2))
    X_count = count_vectorizer.fit_transform([all_texts])
    for transformed in X_count:
        score_vec = transformed.toarray()[0]
        for score, word in zip(score_vec, count_vectorizer.get_feature_names_out()):
            if score > min_occ_in_corpus_for_keyword:
                sp = word.split(" ")
                if len(sp) == 2 and sp[0] == sp[1]:
                    frequent_words.add(sp[0])
                else:
                    frequent_words.add(word)
            
    topic_keywords_dict = {}
    vectorizer = TfidfVectorizer(stop_words = stop_words, ngram_range=(1, 2), vocabulary=frequent_words)
    X = vectorizer.fit_transform(text_list)
        
    for transformed, label, text_l in zip(X, sorted(text_dict.keys()), text_list):
        if len(text_l) < min_nr_of_text_in_cluster:
            topic_keywords_dict[label] = ["-"]*nr_of_words_to_show
        else:
            score_vec = transformed.toarray()[0] #Get a vector with scores for the label, for each of the words in the corpus
            scores_with_words = [(score, word) for score, word in zip(score_vec, vectorizer.get_feature_names_out())]
        
            words = [word for (score, word) in sorted(scores_with_words, reverse=True)]
    
            topic_keywords_dict[label] = words[:nr_of_words_to_show]
        
    print("TEXT_DICT.keys()", text_dict.keys())
    return topic_keywords_dict
        
###################################################################################################
# Helper functions for writing output to files
####################################################################################################


def write_to_file_as_str(file_name, content):
    print("Save high-level clusters in ", file_name)
    with open(file_name, "w") as fn:
        fn.write(json.dumps(content))
            
# As a default, clusters with less than min_nr_of_text_in_cluster will be moved to another cluster with the closest centroid
# If 'put_small_clusters_in_outlier_category' is set to True, they will instead all be moved to an outlier-category.
# If 'exclude_outlier_label' is set to True, this outlier category will not be included in the results.
def encode_sentences(main_path, output_dir, n_clusters,
        transformer_name=DEAULT_TRANSFORMER_MODEL,
        transform_filename_method=None,
        nr_of_words_to_show=DEFAULT_NR_OF_WORDS_TO_SHOW,
        stopwords=[],
        min_occ_in_corpus_for_keyword=2,
        min_nr_of_text_in_cluster=2,
        words_to_remove_before_clustering=[],
        preprocess_method=None,
        get_clustering_model=default_get_cluster_model,
        put_small_clusters_in_outlier_category = False,
        exclude_outlier_label=False,
        recalculate_centroids_after_merge=False,
        fixed_global_min_cos=False,
        high_level_cluster_threshold=0.5,
        pickled_files_name=None,
        file_name_to_save_text_and_embeddings=None,
        exclude_file_method=None,
        n_reduced_dimensions=200,
        sort_high_level_clusters=True):
    """The main function for clustering files found in "main_path" and write the results in "output_dir".

    Args:
    main_path: The path were the .txt-files are located. If there are subdirectories, .txt are search for in these. 
    If there are no subdirectories, .txt files directly in main_path are searched for. If main_path is a list, 
    it is interpreted as list of paths to different sub-corpora. Seperate output_files will then be generated for each subcorpus,
    but the clustering is performed on all files together. 
    output_dir: Where to save the results.
    n_clusters: Configuration parameter to the clustering method. Could, e.g. be number of clusters or cosine distance. (Note that for Agglomerative clustering it is cosine *distance* that is used, i.e., a measure defined as 1-cosine similarity)
    transformer_name: The transformer model to use. Default is "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
    transform_filename_method: The method for tranforming the name of the .txt-file to a date. (A default method will be added).
    nr_of_words_to_show: The number of words to include in the topic labels. 20 is default.
    stopwords: Words to exclude when creating lables. NOTE: These are not the words that are removed before encoding the texts. 
    Default is empty list.
    These only applies to words extracted for cluster lables.
    min_occ_in_corpus_for_keyword: Minimum number of times a word must occur in the corpus to be part of a cluster label. Default is 2.
    min_nr_of_text_in_cluster: Minimum number of texts in a cluster for this cluster to be retained. Default is 2 texts.
    words_to_remove_before_clustering: Words that are removed before the texts are encoded. 
    Default is empty list.
    preprocess_method: Method for pre-processing the texts. default = None
    get_clustering_model= The method to use for clustering. The default is default_get_cluster_model,
    put_small_clusters_in_outlier_category: If clusters that are too small, should be put in their own outlier category. Default is False. 
        (If False, they are merged with their closes neighbouring cluster.)
    exclude_outlier_label: If the outlier category is to be excluded from the result. Default is False,
    recalculate_centroids_after_merge: If is possible to recalculate cluster centroids after outlier clusters have been merge with neigbours. Default is False,
    fixed_global_min_cos: Subtract this number from all the cosine similiarity numbers, that forms the output which defines the prominence for a topic in a text. Default is False. When False, what will be substracted is then instead the minimum cosine similiarity for all texts in the corpus. (A coisine similarity +1 is always used, to avoid negative numbers.)
    high_level_cluster_threshold: The maximum cosine distance when high-level clusters are to be coninued to be merge together. (Note that cosine distance is used, i.e., 1-cosine similarity) Default=0.5
    pickled_files_name: The file name for pickled texts and encodded embeddings. When giving this filename, the texts and codings saved here will be used, and the code will start direclty by clustering embeddings. WARNING: If any of the parameter settings for reading the texts and encoding the embeddings have been changed, these changes will not be put into effect, as old ones are used. Default is None.
    file_name_to_save_text_and_embeddings: If this is given, the read texts, and the embeddings created from these texts are saved in this file. Thereby, this file name can be given to pickled_files_name, and the reading of text files and encoding does not have to be done again. Default is None. (Note that not both file_name_to_save_text_and_embeddings and pickled_files_name can be given.)
    exclude_file_method: method, default = None.
        A method for deciding files to be included.
    n_reduced_dimensions: int, default=200
        Dimensions to use for a PCA dimensionality reduction before the transformers are clustered. If None, no PCA dimensionality reduction is carried out.
    sort_high_level_clusters: boolean, default=True
        Whether to sort the high-level clusters (by making a TSNE projection of their centroids).
    """
    
    
    #https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MKL_THREADING_LAYER"]="GNU"
    os.environ["MKL_THREADING_LAYER"]="TBB"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # If main_path is a list of sub-corpora,
    # Then also the method for transforming filename to date must be a list of methods for each corpus
    if transform_filename_method and (type(transform_filename_method) == list or type(main_path) == list):
        assert(type(transform_filename_method) == list)
        assert(type(main_path) == list)
        assert(len(transform_filename_method) == len(main_path))
        
    # Main path could either be a path, or a list of paths. In the first case, transform to list with one element,
    # The same for transform_filename_method
    if type(main_path) != list:
        main_path = [main_path]
    if not type(transform_filename_method) == list:
        transform_filename_method = [transform_filename_method]

    if pickled_files_name is not None and file_name_to_save_text_and_embeddings is not None:
        print("Can't both provive a 'pickled_files_name' and a 'file_name_to_save_text_and_embeddings'")
        sys.exit()
        
    if file_name_to_save_text_and_embeddings is not None:
        if os.path.exists(file_name_to_save_text_and_embeddings):
            print(f"There is already a file with the name '{file_name_to_save_text_and_embeddings}'. To avoid overwriting it, either provide another filename to pickle the texts and embeddings, or delete the existing one.")
            sys.exit()
    # Read the files
    if not pickled_files_name:
        print("No pickled data, so reading texts from: ", main_path)
        
        to_cluster = []
        to_cluster_raw_texts = [] # Without preprocessing
        
        for corpus_path, transform_filename_method_element in zip(main_path, transform_filename_method):
            print("Reading files from: ", corpus_path)
            read_files(corpus_path, to_cluster, to_cluster_raw_texts, preprocess_method, words_to_remove_before_clustering, transform_filename_method_element, exclude_file_method)
        
          
        print("Finished reading texts")
        
        # Encode text
        print("Using transformer:", transformer_name)
        print("Loading model")
        model = SentenceTransformer(transformer_name)
            
        print("Encoding embeddings")
        texts = [text for (filename, text, m_path) in to_cluster]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)
        print("type(embeddings)", type(embeddings))
        print("Embeddings encoded")
        print("Created embeddings of shape", embeddings.shape)
        
        if file_name_to_save_text_and_embeddings:
            with open(file_name_to_save_text_and_embeddings, 'wb') as save_file:
                pickle.dump((to_cluster, to_cluster_raw_texts, embeddings), save_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Using saved texts and encoded embeddings in: ", pickled_files_name)
        
        with open(pickled_files_name, 'rb') as load_file:
            to_cluster, to_cluster_raw_texts, embeddings = pickle.load(load_file)
 
    print("Clustering")
    if n_reduced_dimensions:
        print("Reduce dimensions")
        pca = PCA(n_components=min(n_reduced_dimensions, len(embeddings)), svd_solver="full") # If there are very few texts that are to be extract topics from, use the number of texts as the dimension
        dimensionallity_reduced_embeddings = pca.fit_transform(embeddings)
    else:
        dimensionallity_reduced_embeddings = embeddings
    clustering = get_clustering_model(n_clusters, min_nr_of_text_in_cluster).fit(dimensionallity_reduced_embeddings)
    
    # Collect all embeddings for a topic, to be able to create a centroid
    print("Computing centroids")
    centroids_dict = get_centroids_dict(clustering.labels_, embeddings)
    
    # Successively merge all categories that are too small, until no too small categories are found anymore
    merged_labels = clustering.labels_
    print("Replacing rare categories")
    replaced_label = "not none"
    nr_of_too_small_clusters = 0
    while replaced_label is not None:
        merged_labels, centroids_dict, replaced_label = merge_one_label_category(merged_labels, min_nr_of_text_in_cluster, centroids_dict, put_small_clusters_in_outlier_category)
        nr_of_too_small_clusters+=1
    print()
    print(f"Number of clusters that were too small: {nr_of_too_small_clusters} ")

    # Always recalculate the centroid for the outlier category (if such a category is used)
    if put_small_clusters_in_outlier_category: # TODO: Add test case for put_small_clusters_in_outlier_category=True
        centroids_dict[OUTLIER_NUMBER] = get_centroid_outlier(merged_labels, embeddings)
    # For the others, only recalculate if that is configured to do
    if recalculate_centroids_after_merge:
        print("Recalculating centroids")
        centroids_dict = get_centroids_dict(merged_labels, embeddings)
        
    all_labels_list = sorted(list(set(merged_labels)))
    if exclude_outlier_label:
        all_labels_list.remove(OUTLIER_NUMBER)
    print("all_labels_list", all_labels_list)

 
    # Construct the high-level-clusters
    high_level_clusters, tsne_ordered, colour_values, colour_values_high_level_clusters = get_high_level_clusters(high_level_cluster_threshold, centroids_dict, n_reduced_dimensions, sort_high_level_clusters)
    # And write the high-level clusters to a file, one time (and the same content, but different filename for each path)
    for corpus_path in main_path:
        write_to_file_as_str(os.path.join(output_dir, os.path.basename(corpus_path + "_high_level_clusters.txt")), high_level_clusters)
        write_to_file_as_str(os.path.join(output_dir, os.path.basename(corpus_path + "_ordered_clusters.txt")), tsne_ordered)
        write_to_file_as_str(os.path.join(output_dir, os.path.basename(corpus_path + "_colour_values.txt")), colour_values.tolist())
        write_to_file_as_str(os.path.join(output_dir, os.path.basename(corpus_path + "_colour_values_high_level_clusters.txt")), colour_values_high_level_clusters.tolist())
            
    # Wikipedia explanation for cosine similarity:
    # For example, two proportional vectors have a cosine similarity of +1, two orthogonal vectors have a similarity of 0,
    # and two opposite vectors have a similarity of −1. In some contexts, the component values of the vectors cannot be
    # negative, in which case the cosine similarity is bounded in [0,1]
    global_min_cos = 2.0 # Save global min cosine distance for centroid for text
    cos_texts = [] # Store cosine similarity for each text, to the centroid of its cluster
    for label, embedding in zip(merged_labels, embeddings):
        cos = cosine_similarity([embedding], [centroids_dict[label]])[0][0] + 1 # Make it always over 0
        cos_texts.append(cos)
        if cos < global_min_cos:
            if exclude_outlier_label is True and label == OUTLIER_NUMBER:
                pass
            else:
                global_min_cos = cos

    global_min_cos = global_min_cos - 0.001 # so that the smallest will not be 0, when subtracting
    print("Global minimal cosine simliarity + 1 (two proportional vectors have a cosine similarity of +1, two orthogonal vectors have a similarity of 0 and and two opposite vectors have a similarity of −1)", global_min_cos)
    if fixed_global_min_cos:
        global_min_cos = fixed_global_min_cos



    # Extract typical sentences for topics and labels for topics
    # Do it separately for each corpus in main_path
    typical_sentence_per_path_dict = {}
    topic_keywords_per_path_dict = {}
    for corpus_path in main_path:
        topic_keywords_dict = get_labels_for_topics(merged_labels, to_cluster, stop_words = stopwords, min_occ_in_corpus_for_keyword=min_occ_in_corpus_for_keyword, nr_of_words_to_show=nr_of_words_to_show, corpus_path=corpus_path, min_nr_of_text_in_cluster=min_nr_of_text_in_cluster)
        topic_keywords_per_path_dict[corpus_path] = topic_keywords_dict
        
        topic_typical_sentences_dict = get_typical_sentences_for_topics(merged_labels, to_cluster, corpus_path, cos_texts, to_cluster_raw_texts)
        typical_sentence_per_path_dict[corpus_path] = topic_typical_sentences_dict

    # Create the output strings, which contain the results of the clustering
    path_to_transform_method_dict = {pa:me for (pa, me) in zip(main_path, transform_filename_method)}
    clustering_output_strs_to_subcorpus_mapping = {} # Map the strings to each subcorpus in main_path
    mapping_old_nr_to_new_nr = {}
    for (filename, text, m_path), label, cos in zip(to_cluster, merged_labels, cos_texts):
        output_row = []
        
        if exclude_outlier_label is True and label == OUTLIER_NUMBER:
            continue
        
        cos = cos - global_min_cos # Cut-off the lower band to make it more expressive
            
        if not fixed_global_min_cos:
            assert(cos > 0.0)
        else:
            if cos <= 0.01:
                print("Min size used for label. Perhaps decrease 'fixed_global_min_cos' ", cos)
                cos = 0.01
        
        if transform_filename_method:
            date = path_to_transform_method_dict[m_path](filename)
        else:
            # TODO: Add default behavious, for transforming filename to date, e.g. just remove .txt
            raise NotImplementedError("transform_filename_method must currently always be provided. No default method for transforming from filename to date")
        
        labels_list = [0]*len(all_labels_list)
        keyword_list = ["-"]*len(all_labels_list)
        for nr, label_category in enumerate(all_labels_list):
            if label == label_category:
                labels_list[nr] = cos
                
                # Just consistency check
                if label_category in mapping_old_nr_to_new_nr:
                    assert(mapping_old_nr_to_new_nr[label_category] == nr)
                else:
                    mapping_old_nr_to_new_nr[label_category] = nr
                    
                topic_words_in_text = []
                topic_keywords_dict = topic_keywords_per_path_dict[m_path]
                for word in topic_keywords_dict[label_category]:
                    if word.lower() in text.lower(): # TODO: Could be done more robustly. Currently not used by visualisation
                        topic_words_in_text.append(word.replace(" ","_"))
                
                
                topic_words_in_text_str = "/".join(topic_words_in_text)
                keyword_list[nr] = topic_words_in_text_str
                
        output_row.append(filename)
        output_row.append(date)
        output_row.extend(labels_list)
        output_row.append("") # Is used by topic-timelines as an indication that cluster strength values are over
        output_row.extend(keyword_list)
        

        to_write = "\t".join([str(el) for el in output_row]) + "\n"
        
        # Make one list of cluster-info to write for each main path
        if m_path not in clustering_output_strs_to_subcorpus_mapping:
            clustering_output_strs_to_subcorpus_mapping[m_path] = []
        clustering_output_strs_to_subcorpus_mapping[m_path].append(to_write)
        
    # Write cluster results for each sub-corpus
    for corpus_path in main_path:
        output_file_name = os.path.join(output_dir, os.path.basename(corpus_path) + "_clustered.txt")
        print("Saves result in", output_file_name)
       
        with open(output_file_name, "w") as ofn:
            for line in clustering_output_strs_to_subcorpus_mapping[corpus_path]:
                ofn.write(line)
       
    print("created", len(all_labels_list), "topics")
    print("all_labels_list", all_labels_list)
    
    # Write the typical texts and labels for each topic
    for corpus_path in main_path:
        cluster_labels = []
        for nr, label_category in enumerate(all_labels_list):
            assert(mapping_old_nr_to_new_nr[label_category] == nr)
            to_write = "\t".join([el.replace(" ", "_") for el in topic_keywords_per_path_dict[corpus_path][label_category]]) + "\n"
            cluster_labels.append(to_write)

        output_file_name_clusters = os.path.join(output_dir, os.path.basename(corpus_path) + "_cluster_labels.txt")
        print("Save cluster labels in", output_file_name_clusters)
        with open(output_file_name_clusters, "w") as ofnc:
            for cluster_label in cluster_labels:
                ofnc.write(cluster_label)

        typical_texts = []
        for nr, label_category in enumerate(all_labels_list):
            typical_texts.append(typical_sentence_per_path_dict[corpus_path][label_category])
        output_file_name_typical_texts = os.path.join(output_dir, os.path.basename(corpus_path) + "_typical_texts.txt")
        print("Save typical texts in", output_file_name_typical_texts)
        with open(output_file_name_typical_texts, "w") as ofnc:
            for t in typical_texts:
                ofnc.write(t + "\n")

