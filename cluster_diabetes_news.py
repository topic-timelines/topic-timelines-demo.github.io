"""Example code for how to run: cluster_transformers.encode_sentences.
"""
import os.path
import sys
from nltk.corpus import stopwords

import cluster_transformers



# The filenamne has the following format, starting with a date.
# 2005-08-04T14-31-00+02-00_%URL%_https__www.svt.se_nyheter_lokalt_vasterbotten_din-halsa-paverkas-av-farfars-matvanor.txt
# The date-part will be returned.
def get_label(doc_path):
    base_name = os.path.basename(doc_path)
    date_str = base_name[:10]
    return date_str
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("A path to a folder with the .txt-files to cluster must be provided as an argument")
        sys.exit()
        
    corpus = sys.argv[1]
    
    numbers = [str(el) for el in range(0, 5000)]
    more_numbers = ["0"+el for el in numbers]
    to_remove = ["säger", "kommer", "tycker", "andra", "finns", "även", "också", "enligt", "fick", "bara", "in", "visar", "ska", "bland", "får", "många", "berättar", "flera", "gå", "få", "genom", "gör", "diabetes", "år", "procent", "vill", "ta", "svt", "fått", "går", "lite", "diabetiker", "sverige", "svensk", "svenska", "svenskt", "mer", "väldigt", "mannen", "kvinnan"] + stopwords.words('swedish') + numbers + more_numbers + ["000"]
    
    print("Will cluster .txt files in: ", corpus)
    
    cluster_transformers.encode_sentences(
        main_path = corpus,
        output_dir="diabetes_svt",
        n_clusters=0.17,
        transform_filename_method=get_label,
        nr_of_words_to_show=50,
        min_occ_in_corpus_for_keyword=100,
        min_nr_of_text_in_cluster=20,
        words_to_remove_before_clustering = to_remove,
        get_clustering_model = cluster_transformers.get_agglomerative_distance_threshold,
        high_level_cluster_threshold=0.28)
    
    


