# MIT License
#
# Copyright (c) 2024, Justin Randall, Smart Interactive Transformations Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from interactovery import Utils, OpenAiWrap, ClusterWrap, VizWrap, MetricChart
from collections import Counter, defaultdict
import pandas as pd
import pickle
import os
import re
import spacy
import codecs
import logging

# Initialize the logger.
log = logging.getLogger('interactoveryLogger')

DEF_ENTITY_VALUE_SOURCES = 'value_sources.csv'
DEF_ENTITY_UNIQUE_VALUES = 'unique_values.txt'

DEF_CLUSTERS_FILENAME = 'hdbscan_clusters.pkl'
DEF_CLUSTERS_DIRNAME = 'clusters'

DEF_CLUSTER_DEFS_FILENAME = 'cluster_definitions.pkl'

DEF_EMBEDDINGS_FILENAME = 'embeddings.pkl'
DEF_EMBEDDINGS_DIRNAME = 'embeddings'

DEF_REDUX_EMBEDDINGS_FILENAME = 'embeddings-reduced.pkl'


def count_utterances(file_path: str, utterance_counts: Counter) -> int:
    by_utterance_count = 0
    with codecs.open(file_path, 'r', 'utf-8') as file:
        for line in file:
            line_trim = line.strip()
            if line_trim in utterance_counts:
                final_count = utterance_counts.get(line_trim)
                by_utterance_count = by_utterance_count + final_count
            else:
                by_utterance_count = by_utterance_count + 1
    return by_utterance_count


class Interactovery:
    """
    Interactovery client interface.
    """
    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 cluster: ClusterWrap,
                 spacy_nlp: spacy.language,
                 ):
        self.openai = openai
        self.cluster = cluster
        self.spacy_nlp = spacy_nlp

    def extract_entities(self, df: pd.DataFrame, output_dir: str):
        entities_by_type = defaultdict(set)
        entities_by_source = defaultdict(set)

        entity_progress = 0
        entity_progress_total = df.shape[0]

        for row in df.itertuples():
            entity_progress = entity_progress + 1
            Utils.progress_bar(entity_progress, entity_progress_total, 'Extracting known entities')

            source = getattr(row, 'source')
            participant = getattr(row, 'participant')
            utterance = getattr(row, 'utterance')

            doc = self.spacy_nlp(utterance)
            for ent in doc.ents:
                # Add the entity text to the set corresponding to its type
                entities_by_type[ent.label_].add(ent.text)
                entities_by_source[ent.text].add(source)

        # Create a directory to store the entity files
        os.makedirs(output_dir, exist_ok=True)

        # Create and write to files for each entity type
        for entity_type, examples in entities_by_type.items():
            entity_name = entity_type.lower().replace(' ', '')
            entity_dir_path = os.path.join(output_dir, entity_name)
            os.makedirs(entity_dir_path, exist_ok=True)

            values_file_path = os.path.join(entity_dir_path, DEF_ENTITY_VALUE_SOURCES)
            with codecs.open(values_file_path, 'w', 'utf-8') as f:
                f.write('source,value\n')
                for example in examples:
                    for source in entities_by_source[example]:
                        f.write(source+','+example + '\n')

            unique_file_path = os.path.join(entity_dir_path, DEF_ENTITY_UNIQUE_VALUES)
            with codecs.open(unique_file_path, 'w', 'utf-8') as f:
                for example in examples:
                    f.write(example+'\n')

    @staticmethod
    def get_intent_utterance_counts(
            *,
            directory: str,
            utterance_volumes: Counter = None,
            incl_descr: bool = True,
            incl_noise: bool = False,
            descr_sep: str = '\n',
    ) -> MetricChart:
        metric_names = []
        metric_counts = []

        title_key = 'Unique Utterances'

        for intent_dir in os.listdir(directory):
            intent_file_name = intent_dir + '.txt'
            intent_file_path = os.path.join(directory, intent_dir, intent_file_name)
            readme_file_path = os.path.join(directory, intent_dir, 'readme.txt')

            if os.path.isfile(intent_file_path):
                if not incl_noise and intent_dir == "-1_noise":
                    continue

                normed_file = re.sub("[0-9]+_(.*?)", "\\1", intent_dir)
                if incl_descr:
                    description = ''
                    if os.path.isfile(readme_file_path):
                        with codecs.open(readme_file_path, 'r', 'utf-8') as rf:
                            readme_value = rf.read()
                            description = descr_sep+'(' + readme_value + ')'
                    normed_file = normed_file + description
                metric_names.append(normed_file)

                if utterance_volumes is not None:
                    title_key = 'Utterance Volume'
                    by_volume_count = count_utterances(intent_file_path, utterance_volumes)
                    metric_counts.append(by_volume_count)
                else:
                    by_unique_count = Utils.count_file_lines(intent_file_path)
                    metric_counts.append(by_unique_count)

        metric_chart = MetricChart(
            title=title_key,
            metrics=metric_names,
            counts=metric_counts,
        )
        return metric_chart

    @staticmethod
    def get_entity_value_counts(
            *,
            directory: str,
            incl_descr: bool = True,
            descr_sep: str = '\n'
    ) -> MetricChart:
        metric_names = []
        metric_counts = []

        title_key = 'Value Volume'

        for entity_dir in os.listdir(directory):
            entity_file_name = DEF_ENTITY_VALUE_SOURCES
            entity_file_path = os.path.join(directory, entity_dir, entity_file_name)
            if os.path.isfile(entity_file_path):
                entity_label = entity_dir
                if incl_descr:
                    entity_explain = spacy.explain(entity_dir.upper())
                    entity_label = entity_label + descr_sep + '(' + entity_explain + ')'

                metric_names.append(entity_label)
                metric_counts.append(Utils.count_file_lines(entity_file_path))

        metric_chart = MetricChart(
            title=title_key,
            metrics=metric_names,
            counts=metric_counts,
        )
        return metric_chart

    @staticmethod
    def detail_entity_types(directory):
        # Reading files and counting lines
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                scrub_file = re.sub("\\.txt", "", file)
                print(f"{scrub_file} - {spacy.explain(scrub_file.upper())}")

    def cluster_and_name_utterances(self,
                                    *,
                                    workspace_dir: str,
                                    output_dir: str,
                                    session_id: str = None,
                                    utterances: list[str] = None,
                                    min_cluster_size=40,
                                    min_samples=5,
                                    epsilon=0.2,
                                    ) -> None:
        """
        Cluster the utterances into groups, use generative AI to name them, and then store the results in files.
        """
        if session_id is None:
            session_id = Utils.new_session_id()

        # Create/Load/Store the embeddings.
        embeddings_file_name = DEF_EMBEDDINGS_FILENAME
        embeddings_dir = os.path.join(workspace_dir, DEF_EMBEDDINGS_DIRNAME)
        embeddings_file_path = os.path.join(embeddings_dir, embeddings_file_name)

        if os.path.isfile(embeddings_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading embeddings from {embeddings_file_name}")
            # embeddings = np.load(embeddings_file_path)
            with open(embeddings_file_path, 'rb') as file:
                embeddings = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Generating embeddings with 'all-MiniLM-L6-v2'")
            embeddings = self.cluster.get_embeddings(utterances=utterances)
            # np.save(embeddings_file_path, embeddings)
            with open(embeddings_file_path, 'wb') as file:
                pickle.dump(embeddings, file)

        # Reduce dimensionality.
        redux_file_name = DEF_REDUX_EMBEDDINGS_FILENAME
        redux_file_path = os.path.join(embeddings_dir, redux_file_name)

        if os.path.isfile(redux_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading reduced embeddings from {redux_file_name}")
            # umap_embeddings = np.load(redux_file_path)
            with open(redux_file_path, 'rb') as file:
                umap_embeddings = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Reducing dimensionality with UMAP")
            umap_embeddings = self.cluster.reduce_dimensionality(embeddings=embeddings)
            # np.save(redux_file_path, embeddings)
            with open(redux_file_path, 'wb') as file:
                pickle.dump(umap_embeddings, file)

        # Create/Load/Store the cluster results.
        clusters_file_name = DEF_CLUSTERS_FILENAME
        clusters_dir = os.path.join(workspace_dir, DEF_CLUSTERS_DIRNAME)
        clusters_file_path = os.path.join(clusters_dir, clusters_file_name)
        if os.path.isfile(clusters_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading clusters from {clusters_file_name}")
            with open(clusters_file_path, 'rb') as file:
                cluster = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Predicting cluster labels with HDBSCAN")
            cluster = self.cluster.hdbscan(
                embeddings=umap_embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
            )
            with open(clusters_file_path, 'wb') as file:
                pickle.dump(cluster, file)

        labels = cluster.labels_

        # Get silhouette score.
        silhouette_avg = self.cluster.get_silhouette(umap_embeddings, cluster)
        log.info(f"{session_id} | cluster_and_name_utterances | Silhouette Score: {silhouette_avg:.2f}")

        clustered_sentences = self.cluster.get_clustered_sentences(utterances, cluster)

        definitions_file_name = DEF_CLUSTER_DEFS_FILENAME
        definitions_file_path = os.path.join(clusters_dir, definitions_file_name)
        if os.path.isfile(definitions_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading definitions from {definitions_file_name}")
            with open(definitions_file_path, 'rb') as file:
                new_definitions = pickle.load(file)

        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Generating definitions from LLMs")
            new_definitions = self.cluster.get_new_cluster_definitions(
                session_id=session_id,
                clustered_sentences=clustered_sentences,
                output_dir=output_dir,
            )
            with open(definitions_file_path, 'wb') as file:
                pickle.dump(new_definitions, file)

        new_labels = [d['name'] for d in new_definitions]

        log.info(f"{session_id} | cluster_and_name_utterances | Visualizing clusters")

        VizWrap.show_cluster_scatter(umap_embeddings, labels, new_labels, silhouette_avg)
