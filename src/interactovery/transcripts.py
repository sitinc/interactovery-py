# MIT License
#
# Copyright (c) 2023, Justin Randall, Smart Interactive Transformations Inc.
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

import pandas as pd
import codecs
import os
import re
import spacy
import logging
from collections import Counter

from interactovery import Utils, OpenAiWrap, CreateCompletions

log = logging.getLogger('transcriptLogger')

sys_prompt_gen_transcript = """You are helping me generate example transcripts.  Do not reply back with anything other 
than the transcript content itself.  No headers or footers.  Only generate a single transcript example for each 
response.  Separate each turn with a blank line.  Each line should start with either "USER: " or "AGENT: "."""

DEF_TS_COMBINED_FILENAME = 'transcripts_combined.csv'


class Utterances:
    """
    Utterances metadata and content.
    """

    def __init__(self,
                 *,
                 source: str,
                 utterances: list[str]):
        """
        Construct a new instance.

        :param source: The source (file name, URL, etc.) of the utterances.
        :param utterances: The utterances.
        """
        self.source = source
        self.utterances = utterances
        utterance_count = len(utterances)
        self.utterance_count = utterance_count

        utterances_set = set(utterances)
        unique_utterances = list(utterances_set)
        self.unique_utterances = unique_utterances
        unique_utterance_count = len(unique_utterances)
        self.unique_utterance_count = unique_utterance_count

        volume_utterance_count_map = Counter(utterances)
        self.volume_utterance_count_map = volume_utterance_count_map

        volume_utterance_counts_values = sum(volume_utterance_count_map.values())

        if volume_utterance_counts_values != utterance_count:
            raise Exception(f'Consistency failure: ({volume_utterance_counts_values} != {utterance_count})')

    def __str__(self):
        return (f"Utterances(source={self.source}" +
                f", utterances=..." +
                f", utterance_count={self.utterance_count}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count}" +
                f", volume_utterance_count_map=..." +
                ")")

    def __repr__(self):
        return (f"Utterances(source={self.source!r}" +
                f", utterances=..." +
                f", utterances_count={self.utterance_count!r}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count!r}" +
                f", volume_utterance_count_map=..." +
                ")")


class Transcripts:
    """
    Utility class for working with transcript generation and processing.
    """

    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 spacy_nlp: spacy.language,
                 max_transcripts: int = 2000):
        """
        Construct a new instance.

        :param openai: The OpenAiWrap client instance.
        :param spacy_nlp: The spaCy language model instance.
        :param max_transcripts: The maximum number of transcripts that can be generated by gen_agent_transcripts.
        """
        self.openai = openai
        self.spacy_nlp = spacy_nlp
        self.max_transcripts = max_transcripts

    def gen_agent_transcript(self,
                             *,
                             user_prompt: str,
                             model: str,
                             session_id: str = None
                             ):
        """
        Generate an agent transcript.
        :param model: The OpenAI chat completion model.
        :param user_prompt: The chat completion user prompt.
        :param session_id: The session ID.
        :return: the result transcript.
        """
        if session_id is None:
            session_id = Utils.new_session_id()

        cmd = CreateCompletions(
            session_id=session_id,
            model=model,
            sys_prompt=sys_prompt_gen_transcript,
            user_prompt=user_prompt,
        )
        return self.openai.execute(cmd).result

    def gen_agent_transcripts(self,
                              *,
                              user_prompt: str,
                              session_id: str = None,
                              quantity: int = 5,
                              model: str = "gpt-4-1106-preview",
                              output_dir: str = "output",
                              offset: int = 0,
                              ) -> None:
        """
        Generate a series of agent transcripts and output them to files.
        :param user_prompt: The chat completion user prompt.
        :param session_id: The session ID.
        :param quantity: The number of transcripts to generate.
        :param model: The OpenAI chat completion model.  Default is "gpt-4-1106-preview"
        :param output_dir: The transcript file output directory.
        :param offset: The transcript file number offset.
        :return: transcripts will be output to file system.
        """
        if quantity > self.max_transcripts:
            raise Exception(f"max quantity is {self.max_transcripts} unless you set max_transcripts via constructor")

        if session_id is None:
            session_id = Utils.new_session_id()

        os.makedirs(output_dir, exist_ok=True)
        file_progress = 0
        file_progress_total = quantity

        for i in range(offset, offset+quantity):
            file_progress = file_progress + 1
            Utils.progress_bar(file_progress, file_progress_total, 'Generating transcripts')

            final_file_name = f'{output_dir}/transcript{i}.txt'

            if os.path.exists(final_file_name):
                continue

            log.debug(f"{session_id} | gen_agent_transcripts | Generating example transcript #{i}")
            gen_transcript = self.gen_agent_transcript(
                session_id=session_id,
                model=model,
                user_prompt=user_prompt,
            )
            with codecs.open(final_file_name, 'w', 'utf-8') as f:
                f.write(gen_transcript)

    @staticmethod
    def concat_transcripts(dir_name: str, file_name: str) -> bool:
        """
        Concatenate transcripts
        :param dir_name: The directory with transcripts.
        :param file_name: The name of the combined transcripts file.
        :return: True if no errors occurred.
        """
        combined_name = f"{dir_name}/{file_name}"
        for file in os.listdir(dir_name):
            ts_file_name = os.path.join(dir_name, file)
            if os.path.isfile(ts_file_name):
                with codecs.open(ts_file_name, 'r', 'utf-8') as rf:
                    lines = rf.read()
                    lines = lines + '\n\n'

                with codecs.open(combined_name, 'a+', 'utf-8') as wf:
                    wf.write(lines)
        return True

    def split_sentences(self, utterance: str) -> list[str]:
        doc = self.spacy_nlp(utterance)

        utterance_sentences = [sent.text for sent in doc.sents]

        utterances_lines = []
        for sentence in utterance_sentences:
            if len(sentence.strip()) != 0:
                utterances_lines.append(sentence)

        return utterances_lines

    def process_transcript_to_csv(self, file_name: str) -> bool:
        try:
            csv_lines = ["participant,utterance"]
            invalid_lines = []
            csv_file = re.sub("\\.txt", ".csv", file_name)
            with codecs.open(file_name, 'r', 'utf-8') as f:
                lines = f.readlines()

            for line in lines:
                valid_line = re.search("^(USER|AGENT): (.*)\\.?$", line)
                if valid_line is None:
                    empty_line = re.search("\\r?\\n", line)
                    if empty_line is None:
                        invalid_lines.append(line)
                else:
                    participant = valid_line.group(1)
                    utterances = valid_line.group(2)

                    utterances = re.sub(",", "", utterances)

                    if re.search("\\.\\s*", utterances) is not None:
                        utterances_lines = self.split_sentences(utterances)

                        for final_line in utterances_lines:
                            csv_line = participant + ',' + final_line
                            csv_lines.append(csv_line)
                    else:
                        csv_line = participant + ',' + utterances
                        csv_lines.append(csv_line)

            csv_text = "\n".join(csv_lines)

            with codecs.open(csv_file, 'w', 'utf-8') as csv:
                csv.write(csv_text)

            if len(invalid_lines) > 0:
                print(invalid_lines)
                return False
            return True
        except UnicodeDecodeError as err:
            print(f"Error processing file: {file_name}: {err.reason}")
            return False

    @staticmethod
    def get_transcript_utterances(*,
                                  file_name: str,
                                  col_name: str,
                                  remove_dups: bool = True) -> list[str]:
        """Get utterances from a CSV file."""
        # Load the CSV file to a data frame.
        df = pd.read_csv(file_name)
        # Get the named column as a list.
        utterances = df[col_name].tolist()

        # Remove duplicates, default behaviour.
        if remove_dups:
            utterances_set = set(utterances)
            utterances = list(utterances_set)

        return utterances

    def process_ts_lines_to_csv(self,
                                *,
                                source: str = None,
                                lines: list[str] = None) -> list[str]:
        csv_lines = []
        invalid_lines = []

        for ts_line in lines:
            valid_line = re.search("^(USER|AGENT): (.*)\\.?$", ts_line)
            if valid_line is None:
                empty_line = re.search("\\r?\\n", ts_line)
                if empty_line is None:
                    invalid_lines.append(ts_line)
            else:
                participant = valid_line.group(1)
                utterances = valid_line.group(2)

                utterances = re.sub(",", "", utterances)

                if re.search("\\.\\s*", utterances) is not None:
                    utterances_lines = self.split_sentences(utterances)

                    for final_line in utterances_lines:
                        csv_line = source + ',' + participant + ',' + final_line
                        csv_lines.append(csv_line)
                else:
                    csv_line = source + ',' + participant + ',' + utterances
                    csv_lines.append(csv_line)

        return csv_lines

    def concat_and_process_ts_to_csv(self,
                                     *,
                                     in_dir: str = None,
                                     out_dir: str = None,
                                     out_file: str = DEF_TS_COMBINED_FILENAME) -> str | None:
        """
        Concatenate and process the transcripts
        :param in_dir: The input directory with transcripts.
        :param out_dir: The output directory for the combined CSV.
        :param out_file: The name of the combined transcripts CSV output file.
        :return: True if no errors occurred.
        """
        if in_dir is None:
            raise Exception('in_dir is required')

        if out_dir is None:
            raise Exception('out_dir is required')

        out_file_path = os.path.join(out_dir, out_file)

        if os.path.isfile(out_file_path):
            return out_file_path

        try:
            os.makedirs(out_dir, exist_ok=True)

            with codecs.open(out_file_path, 'w+', 'utf-8') as wf:
                wf.write("source,participant,utterance\n")

            ts_files = os.listdir(in_dir)

            file_progress = 0
            file_progress_total = len(ts_files)

            for ts_file in ts_files:
                file_progress = file_progress + 1
                Utils.progress_bar(file_progress, file_progress_total, 'Assembling transcripts to CSV file')
                try:
                    ts_file_path = os.path.join(in_dir, ts_file)
                    if os.path.isfile(ts_file_path):
                        with codecs.open(ts_file_path, 'r', 'utf-8') as rf:
                            lines = rf.readlines()

                            csv_lines = self.process_ts_lines_to_csv(
                                source=ts_file,
                                lines=lines,
                            )

                            csv_text = "\n".join(csv_lines)
                            csv_text = csv_text + '\n'

                            with codecs.open(out_file_path, 'a+', 'utf-8') as wf:
                                wf.write(csv_text)
                except UnicodeDecodeError as err:
                    log.error(f"Unicode decode error for file: {ts_file}: {err.reason}")
                    return None
        except Exception as err:
            log.error(f"Unhandled exception: {err}")
            return None

        return out_file_path

    @staticmethod
    def get_combined_utterances(*,
                                in_dir: str,
                                source: str | None = DEF_TS_COMBINED_FILENAME,
                                ) -> pd.DataFrame:

        path = os.path.join(in_dir, source)
        df = pd.read_csv(path)
        return df

    @staticmethod
    def get_transcript_utterances_for_party(*,
                                            party: str,
                                            file_name: str,
                                            party_col_name: str = 'participant',
                                            utter_col_name: str = 'utterance') -> Utterances:
        """Get utterances from a CSV file."""
        # Load the CSV file to a data frame.
        df = pd.read_csv(file_name)

        # Select 'utterance' values based on the participant mask
        mask = df[party_col_name] == party
        utterances_list = df.loc[mask, utter_col_name].tolist()

        utterances = Utterances(source=file_name, utterances=utterances_list)

        return utterances
