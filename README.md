# ProjectSchemasAnnotator
Project developed as part of the SCHEMAS project (financed by the Serbian ministry of Science). 
Two-Stage Machine Learning Pipeline for Schema Annotation: Boundary Detection and Multi-Label Classification
Introduction

Cognitive linguistics, over the past few decades, has foregrounded the intricate interplay between conceptual structures and linguistic expressions, demonstrating that language is not an autonomous formal system but rather one deeply rooted in embodied cognition (Evans & Green, 2006; Langacker, 2008). From the early works of Lakoff, Johnson, and their contemporaries, it has become a foundational principle that meaning is grounded in sensory-motor experiences and shaped by patterns of bodily interaction with the environment (Johnson, 1987; Lakoff, 1987). Within this theoretical context, image schemas have emerged as a particularly salient concept: they are recurrent patterns of perception and movement that inform and structure more complex conceptualizations. Image schemas are understood as basic, preverbal gestalts that arise from embodied experience and recur across multiple conceptual domains, providing a scaffold upon which we build more elaborate metaphorical and abstract reasoning. While their role in conceptual meaning and metaphorical extension is well recognized, less attention has been paid to how image schemas are syntactically realized and how their underlying conceptual patterns might shape the structural patterns we observe in language. This gap not only poses theoretical questions but also presents a practical challenge for computational approaches to language—how might we design models that automatically detect these image-schematic patterns in naturally occurring texts? Addressing such a question opens up a fruitful area of inquiry: if image schemas are truly foundational to our conceptual apparatus, then examining their formal and syntactic realizations can yield deeper insights into how language is constructed, processed, and understood, including insights gleaned through machine learning (ML) methods.
At the core of much cognitive linguistic research lies the concept of the image schema: a dynamic, embodied pattern of experience that emerges through recurring sensorimotor interactions with our environment (Johnson, 1987; Lakoff, 1987; Mandler, 2004). Image schemas are not static mental pictures; rather, they are continuous and interactive patterns of experience, such as moving along a path, exerting force, or being contained within a bounded space. According to Johnson (1987), these schemas arise early in cognitive development as infants learn to balance their bodies, manipulate objects, perceive objects entering or leaving containers, and navigate through space. Such patterns become internalized as mental templates that can be metaphorically extended to structure more abstract conceptual domains. For example, the SOURCE-PATH-GOAL schema—one of the most frequently cited examples—originates in our repeated experience of physically moving from one location to another and then shapes our understanding of events, life journeys, and abstract conceptual progressions (Lakoff, 1987; Mandler, 2004).
Because image schemas are grounded in bodily experience, they are generally seen as universal or near-universal cognitive structures. They emerge from bodily interactions common to all humans, such as UP-DOWN arising from our vertical orientation or CONTAINER from our awareness of boundaries and enclosures (Johnson, 1987; Evans & Green, 2006). However, the particular linguistic and cultural elaborations of these schemas may vary, allowing some universality in conceptual grounding to coexist with linguistic diversity. Research in cognitive linguistics has focused heavily on how image schemas shape metaphorical mappings, conceptual blends, and lexical semantics, demonstrating their pervasive influence across numerous languages and conceptual domains.
Despite this rich body of work, the attention to image schemas has primarily concentrated on their conceptual and semantic dimensions. Cognitive linguists have extensively explored how image schemas inform meaning, conceptual metaphors, and reasoning (Lakoff, 1987; Johnson, 1987), but the question of how these conceptual patterns manifest syntactically—how they are encoded, hinted at, or reinforced by the grammatical structures of language—has received comparatively less systematic scrutiny. Although some early studies have hinted at possible correlations between syntactic constructions and underlying image-schematic structures (Clausner & Croft, 1999), the field as a whole has yet to fully engage with the challenge of mapping image schemas onto syntactic patterns. Moreover, a machine learning perspective on this problem—developing algorithms to detect and label potential image-schematic structures in text—remains largely unexplored.
The question at the heart of this study is how image schemas, as conceptual building blocks, are manifested in the syntactic fabric of language. If image schemas truly underpin a wide range of conceptual and linguistic phenomena, it follows that their influence might be detectable not only at the semantic or conceptual level but also in the structural arrangement of clauses, phrases, and sentences. For instance, we might ask whether certain image schemas correlate strongly with particular construction types, prepositional phrases, verb classes, or argument structures. Another avenue of inquiry might involve examining how force dynamics—an essential component of many image schemas—map onto grammatical patterns that encode agency, causation, and affectedness. Crucially, such mapping raises the possibility of training computational models to automate the detection of these schematic cues, offering a scalable approach to analyzing large corpora.
To approach this problem, one significant hurdle must be addressed: the relative lack of empirical research employing large-scale, corpus-driven methods to investigate the syntactic realizations of image schemas. Much of the seminal work in cognitive linguistics and image schema research has relied on introspective data, small sets of examples, or carefully constructed experimental stimuli (Johnson, 1987; Lakoff & Johnson, 1999). While these methods have yielded valuable theoretical insights, they do not easily lend themselves to demonstrating systematic correlations between underlying image schemas and particular syntactic constructions in naturally occurring language. There is a need for comprehensive, empirical corpus analyses that can test hypotheses about the systematic patterns through which image schemas might emerge in syntax and structure our linguistic expressions. Furthermore, integrating corpus-based annotations with a machine learning system offers a powerful synergy: the annotated corpus can train the model to recognize subtle syntactic cues associated with specific image schemas, thereby enabling deeper corpus analysis at scale.
Despite advances in corpus linguistics and the availability of extensive corpora, cognitive linguistic research has only recently begun to embrace large-scale empirical approaches more fully. While some recent studies have investigated metaphorical patterns, collocations, and semantic frames through corpus methods, image schema research remains somewhat behind this curve. Few studies have explicitly linked certain syntactic constructions to underlying image schemas based on corpus evidence, leaving this a relatively under-explored frontier in cognitive linguistics. As a result, we lack robust quantitative data that could either confirm or challenge existing theoretical claims about how deeply image schemas permeate language structure. This is precisely where computational tools—such as machine learning models for boundary detection, multi-label classification, or chunk-level annotation—can bridge the gap, offering systematic, large-scale evidence.
This lack of corpus-level research represents both a gap in the literature and a new opportunity. Establishing clear, reproducible correlations between image schemas and syntactic patterns would not only provide empirical support for theoretical claims but also facilitate cross-linguistic comparisons. Are certain image schemas universally associated with particular syntactic structures, or do languages vary widely in how they encode these conceptual patterns? Answering these questions would broaden our understanding of the language-cognition interface, shedding new light on the interplay between conceptual patterns and grammatical form. In tandem, machine learning approaches could be extended cross-linguistically, potentially detecting universal vs. language-specific features in how image schemas surface in syntax.
To address these concerns, this paper will draw on the SCHEMAS team project corpus, a specially curated dataset that has been assembled to facilitate the study of image schemas and their linguistic realizations. The SCHEMAS corpus brings together a range of carefully selected texts that have been annotated by trained researchers to reflect known or hypothesized image-schematic patterns. This corpus provides a unique resource for the present study, allowing for the systematic examination of syntactic patterns associated with particular schemas across a large and varied dataset. By leveraging this corpus, we seek to identify, quantify, and analyze the syntactic manifestations of image schemas in authentic language use. Moreover, an integral part of this research is the development of a machine learning model that will be trained on these annotated data, aiming to automate the detection of image-schematic cues and thereby scale the analysis to larger corpora or different text genres.
In the following sections, this paper will begin by surveying the theoretical landscape, situating image schemas within cognitive linguistics and reviewing relevant literature on their conceptual importance. It will then present a methodological framework for operationalizing image schemas in syntactic annotation and analysis. Drawing on the SCHEMAS corpus, the study will investigate the extent to which certain image schemas correlate with identifiable syntactic structures, describing the patterns that emerge and discussing their theoretical implications. Finally, it will consider the broader significance of these findings for our understanding of the relationship between embodied cognition and linguistic form, offering suggestions for future research that might build upon these insights.
By approaching image schemas from a corpus-based, syntactically oriented perspective—and integrating machine learning to detect and classify the schematic cues in text—this study aims to enrich the field’s understanding of how embodied conceptual patterns shape linguistic structure. It will illustrate that the influence of image schemas on language is not limited to semantics or metaphor; rather, these fundamental embodied concepts can also guide and constrain the way we construct our utterances. Such insights will underscore the importance of integrating theoretical claims with empirical corpus data while demonstrating the feasibility of a computational approach that leverages ML for large-scale image-schema detection. Ultimately, this convergence of corpus methods, linguistic theory, and machine learning will contribute to a more nuanced and comprehensive account of language as an embodied, cognitive, and structured phenomenon.

Methodology

Consider the task of an annotator. Upon encountering a text, the hypothetical annotator, following the annotation guidelines of the SCHEMAS project group as outlined in Antović and colleagues (2023) and Figar & Veličković (2023), might read a text and, remembering the examples offered in guidelines, annotate an example sentence:

[...]“This legislation is a giant step forward<ms><F><P++><spec><forward> in our fight to combat<ms><F+><L> the fentanyl crisis, crack down on the dealers peddling<ms><F><P+><L> death in our communities, and accelerate<ms><F+><P+><spec><forward><L> our state’s public health response to get this deadly drug off our streets<s><F+><P><L--> and save lives,” House Speaker Alec Garnett, a Democrat, said after the bill’s passage<ms><F><P+><spec><end path>.”[...] (taken from the SCHEMAS linguistic corpus).

Sequences starting from <ms> or <s> and terminating in the last symbol (“>”) are examples of schematic complexes, in this denoting whether a sequence of text has been identified as containing schemas - <Force>, <Path>, <Link>, <Balance> and <Containment>, scalar modification, the SCALE schema operating on the five basic schemas, in the annotation cluster realized as either a plus or a minus sign, denoting whether the valence is positive or negative as well as intensity (the higher the number the greater the positive or negative valence, respectively). 
The presence of several clusters within the sentence suggests that the annotator associated particular chunks of the overall sentence with specific schematic clusters. Thus, “This legislation is a giant step forward” is associated with “<ms><F><P++><spec><forward>” and “in our fight to combat” is associated with “<ms><F+><L>”. Once annotated in this manner, further analysis can be performed to obtain raw and relative counts of schemas, either as clusters or as individual schemas. Note, however, that a single sentence can contain several clusters with no cluster in the corpus transcending sentential boundaries. A probable model of the annotation procedure can thus be construed in the following way:

The annotator familiarizes her/himself with sequences serving as examples of particular schemas/schematic complexes.
The annotator keeps the memorized examples active in memory as incoming text is read and parsed.
As the text is being read, the processing of particular lexical items and syntactic structures increases the probability of schemas being detected and annotated.
Once the build-up of probability reaches a certain threshold, schemas are recognized as annotated. The build-up occurs simultaneously for several schemas.
The moment of annotation terminates the previous build-up of probability, meaning that after a particular sentential chunk is annotated the build-up procedure starts again.
As there are no trans-sentential schemas, only the local sequences present within a sentence are relevant for the build-up.

Consider now how a computational variant of a schema annotator might approximate human annotation behavior. At the most fundamental level, the system must parse any incoming text and segment it into sentences. Each sentence must then be further subdivided into individual tokens, thereby enabling an attention-like mechanism to monitor token-to-token interactions and begin incrementally assigning probabilities of encountering particular schemas. Once these probabilities exceed a certain threshold, the computational annotator would finalize a “chunk” and assign the relevant schema labels to that segment, mirroring the way a human annotator would stop reading, annotate the recognized schemas, and then continue.
Because chunks can exhibit substantial syntactic variation—including both fully formed syntactic structures and extraneous fragments (“the fentanyl crisis, crack down on the dealers peddling<ms><F><P+><L>”)—the computational annotator must also learn to selectively weight or ignore portions of each chunk. In other words, the system cannot simply rely on a single, deterministic algorithm; rather, it must adapt to diverse linguistic configurations by dynamically adjusting the probability of encountering particular schemas. Such adaptability is precisely why a machine learning approach becomes indispensable, as it enables the model to generalize from annotated examples and to manage the inherent variability of naturally occurring language data. With that said, the first step in the process of building our predictive model (and the source of the data to be used in “ On the Lexical and Syntactic Manifestation of Image Schemas in English - Insights from the SCHEMAS Team Annotated Corpus”)  consists of formalizing a method of reliably capturing the textual segment found to the left of a schema. From the next segment on, the text will concurrently discuss the logic, the data and the implementation of the overall algorithm. 
The code was implemented in Python. Jupyter notebook and data is provided here: https://github.com/MladenPopovicFilFak/ProjectSchemasAnnotator. 

	3. Textual Segmentation Tool - From Corpus to Training Datasets

The first step to building our synthetic annotator by necessity involves the pre-processing of annotated text so that the resulting database preserves meaningful structures (chunks and associated annotations) while making them machine-readable. At its most basic level an algorithm parsing the human-annotated corpus should 1) preserve the textual segment found to the left of an annotation and 2) preserve the annotation. 
This directly feeds into the architecture of the annotation model - given the requirements of the model to both segment the text and annotate it, the decision was made to handle the process in two steps. First, a sub-model based on BERT would be trained on the first segment of the data (1) in order to recognize schema-bearing chunks and to extract it from any random text serving as input. The second BERT-based sub-model, trained on textual segments from (1) and the associated schema complexes (2), would then take the output from the first sub-model and predict the presence of schemas. 

Cell 1: Necessary libraries. Cell 1 handles the importing of necessary libraries. Chardet is used for encoding detection, nltk for linguistic parsing, torch and sklearn for setting up the learning environment and the transformer library is used to handle BERT-related operations. Boundary labels are used to annotate the schema-carrying segments so that the first word of a segment is annotated ‘B-Chunk’ signifying the onset of said segment, ‘I-Chunks’ labeling the words found between the onset of a segment and the schema cluster. The final ‘O’ label is applied to segments that are not schema-carrying.

#######################################################################CELL 1: SETUP, INSTALLS, AND IMPORTS #######################################################################

from google.colab import drive
drive.mount('/content/drive')


!pip install chardet transformers sentencepiece nltk


import os
import re
import chardet
import nltk
import torch
import numpy as np
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# Download NLTK sentence tokenizer data
nltk.download('punkt')


# Global configs
MODEL_NAME = "bert-base-uncased"   # Or another HF model
MAX_LEN = 128


# For Stage 1 boundary detection
BOUNDARY_LABELS = ["O", "B-CHUNK", "I-CHUNK"]
label2id_boundary = {lab: i for i, lab in enumerate(BOUNDARY_LABELS)}
id2label_boundary = {i: lab for i, lab in enumerate(BOUNDARY_LABELS)}


# For Stage 2 multi-label classification
SCHEMA_LABELS = ["P", "F", "L", "B", "C"]
SCHEMA2ID = {s: i for i, s in enumerate(SCHEMA_LABELS)}

Cell 2 is responsible for parsing raw text data to identify and extract meaningful chunks based on inline tags. This parsing logic is pivotal for preparing data for both Stage 1 (Boundary Detection) and Stage 2 (Schema Classification) of the pipeline. Specifically, it:
Splits Text into Chunks: Uses regular expressions to identify and separate text segments marked by inline tags (e.g., <P>, <F+>).
Processes Tags: Collapses complex tags into core schema labels, ensuring consistency and relevance.
Prepares Data for Training: Structures the data into formats suitable for both boundary detection and schema classification tasks.

#######################################################################
CELL 2: PARSING LOGIC FOR INLINE TAGS -> CHUNKS
#######################################################################


import re
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize


# Define SCHEMA_LABELS and SCHEMA2ID (Ensure consistency)
SCHEMA_LABELS = ["P", "F", "L", "B", "C"]
SCHEMA2ID = {s: i for i, s in enumerate(SCHEMA_LABELS)}
SCHEMA_ID2LABEL = {i: s for i, s in enumerate(SCHEMA_LABELS)}


def parse_clusters_in_sentence(sentence):
    """
    Splits a sentence on consecutive <...> tags.
    Associates each tag group with the preceding chunk.
    Returns a list of dicts:
      [
        {
          "chunk_text": "...",
          "raw_tags": [...],
        },
        ...
      ]
    """
    # Pattern to identify consecutive tags as one group
    pattern = r'((?:<[^>]+>)+)'
    parts = re.split(pattern, sentence)


    chunks = []


    # Iterate over parts in pairs: text + tags
    for i in range(0, len(parts), 2):
        text_chunk = parts[i].strip()
        tags = []
        if i + 1 < len(parts):
            tags = re.findall(r'<([^>]+)>', parts[i + 1])
        if text_chunk:
            chunks.append({
                'chunk_text': text_chunk,
                'raw_tags': tags.copy()
            })


    return chunks


def collapse_raw_tags(raw_tags):
    """
    Convert tags like ["ms", "F+", "P++"] to a set of [P, F, L, B, C].
    Discard irrelevant tags (ms, spec, forward, etc.).
    """
    core_set = set()
    for rt in raw_tags:
        if not rt:
            continue
        base = rt[0].upper()  # Ensure case insensitivity
        if base in {"P", "F", "L", "B", "C"}:
            core_set.add(base)
    return core_set


def parse_text_into_stage_data(raw_text, tokenizer):
    """
    1) Sentence-splits the text.
    2) For each sentence, parse chunk boundaries (Stage 2) + create B/I/O (Stage 1).
    Returns:
      stage1_data: [{"tokens": [...], "labels": ["B-CHUNK","I-CHUNK",...]}]
      stage2_data: [{"chunk": "...", "labels": [0/1,...]}]
    """
    sentences = sent_tokenize(raw_text)
    stage1_data = []
    stage2_data = []


    for sent_idx, sent in enumerate(sentences, 1):
        # Identify chunk boundaries
        sent_chunks = parse_clusters_in_sentence(sent)
        print(f"\nProcessing Sentence {sent_idx}: {sent}")
        print(f"Identified Chunks: {sent_chunks}")


        # Reconstruct clean_sentence without tags
        clean_sentence = re.sub(r'<[^>]+>', '', sent).strip()
        print(f"Clean Sentence: {clean_sentence}")


        # Tokenize the clean_sentence with offset mapping
        encoding = tokenizer(clean_sentence, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offset_mappings = encoding['offset_mapping']
        print(f"Tokens: {tokens}")
        print(f"Offset Mappings: {offset_mappings}")


        # Initialize labels as "O"
        label_sequence = ["O"] * len(tokens)


        # Track the last assigned character to prevent overlapping assignments
        last_assigned_char = 0


        for ch_idx, ch in enumerate(sent_chunks, 1):
            chunk_text = ch["chunk_text"]
            raw_tag_set = collapse_raw_tags(ch["raw_tags"])


            if not raw_tag_set:
                # This chunk has no labels, so tokens remain "O"
                print(f"Chunk {ch_idx}: '{chunk_text}' - No Labels Assigned")
                continue  # Do not assign labels to these tokens


            # Find the chunk_text in clean_sentence starting from last_assigned_char
            start_char = clean_sentence.find(chunk_text, last_assigned_char)
            if start_char == -1:
                print(f"Warning: Chunk '{chunk_text}' not found in clean_sentence.")
                continue


            end_char = start_char + len(chunk_text)
            print(f"Chunk {ch_idx}: '{chunk_text}' - Start: {start_char}, End: {end_char}")


            # Assign labels to tokens within [start_char, end_char)
            first_token = True
            for i, (token_start, token_end) in enumerate(offset_mappings):
                if token_start >= start_char and token_end <= end_char:
                    if label_sequence[i] == "O":  # Only assign if not already labeled
                        if first_token:
                            label_sequence[i] = "B-CHUNK"
                            first_token = False
                        else:
                            label_sequence[i] = "I-CHUNK"


            # Assign Stage 2 labels
            label_vec = [0] * len(SCHEMA_LABELS)
            for t in raw_tag_set:
                if t in SCHEMA2ID:
                    idx = SCHEMA2ID[t]
                    label_vec[idx] = 1
            print(f"Raw Tags: {ch['raw_tags']}")
            print(f"Assigned Labels: {label_vec}")


            stage2_data.append({
                "chunk": chunk_text,
                "labels": label_vec
            })


            # Update last_assigned_char to end_char to prevent overlapping
            last_assigned_char = end_char


        # Append to Stage1 data
        stage1_data.append({
            "tokens": tokens,
            "labels": label_sequence
        })


    return stage1_data, stage2_data

SCHEMA2ID and SCHEMA_ID2LABEL: Dictionaries that map schema labels to unique integer IDs and vice versa. These mappings are crucial for model training and prediction, ensuring that labels are consistently represented numerically.

# Define SCHEMA_LABELS and SCHEMA2ID (Ensure consistency)
SCHEMA_LABELS = ["P", "F", "L", "B", "C"]
SCHEMA2ID = {s: i for i, s in enumerate(SCHEMA_LABELS)}
SCHEMA_ID2LABEL = {i: s for i, s in enumerate(SCHEMA_LABELS)}

The function parse_clusters_in_sentence(sentence)uses regular expressions to detect and group consecutive inline tags within a sentence.

def parse_clusters_in_sentence(sentence):
    """
    Splits a sentence on consecutive <...> tags.
    Associates each tag group with the preceding chunk.
    Returns a list of dicts:
      [
        {
          "chunk_text": "...",
          "raw_tags": [...],
        },
        ...
      ]
    """
    # Pattern to identify consecutive tags as one group
    pattern = r'((?:<[^>]+>)+)'
    parts = re.split(pattern, sentence)


    chunks = []


    # Iterate over parts in pairs: text + tags
    for i in range(0, len(parts), 2):
        text_chunk = parts[i].strip()
        tags = []
        if i + 1 < len(parts):
            tags = re.findall(r'<([^>]+)>', parts[i + 1])
        if text_chunk:
            chunks.append({
                'chunk_text': text_chunk,
                'raw_tags': tags.copy()
            })


    return chunks

The purpose of this function is to use regular expressions to detect and group consecutive inline tags within a sentence and to associate each group of tags with the preceding text segment, effectively mapping text chunks to their respective tags. The regular expression pattern r'((?:<[^>]+>)+)' works by 1) defining the non-capturing group to the left of the first annotation (?:), 2) matching any sequence of characters enclosed within < and >, representing a single tag <[^>]+>, 3) matching one or more consecutive tags (?:<[^>]+>)+ and, finally 4) capturing the whole sequence of consecutive tags  ((?:<[^>]+>)+).
Then, re.split divides sentences into alternating segments of text and tag groups so that "This is a sample <P>text<F+>." gets segmented as parts: ['This is a sample ', '<P><F+>', '.'] 
The function then iterates over parts, generating a dictionary consisting of the textual chunks and raw tags. Example: [ { "chunk_text": "This is a sample", "raw_tags": ["P", "F+"] } ]
The collapse_raw_tags(raw_tags)is used to simplify the schemas present in corpus. Seeing that the quantity of possible schemas and their combinations is high, we decided to first test-run our pipeline using a reduced set of schemas. Thus, all specifiers such as <FORWARD>, <UP>, <DOWN> and others got removed, while scaled schemas were reduced to their base variant: <F+>, <F++>, <F+++> and other scaled versions got collapsed into a single <F> category. 

def collapse_raw_tags(raw_tags):
    """
    Convert tags like ["ms", "F+", "P++"] to a set of [P, F, L, B, C].
    Discard irrelevant tags (ms, spec, forward, etc.).
    """
    core_set = set()
    for rt in raw_tags:
        if not rt:
            continue
        base = rt[0].upper()  # Ensure case insensitivity
        if base in {"P", "F", "L", "B", "C"}:
            core_set.add(base)
    return core_set

The parse_text_into_stage_data(raw_text, tokenizer)function processes the raw texts that serve as inputs and builds the training datasets for both models. The output of the function is are two lists of dictionaries with stage1_data listing tokens along with corresponding boundary labels and stage2_data containing extracted textual chunks along with the corresponding schemas, rendered as a multi-hot vector indicating the presence or absence of a schema. To illustrate, the output of the function has the following shape:

stage1_data = [
    {
        "tokens": ["Brent", "crude", "'", "s", "rise", "above", "that", "milestone", "."],
        "labels": ["B-CHUNK", "I-CHUNK", "O", "O", "O", "O", "O", "O", "O"]
    },
    ...
]

stage2_data = [
    {
        "chunk": "Brent crude's rise above that milestone",
        "labels": [1, 1, 0, 0, 0]  # Example: P and F schemas present
    },
    ...
]

def parse_text_into_stage_data(raw_text, tokenizer):
    """
    1) Sentence-splits the text.
    2) For each sentence, parse chunk boundaries (Stage 2) + create B/I/O (Stage 1).
    Returns:
      stage1_data: [{"tokens": [...], "labels": ["B-CHUNK","I-CHUNK",...]}]
      stage2_data: [{"chunk": "...", "labels": [0/1,...]}]
    """
    sentences = sent_tokenize(raw_text)
    stage1_data = []
    stage2_data = []


    for sent_idx, sent in enumerate(sentences, 1):
        # Identify chunk boundaries
        sent_chunks = parse_clusters_in_sentence(sent)
        print(f"\nProcessing Sentence {sent_idx}: {sent}")
        print(f"Identified Chunks: {sent_chunks}")


        # Reconstruct clean_sentence without tags
        clean_sentence = re.sub(r'<[^>]+>', '', sent).strip()
        print(f"Clean Sentence: {clean_sentence}")


        # Tokenize the clean_sentence with offset mapping
        encoding = tokenizer(clean_sentence, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offset_mappings = encoding['offset_mapping']
        print(f"Tokens: {tokens}")
        print(f"Offset Mappings: {offset_mappings}")


        # Initialize labels as "O"
        label_sequence = ["O"] * len(tokens)


        # Track the last assigned character to prevent overlapping assignments
        last_assigned_char = 0


        for ch_idx, ch in enumerate(sent_chunks, 1):
            chunk_text = ch["chunk_text"]
            raw_tag_set = collapse_raw_tags(ch["raw_tags"])


            if not raw_tag_set:
                # This chunk has no labels, so tokens remain "O"
                print(f"Chunk {ch_idx}: '{chunk_text}' - No Labels Assigned")
                continue  # Do not assign labels to these tokens


            # Find the chunk_text in clean_sentence starting from last_assigned_char
            start_char = clean_sentence.find(chunk_text, last_assigned_char)
            if start_char == -1:
                print(f"Warning: Chunk '{chunk_text}' not found in clean_sentence.")
                continue


            end_char = start_char + len(chunk_text)
            print(f"Chunk {ch_idx}: '{chunk_text}' - Start: {start_char}, End: {end_char}")


            # Assign labels to tokens within [start_char, end_char)
            first_token = True
            for i, (token_start, token_end) in enumerate(offset_mappings):
                if token_start >= start_char and token_end <= end_char:
                    if label_sequence[i] == "O":  # Only assign if not already labeled
                        if first_token:
                            label_sequence[i] = "B-CHUNK"
                            first_token = False
                        else:
                            label_sequence[i] = "I-CHUNK"


            # Assign Stage 2 labels
            label_vec = [0] * len(SCHEMA_LABELS)
            for t in raw_tag_set:
                if t in SCHEMA2ID:
                    idx = SCHEMA2ID[t]
                    label_vec[idx] = 1
            print(f"Raw Tags: {ch['raw_tags']}")
            print(f"Assigned Labels: {label_vec}")


            stage2_data.append({
                "chunk": chunk_text,
                "labels": label_vec
            })


            # Update last_assigned_char to end_char to prevent overlapping
            last_assigned_char = end_char


        # Append to Stage1 data
        stage1_data.append({
            "tokens": tokens,
            "labels": label_sequence
        })


    return stage1_data, stage2_data

To illustrate, the input of the sentence: “Today another American president faces rising<ms><P><F><Spec><UP> fuel prices, spurred<ms><F+><P++><L+> by a challenge mostly out of his control, an invasion<s><F++><P++><L++><C+> of Ukraine by Russia, a top oil and gas producer intent to use its energy supplies as a weapon when necessary.”, would produce the following output: 

Stage 1 Data: Sentence 1: today: B-CHUNK another: I-CHUNK american: I-CHUNK president: I-CHUNK faces: I-CHUNK rising: I-CHUNK fuel: B-CHUNK prices: I-CHUNK ,: I-CHUNK spurred: I-CHUNK by: B-CHUNK a: I-CHUNK challenge: I-CHUNK mostly: I-CHUNK out: I-CHUNK of: I-CHUNK his: I-CHUNK control: I-CHUNK ,: I-CHUNK an: I-CHUNK invasion: I-CHUNK of: O ukraine: O by: O russia: O ,: O a: O top: O oil: O and: O gas: O producer: O intent: O to: O use: O its: O energy: O supplies: O as: O a: O weapon: O when: O necessary: O .: O

and:

Stage 2 Data:
Chunk 1: Today another American president faces rising
Assigned Labels: ['P', 'F']

Chunk 2: fuel prices, spurred
Assigned Labels: ['P', 'F', 'L']

Chunk 3: by a challenge mostly out of his control, an invasion
Assigned Labels: ['P', 'F', 'L', 'C']

Stage 1 data is used for the training and validation of the segmentation model while Stage 2 data is used for the training and validation of the annotator. The models are deployed in tandem so that any input text is first segmented by the segmentation model and then passed onto the annotator model of schema assignment. 
Cell 3 is responsible for transforming unstructured text data into a structured format, laying the groundwork for effective model training in subsequent cells. The code contained in the cell applies the previously defined function and iterates over all files as our corpus 
Reading Raw Text Files: Traverses specified directories to locate and read all .txt files containing the raw data.
Encoding Detection and Handling: Utilizes the chardet library to accurately detect the encoding of each text file, ensuring correct reading of diverse datasets.
Parsing Text into Structured Data:
Stage 1 (Boundary Detection): Prepares data with token-level BIO (Begin, Inside, Outside) labels to identify chunk boundaries.
Stage 2 (Schema Classification): Structures data with multi-hot labels corresponding to predefined schema categories for each identified chunk.

####################################################################CELL 3: READ ALL TXT FILES, DETECT ENCODING, PARSE -> STAGE1 & STAGE2 #######################################################################
import os
import chardet
from transformers import AutoTokenizer
# Define your tokenizer (ensure it matches the one used in parsing logic)
MODEL_NAME = "bert-base-uncased"  # Replace with your specific model if different
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


FOLDER_PATH = "/content/drive/MyDrive/02 ENGLISH CORPUS"  # Change depending on the location of your input files


all_stage1 = []
all_stage2 = []


txt_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')]
print(f"Found {len(txt_files)} text files.")


for filename in txt_files:
    full_path = os.path.join(FOLDER_PATH, filename)


    # 1) Detect encoding
    with open(full_path, 'rb') as f:
        raw_data = f.read(2048)
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
        if not encoding:
            encoding = 'utf-8'  # Fallback encoding
            print(f"Encoding not detected for {filename}. Using fallback encoding 'utf-8'.")


    # 2) Read file with detected encoding
    try:
        with open(full_path, 'r', encoding=encoding, errors='replace') as f:
            file_text = f.read()
    except Exception as e:
        print(f"Error reading {filename} with encoding {encoding}: {e}")
        continue  # Skip to the next file in case of an error


    # 3) Parse text -> stage1, stage2
    stage1_data, stage2_data = parse_text_into_stage_data(file_text, tokenizer)
    all_stage1.extend(stage1_data)
    all_stage2.extend(stage2_data)


    print(f"Processed file: {filename}")


print(f"\nTotal Stage1 examples: {len(all_stage1)}")
print(f"Total Stage2 examples: {len(all_stage2)}")
Cell 3 completes the data preprocessing step of the overall procedure. The pipeline contained in the three cells presented above can 








References:
Clausner, T. C., & Croft, W. (1999). Domains and image schemas. Cognitive Linguistics, 10(1), 1–31. https://doi.org/10.1515/cogl.1999.001
Evans, V., & Green, M. (2006). Cognitive linguistics: An introduction. Edinburgh University Press.
Johnson, M. (1987). The body in the mind: The bodily basis of meaning, imagination, and reason. University of Chicago Press.
Lakoff, G. (1987). Women, fire, and dangerous things: What categories reveal about the mind. University of Chicago Press.
Lakoff, G., & Johnson, M. (1999). Philosophy in the flesh: The embodied mind and its challenge to Western thought. Basic Books.
Langacker, R. W. (2008). Cognitive grammar: A basic introduction. Oxford University Press.
Mandler, J. M. (2004). The foundations of mind: Origins of conceptual thought. Oxford University Press.
Antović, M., Jovanović, V. Ž., & Figar, V. (2023). Dynamic schematic complexes: Image schema interaction in music and language cognition reveals a potential for computational affect detection. Pragmatics & Cognition, 30(2), 258-295.






