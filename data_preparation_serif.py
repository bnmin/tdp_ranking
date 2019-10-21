from HAT.backend.utils.serifxml import Document

from data_preparation import choose_candidates_padded, choose_candidates
from data_structures import get_root_node, get_padding_node, Node, DCT_word


def create_snt_node_lists(doc_path):
    snt_list = []
    node_list = []
    doc_token_starts = []

    # Add DCT manually since not a sentence in SERIF XML (unlike in training data)
    snt_list.append(['DCT'])
    node_list.append(Node(snt_index_in_doc=0, start_word_index_in_snt=0, end_word_index_in_snt=0, node_index_in_doc=0,
                          start_word_index_in_doc=0, end_word_index_in_doc=0, words=DCT_word, label='TIMEX'))
    doc_token_starts.append(0)

    doc_theory = Document(doc_path)
    for sentence_idx, sentence in enumerate(doc_theory.sentences, start=1):  # DCT sentence is 0
        sentence_theory = sentence.sentence_theories[0]  # Pick first/best sentence theory (normally there is only one)
        if len(sentence_theory.token_sequence) < 1:
            continue

        # Track indices of tokens in the sentence and "document", where it doesn't actually matter what token index
        # it is in the real document as long as it's the right token index in the sentence list we create (which
        # the model presumes to be the document)
        sentence_token_starts = [token.start_char for token in sentence_theory.token_sequence]
        doc_token_starts.extend(sentence_token_starts)

        # Add sentence
        tokens = [i.text for i in sentence_theory.token_sequence]
        snt_list.append(tokens)

        # Add nodes
        for value_mention in sentence_theory.value_mention_set:
            if value_mention.value_type == 'TIMEX2.TIME':
                first_token_start_char = value_mention.tokens[0].start_char
                last_token_start_char = value_mention.tokens[-1].start_char

                start_word_index_in_snt = sentence_token_starts.index(first_token_start_char)
                end_word_index_in_snt = sentence_token_starts.index(last_token_start_char)
                start_word_index_in_doc = doc_token_starts.index(first_token_start_char)
                end_word_index_in_doc = doc_token_starts.index(last_token_start_char)
                words = ' '.join(i.text for i in value_mention.tokens)

                # Set node_index_in_doc later
                node_list.append(Node(sentence_idx, start_word_index_in_snt, end_word_index_in_snt, -1,
                                      start_word_index_in_doc, end_word_index_in_doc, words, 'TIMEX'))

        for event_mention in sentence_theory.event_mention_set:
            first_token_start_char = event_mention.anchor_node.start_token.start_char
            last_token_start_char = event_mention.anchor_node.end_token.start_char

            start_word_index_in_snt = sentence_token_starts.index(first_token_start_char)
            end_word_index_in_snt = sentence_token_starts.index(last_token_start_char)
            start_word_index_in_doc = doc_token_starts.index(first_token_start_char)
            end_word_index_in_doc = doc_token_starts.index(last_token_start_char)
            words = ' '.join(i.text for i in event_mention.anchor_node.tokens)

            # Set node_index_in_doc later
            node_list.append(Node(sentence_idx, start_word_index_in_snt, end_word_index_in_snt, -1,
                                  start_word_index_in_doc, end_word_index_in_doc, words, 'EVENT'))

    # Sort nodes by occurrence in document
    node_list.sort(key=lambda node: (node.snt_index_in_doc, node.start_word_index_in_snt))
    for i, node in enumerate(node_list):
        node.node_index_in_doc = i

    return snt_list, node_list


def make_one_doc_test_data(doc_path, pad_candidates):
    snt_list, node_list = create_snt_node_lists(doc_path)

    # create test instance list
    test_instance_list = []
    root_node = get_root_node()
    padding_node = get_padding_node()

    for c_node in node_list:
        if pad_candidates:
            instance = choose_candidates_padded(node_list, root_node, padding_node, c_node, None, None)
        else:
            instance = choose_candidates(node_list, root_node, c_node, None, None)

        test_instance_list.append(instance)

    return [snt_list, test_instance_list]


def make_test_data(test_file, pad_candidates=False):
    # Model expects a list of documents
    test_data = [make_one_doc_test_data(test_file, pad_candidates)]
    return test_data
