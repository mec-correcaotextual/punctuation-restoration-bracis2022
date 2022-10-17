# The repository is the implementation of the paper Ponctuation Restauration for Portuguese


Punctuation Restoration is an essential post-processing task of text generation methods, such as Speech-to-Text (STT) and Machine Translation (MT). Usually, the generation models employed in those tasks produce unpunctuated text, which is difficult for human readers and might degrade the performance of many downstream text processing tasks. Thus, many techniques exist to restore the text's punctuation. For instance, approaches based on Conditional Random Fields (CRF) and pre-trained models, such as the Bidirectional Encoder Representations from Transformers (BERT), have been widely applied. In the last few years, however, one approach has gained significant attention: casting the Punctuation Restoration problem into a sequence labeling task. In Sequence Labeling, each punctuation symbol becomes a label (e.g., COMMA, QUESTION, and PERIOD) that sequence tagging models can predict. This approach has achieved competitive results against state-of-the-art punctuation restoration algorithms. However, most research focuses on English, lacking discussion in other languages, such as Brazilian Portuguese. Therefore, this paper conducts an experimental analysis comparing the Bi-Long Short-Term Memory (BI-LSTM) + CRF model and BERT to predict punctuation in Brazilian Portuguese. We evaluate those approaches in the IWSLT2 2012-03 and OBRAS dataset in terms of precision, recall, and F1-score. The results showed that BERT achieved competitive results in terms of punctuation prediction, but it requires much more GPU resources for training than the BI-LSTM + CRF algorithm.",
KEYWORDS="Natural Language Processing; Deep Learning



@INPROCEEDINGS{226026,
    AUTHOR="Tiago De Lima and Pericles Miranda and Rafael Ferreira Mello and Moesio Wenceslau and Ig Ibert Bittencourt and Thiago Cordeiro and Jário José",
    TITLE="Sequence Labeling Algorithms for Punctuation Restoration in Brazilian Portuguese Texts",
    BOOKTITLE="BRACIS 2022 () ",
    ADDRESS="",
    DAYS="28-1",
    MONTH="nov",
    YEAR="2022",
    URL="http://XXXXX/226026.pdf"
}


