import numpy as np


class Document(object):
    def __init__(self):
        self.docs = []
        self.i2w = []
        self.w2i = {}
        self.doc_lengths = None
        self.V = 0
        self.D = 0

    def fit(self, path: str):
        doc_length_list = []
        with open(path) as f:
            for line in f:
                doc = []
                for w in line.split():
                    w_id = self.w2i.get(w, len(self.w2i))
                    doc.append(w_id)
                    if w_id == len(self.w2i):
                        self.w2i[w] = w_id
                        self.i2w.append(w)
                self.docs.append(np.array(doc, dtype=np.uint32))
                doc_length_list.append(len(doc))
            self.doc_lengths = np.array(doc_length_list)
            self.V = len(self.i2w)
            self.D = len(self.docs)
        return self

    def fit_transform(self, path: str) -> list:
        self.fit(path=path)
        return self.docs

    def transform(self, doc: str) -> list:
        return [self.w2i[w] for w in doc.split()]

    def get_documents(self) -> list:
        return self.docs

    def get_document(self, doc_id: int) -> list:
        return self.docs[doc_id]

    def get_vocabulary(self) -> dict:
        return self.w2i

    def get_num_docs(self) -> int:
        return self.D

    def get_num_vocab(self) -> int:
        return self.V

    def get_word(self, word_id: int) -> int:
        return self.i2w[word_id]

    def get_ith_doc_len(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def get_doc_lengths(self) -> list:
        return self.doc_lengths
