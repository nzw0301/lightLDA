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
            for l in f:
                doc = []
                for w in l.split():
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

    def fit_transform(self, path: str):
        self.fit(path=path)
        return self.docs

    def transform(self, doc: str):
        return [self.w2i[w] for w in doc.split()]

    def get_documents(self):
        return self.docs

    def get_document(self, doc_id: int):
        return self.docs[doc_id]

    def get_vocabulary(self):
        return self.w2i

    def get_nb_docs(self):
        return self.D

    def get_nb_vocab(self):
        return self.V

    def get_word(self, word_id: int):
        return self.i2w[word_id]

    def get_minibatch(self, nb_minibatch: int):
        return [self.get_document(doc_id=doc_id)
                for doc_id in np.random.randint(low=0,
                                                high=self.D,
                                                size=nb_minibatch)]

    def get_ith_doc_len(self, doc_id: int):
        return self.doc_lengths[doc_id]

    def get_doc_lengths(self):
        return self.doc_lengths

if __name__ == '__main__':
    d = Document()
    print(d.fit_transform("sample.txt"))
    print(d.get_doc_lengths())
    print(d.get_ith_doc_len(0))
    print(d.get_vocabulary())
    print(d.get_nb_docs())
    print(d.get_nb_vocab())
