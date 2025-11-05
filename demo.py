# import spacy
# nlp = spacy.load("en_coreference_web_trf")
# nlp.add_pipe("coreferee")
# doc = nlp("Riya told Surya that she will send him the PDF.")
# for chain in doc._.coref_chains:
#     print(chain)

from qdrant_client import QdrantClient
import atexit

client = QdrantClient(path="./qdrant_local")

@atexit.register
def close_qdrant():
    try:
        client.close()
    except Exception:
        pass  # Avoid msvcrt import warning
