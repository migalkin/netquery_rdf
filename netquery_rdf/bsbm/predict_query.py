from netquery_rdf.data_utils import load_queries_by_formula, load_test_queries_by_formula, load_queries
from netquery_rdf.bsbm.data_utils import load_graph
import torch
from netquery_rdf.model import QueryEncoderDecoder
from netquery_rdf.utils import *
from netquery_rdf.utils import eval_auc_queries, eval_perc_queries
import pickle

def predict_query():
    graph, feature_modules, node_maps = load_graph("./bsbm_data", 128)
    out_dims = {mode: 128 for mode in graph.relations}

    enc = get_encoder(0, graph, out_dims, feature_modules, False)
    dec = get_metapath_decoder(graph, out_dims, 'bilinear')
    inter_dec = get_intersection_decoder(graph, out_dims, 'mean')

    model = QueryEncoderDecoder(graph, enc, dec, inter_dec)
    model.load_state_dict(torch.load("bsbm_data-0-128-0.010000-bilinear-mean.log-edge_conv", map_location='cpu'))
    model.eval()

    test_queries = load_test_queries_by_formula("./bsbm_data/bsbm_queries_test_new.pkl")
    auc, rel_aucs = eval_auc_queries(test_queries['one_neg']['1-chain'], model)
    print(auc, rel_aucs)
    # for q in test_queries:
    #     print(model.forward(q))

def clean_from_empty_negs():
    files = ["./bsbm_data/val_edges.pkl", "./bsbm_data/test_edges.pkl"]
    for f in files:
        cleaned = []
        dump = pickle.load(open(f, "rb"))
        for d in dump:
            if len(d[1])>0:
                cleaned.append(d)
        pickle.dump(cleaned, open(f+"_cleaned","wb"), protocol=pickle.HIGHEST_PROTOCOL)

def clean_complex_queries():
    q2 = load_queries("./bsbm_data/val_queries_2.pkl")
    test = q2[800:]
    val = q2[:800]
    pickle.dump(test, open("./bsbm_data/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(val, open("./bsbm_data/val_queries_2_upd.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    q3 = load_queries("./bsbm_data/val_queries_3.pkl")
    test3 = q3[800:]
    val3 = q3[:800]
    pickle.dump(test3, open("./bsbm_data/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(val3, open("./bsbm_data/val_queries_3_upd.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def sample_train23():
    q2 = load_queries("./bsbm_data/train_queries_2.pkl")
    q3 = load_queries("./bsbm_data/train_queries_3.pkl")

    q2_reduced = random.sample(q2, 10000)
    q3_reduced = random.sample(q3, 10000)

    pickle.dump(q2_reduced, open("./bsbm_data/train_queries_2_reduced.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(q3_reduced, open("./bsbm_data/train_queries_3_reduced.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Done")

if __name__=="__main__":
    #clean_from_empty_negs()
    #clean_complex_queries()
    sample_train23()