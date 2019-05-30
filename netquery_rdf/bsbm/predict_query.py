from netquery_rdf.data_utils import load_queries_by_formula, load_test_queries_by_formula
from netquery_rdf.bsbm.data_utils import load_graph
import torch
from netquery_rdf.model import QueryEncoderDecoder
from netquery_rdf.utils import *
from netquery_rdf.utils import eval_auc_queries, eval_perc_queries

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

if __name__=="__main__":
    predict_query()