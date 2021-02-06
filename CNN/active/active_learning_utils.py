import numpy as np
import pdb

def unshuffle_indices(document_ids, queried_ids):
  new_idxs = []
  for query_id in queried_ids:
    idx = np.where(document_ids, query_id)
    new_idxs.append(idx)
  return new_idxs

def perform_random_query(unqueried, num_to_query):
  idxs_queried = np.random.choice(unqueried, num_to_query, replace=False)
  return idxs_queried
  
def make_first_query(full_train_size, num_to_start):
  unqueried_idxs = np.arange(int(full_train_size))
  queried_idxs   = perform_random_query(unqueried_idxs, num_to_start)
  unqueried_idxs = update_unqueried_list(unqueried_idxs, queried_idxs)
  return unqueried_idxs, queried_idxs

def update_unqueried_list(unqueried, newly_queried):
  unqueried_idxs = np.setdiff1d(unqueried, newly_queried) #np.delete uses indices
  return unqueried_idxs

def get_idxs_to_query(num_total_queries, num_queries_complete,
                      num_to_query, unqueried):
  if num_queries_complete == num_total_queries:
    print("Performed all queries")
    idx_to_query= None
  elif num_queries_complete == (num_total_queries - 1):
    print("Performing the final query")
    idx_to_query = unqueried
  else:
    print("Choosing {} documents to query".format(num_to_query))
    idx_to_query = perform_random_query(unqueried, num_to_query)
  return idx_to_query
