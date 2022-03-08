'''
Created by trangvu on 21/01/21
'''
import argparse
import logging
import numpy as np
import faiss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train_kmeans(x, k, niter=100, nredo=1, verbose=1):
  """
      Runs k-means clustering on one or several GPUs
      """
  assert np.all(~np.isnan(x)), 'x contains NaN'
  assert np.all(np.isfinite(x)), 'x contains Inf'

  ngpus = faiss.get_num_gpus()
  print("number of GPUs:", ngpus)

  if ngpus > 0:
    gpu_ids = range(ngpus)
  else:
    gpu_ids = None

  d = x.shape[1]
  # kmeans = faiss.Kmeans(d=d, k=k, niter=niter, verbose=verbose, nredo=nredo, gpu=True)
  kmeans = faiss.Clustering(d, k)
  kmeans.verbose = bool(verbose)
  kmeans.niter = niter
  kmeans.nredo = nredo

  # otherwise the kmeans implementation sub-samples the training set
  kmeans.max_points_per_centroid = 10000000

  if gpu_ids is not None:
    res = [faiss.StandardGpuResources() for i in gpu_ids]

    flat_config = []
    for i in gpu_ids:
      cfg = faiss.GpuIndexFlatConfig()
      cfg.useFloat16 = False
      cfg.device = i
      flat_config.append(cfg)

    if len(gpu_ids) == 1:
      index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
      indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                 for i in range(len(gpu_ids))]
      index = faiss.IndexProxy()
      for sub_index in indexes:
        index.addIndex(sub_index)
  else:
    index = faiss.IndexFlatL2(d)

  kmeans.train(x, index)

  # Centroids after clustering
  centroids = faiss.vector_float_to_array(kmeans.centroids)
  print("centroids shape {}".format(centroids.shape))

  return centroids.reshape(k, d)

def compute_cluster_assignment(centroids, x):
  assert centroids is not None, "should train before assigning"
  d = centroids.shape[1]
  index = faiss.IndexFlatL2(d)
  index.add(centroids)
  distances, labels = index.search(x, 1)
  return labels.ravel()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True, help="npy files that store vector representation")
  parser.add_argument("--output", required=True, help="output the cluster labels")
  parser.add_argument('--k', default=5, type=int, help="number of clusters")
  args = parser.parse_args()

  k = args.k

  data = np.load(args.data)
  clusters = [[] for _ in range(k)]

  centroids = train_kmeans(data, k)
  labels = compute_cluster_assignment(centroids, data)

  with open(args.output, 'w', encoding='utf-8') as fout:
    for id, cluster in enumerate(labels):
      fout.write("{}\n".format(cluster))
      clusters[cluster].append(id)

  print("Finish clusering {} into {} clusters".format(args.data, args.k))
  print("Saving to {}".format(args.output))

  # print("Cluster results: ")
  # print(clusters)
  # print("Centroids :")
  # print(centroids)


if __name__ == "__main__":
  main()