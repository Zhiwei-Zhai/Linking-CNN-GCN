import numpy as np
import pickle as pkl
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def load_patch_label_pkl(caseDir):
  with open("{0}/patch.pkl".format(caseDir), 'rb') as f:
    patches = pkl.load(f, encoding='latin1')

  with open("{0}/Label.pkl".format(caseDir), 'rb') as f:
    labels=pkl.load(f, encoding='latin1')

  with open("{0}/ind.pkl".format(caseDir), 'rb') as f:
    inds= pkl.load(f, encoding='latin1')

  with open("{0}/graph.pkl".format(caseDir), 'rb') as f:
    N_inds = pkl.load(f, encoding='latin1')

  patches = np.asarray(patches)  # size of N * 32 *32 *5
  mean, std = np.mean(patches, axis=(1,2,3), keepdims=True), np.std(patches, axis=(1,2,3), keepdims=True)
  # print('The average and STD is {} and {}'.format(mean, std))
  patches -= mean
  patches /= std
  labels = np.asarray(labels)
  return patches.astype(np.float16), labels.astype(np.int8), np.asarray(inds,dtype=np.int8), N_inds

def load_patch_label_npy(caseDir, normalization='gnormalization'):
  patch_file = "{0}/patch.npy".format(caseDir)
  patches = np.load( patch_file )

  Label_file = "{0}/Label.npy".format(caseDir)
  labels = np.load( Label_file )

  ind_file = "{0}/ind.npy".format(caseDir)
  inds= np.load( ind_file )

  N_inds_file = "{0}/graph.npy".format(caseDir)
  N_inds = np.load( N_inds_file ).item()  # the graph was a dictionary, saved with numpy.save

  patches = np.asarray(patches)
  if normalization is 'gnormalization':
    mean, std = np.mean(patches), np.std(patches)
  else:
    mean, std = np.mean(patches, axis=(1, 2, 3), keepdims=True), np.std(patches, axis=(1, 2, 3), keepdims=True)

  patches -= mean
  patches /= std
  labels = np.asarray(labels)
  return patches.astype(np.float32), labels.astype(np.int8), np.asarray(inds, dtype=np.uint32), N_inds

def list_to_ndarray(inlist):  # Constrain a list to numpy array with the minimum neighbor number
  len = 10
  for ele in inlist:
    len = np.minimum(ele.shape[0], len)
  outlist = []
  for ele in inlist:
    selet = np.random.choice( ele.shape[0], len )
    outlist.append( ele[selet] )
  return np.asarray( outlist )

class graphDataset(object):
  def __init__(self,
               patches,
               labels,
               ind_list,
               N_ind_list, randomSort = True, Num_neighbor = 3, Dir = './result/', name='none'
               ):

    if patches.shape[0] != labels.shape[0]:
      raise TypeError('The number of patches {0} is miss-matching with the number of labels {1}'.format(patches.shape[0], labels.shape[0]))
    self._patches = patches
    self._inds = ind_list
    self.path = Dir
    self.name = name
    self._n_inds = N_ind_list # neighbor index list
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self.Num_neighbor = Num_neighbor
    if np.nan in labels:
      Slct = np.extract( labels[:,0] != np.nan, np.arange( labels.shape[0] ) )
      self.num_nodes = len(Slct)
      self.patches = patches[Slct, :] #collect the patches with labels
      self.labels = labels[Slct, :]
      self.inds = ind_list[Slct]
    else:
      self.num_nodes = patches.shape[0]
      self.patches = patches
      self.labels = labels
      self.inds = ind_list
    if randomSort :  # random sort the patches, labels and the corresponding inds
      self.shuffle_nodes()

  def Pick_neighbors(self, n_indx):
    if len(n_indx) >= self.Num_neighbor:  # select fix number of Neighbor,
      n_indx = np.random.permutation(n_indx)
      L_indx = n_indx[0:self.Num_neighbor]
    else:
      L_indx = n_indx
      L_indx.append(np.random.choice(n_indx, self.Num_neighbor - len(n_indx)))
    return L_indx

  def shuffle_nodes(self):
    Ind_rand = np.arange(self.num_nodes)
    np.random.shuffle(Ind_rand)
    self.patches = self.patches[Ind_rand, :]
    self.labels = self.labels[Ind_rand, :]
    self.inds = self.inds[Ind_rand]

  def next_node(self, node_num, WithNeighbor=False):
    start = self._index_in_epoch
    if WithNeighbor:  # if collecting the neighbor patches
      if start + node_num > self.num_nodes:
        self._epochs_completed +=1
        rest_num_examples = self.num_nodes - start
        images_rest_part = self.patches[start:self.num_nodes]
        labels_rest_part = self.labels[start:self.num_nodes]
        ind_rest_part = self.inds[start:self.num_nodes]
        # colletec the Neighbor patches
        start = 0
        self._index_in_epoch = node_num-rest_num_examples
        end = self._index_in_epoch
        images_new_part = self.patches[start:end]
        labels_new_part = self.labels[start:end]
        ind_new_part = self.inds[start:end]
        ind_node = np.concatenate((ind_rest_part,ind_new_part), axis=0)

        N_image = []
        for k in range(node_num):
          try:
            n_indx = []
            for x in self._n_inds[ ind_node[k] ]:
              xtup = np.where(self._inds == x)
              if len(xtup) != 0:
                n_indx.append( xtup[0][0] )

            L_indx = self.Pick_neighbors(n_indx)      # pick fixed number of neighbors
            N_image.append( self._patches[ L_indx ] )
          except:
            N_image.append([])
        return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
          (labels_rest_part, labels_new_part), axis=0), np.asarray(N_image, dtype=np.float16)
      else: #
        self._index_in_epoch +=node_num
        end = self._index_in_epoch
        images = self.patches[start:end]
        labels = self.labels[start:end]
        ind_node = self.inds[start:end]

        N_image = []
        for k in range(node_num):
          try:
            n_indx = []
            for x in self._n_inds[ind_node[k]]:
              xtup = np.where(self._inds == x)
              if len(xtup) != 0:
                n_indx.append(xtup[0][0])

            L_indx = self.Pick_neighbors(n_indx)
            N_image.append(self._patches[L_indx])
          except:
            N_image.append([])
        # return  images, labels, list_to_ndarray(N_image) #np.asarray(N_image, dtype=np.float16)
        return images, labels, np.asarray(N_image, dtype=np.float16)
    else: # if not collecting the neighbor patches
      if start + node_num > self.num_nodes:
        self._epochs_completed += 1
        rest_num_examples = self.num_nodes - start
        images_rest_part = self.patches[start:self.num_nodes]
        labels_rest_part = self.labels[start:self.num_nodes]

        start = 0
        self._index_in_epoch = node_num - rest_num_examples
        end = self._index_in_epoch
        images_new_part = self.patches[start:end]
        labels_new_part = self.labels[start:end]

        return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
          (labels_rest_part, labels_new_part), axis=0)
      else:
        self._index_in_epoch += node_num
        end = self._index_in_epoch
        images = self.patches[start:end]
        labels = self.labels[start:end]

        return images, labels


def load_data( path = 'path', case_name = 'case_name', Num_neighbor=2, shuffel=True, nornalization='gnormalization'):
  path_data, Case_n = path, case_name

  train_data = []
  for i in range(len(Case_n)):
    patient_i = '{0}/{1}'.format(path_data, Case_n[i])

    [xPatches, ylabel, inds, n_inds] = load_patch_label_npy( caseDir=patient_i, normalization=nornalization)
    train_data.append( graphDataset(xPatches, ylabel, inds, n_inds, randomSort=shuffel, Num_neighbor=Num_neighbor) )

  return train_data

def DataGenerator(Graphes, batch_sz, withNeighbor = False):
  Num_graph = len(Graphes)
  indx=0

  while 1:
    i = indx % Num_graph
    indx +=1
    if withNeighbor:
      X_batch, Y_batch, NX_batch = Graphes[i].next_node(node_num=batch_sz, WithNeighbor=withNeighbor) #NX_batch: [b, N, Patchsz]
      yield [np.expand_dims(X_batch, 4), NX_batch], Y_batch
    else:
      X_batch, Y_batch = Graphes[i].next_node(node_num=batch_sz, WithNeighbor=withNeighbor)
      yield np.expand_dims(X_batch, 4), Y_batch


if __name__ == "__main__":
  path_data = '/exports/lkeb-hpc/zzhai/Data/Artery_Vein/Orient/'
  Cases = [
  '/AV_challengeav01Ncontrast_HRes/right/',
  '/AV_challengeav01Ncontrast_HRes/left/',
  '/AV_challengeav02Ncontrast_HRes/right',
  '/AV_challengeav02Ncontrast_HRes/left'
  ]
  Case_train = [ Cases[1]]
  # Case_train = [ Cases[1], Cases[2], Cases[3]]
  # Case_val= [ Cases[0]  ]
  # Case_test= [ Cases[0]  ]
  traingraphes = load_data(path_data, Case_train, Num_neighbor=2)
  for graph in traingraphes:
    for i in range(100):
      X_batch, Y_batch, NX_batch = graph.next_node(node_num=3, WithNeighbor=True)
      print(NX_batch.shape)
