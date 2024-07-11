import numpy as np
class Code:
    def __init__(self,H_filename):
        self.load_code(H_filename)
        
    def gf2elim(self,M):
          m,n = M.shape
          i=0
          j=0
          record_col_exchange_index = []
          while i < m and j < n:
              #print(M)
              # find value and index of largest element in remainder of column j
              if np.max(M[i:, j]):
                  k = np.argmax(M[i:, j]) +i
            # swap rows
                  #M[[k, i]] = M[[i, k]] this doesn't work with numba
                  if k !=i:
                      temp = np.copy(M[k])
                      M[k] = M[i]
                      M[i] = temp              
              else:
                  if not np.max(M[i, j:]):
                      M = np.delete(M,i,axis=0) #delete a all-zero row which is redundant
                      m = m-1  #update according info
                      continue
                  else:
                      column_k = np.argmax(M[i, j:]) +j
                      temp = np.copy(M[:,column_k])
                      M[:,column_k] = M[:,j]
                      M[:,j] = temp
                      record_col_exchange_index.append((j,column_k))
          
              aijn = M[i, j:]
              col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
              col[i] = 0 #avoid xoring pivot row with itself
              flip = np.outer(col, aijn)
              M[:, j:] = M[:, j:] ^ flip
              i += 1
              j +=1
          return M,record_col_exchange_index
          
    def generator_matrix(self,parity_check_matrix):
          # H assumed to be full row rank to obtain its systematic form
          tmp_H = np.copy(parity_check_matrix)
          #reducing into row-echelon form and record column 
          #indices involved in swapping
          row_echelon_form,record_col_exchange_index = self.gf2elim(tmp_H)
          H_shape = row_echelon_form.shape
          # H is reduced into [I H_2]
          split_H = np.hsplit(row_echelon_form,(H_shape[0],H_shape[1])) 
          #Generator matrix in systematic form [H_2^T I] in GF(2)
          G1 = split_H[1].T
          G2 = np.identity(H_shape[1]-H_shape[0],dtype=int)
          G = np.concatenate((G1,G2),axis=1)
          #undo the swapping of columns in reversed order
          for i in reversed(range(len(record_col_exchange_index))):
              temp = np.copy(G[:,record_col_exchange_index[i][0]])
              G[:,record_col_exchange_index[i][0]] = \
                  G[:,record_col_exchange_index[i][1]]
              G[:,record_col_exchange_index[i][1]] = temp
          #verify ths syndrome equal to all-zero matrix
          Syndrome_result = parity_check_matrix.dot(G.T)%2
          if np.all(Syndrome_result==0):
            print("That's it, generator matrix created successfully with shape:",G.shape)
          else:
            print("Something wrong happened, generator matrix failed to be valid")     
          return G
    def load_code(self,H_filename):
    	# parity-check matrix; Tanner graph parameters
    	# H_filename = format('./LDPC_matrix/LDPC_576_432.alist')
    	# G_filename = format('./LDPC_matrix/LDPC_576_432.gmat')
        with open(H_filename,'rt') as f:
            line= str(f.readline()).strip('\n').split(' ')
    		# get n and m (n-k) from first line
            n,m = [int(s) for s in line]
            #assigned manually for redundant check matrix otherwise
           
    #################################################################################################################
            var_degrees = np.zeros(n).astype(int) # degree of each variable node
            chk_degrees = np.zeros(m).astype(int) # degree of each check node
    
    		# initialize H
            H = np.zeros([m,n]).astype(int)
            line =  str(f.readline()).strip('\n').split(' ')
            max_var_degree, max_chk_degree = [int(s) for s in line]
            line =  str(f.readline()).strip('\n').split(' ')
           # var_degree_dist = [int(s) for s in line[0:-1]] 
            line =  str(f.readline()).strip('\n').split(' ')
           # chk_degree_dist = [int(s) for s in line[0:-1]]
    
            var_edges = [[] for _ in range(0,n)]
            for i in range(0,n):
                line =  str(f.readline()).strip('\n').split(' ')
                var_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
                var_degrees[i] = len(var_edges[i])
                H[var_edges[i], i] = 1
      
            chk_edges = [[] for _ in range(0,m)]
            for i in range(0,m):
                line =  str(f.readline()).strip('\n').split(' ')
                chk_edges[i] = [(int(s)-1) for s in line if s not in ['0','']]
                chk_degrees[i] = len(chk_edges[i])
      
    ################################################################################################################
    # numbering each edge in H with a unique number whether horizontally or vertically       
            d = [[] for _ in range(0,n)]
            edge = 0
            for i in range(0,n):
                for j in range(0,var_degrees[i]):
                    d[i].append(edge)
                    edge += 1
      
            u = [[] for _ in range(0,m)]
            edge = 0
            for i in range(0,m):
                for j in range(0,chk_degrees[i]):
                    v = chk_edges[i][j]
                    for e in range(0,var_degrees[v]):
                        if (i == var_edges[v][e]):
                            u[i].append(d[v][e])          
        self.H = H
        self.max_chk_degree = max_chk_degree
        self.check_matrix_column = n
        self.check_matrix_row = m  
        self.G = self.generator_matrix(self.H) 
        #effective number of message bits in a codeword        
        self.k = self.G.shape[0]   